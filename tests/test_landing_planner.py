import copy
import importlib.util
import unittest
import numpy as np
from motion_control.landing_planner import LandingPlanner,load_config,propagate
from motion_control.root_tracking import planar_rotation
from pose_sim.source_phase import SourceLandingWindow


class SourcePhaseTests(unittest.TestCase):
    def test_invalid_history_or_time_window_fails_closed(self):
        for key,value in [('history_samples',1),('maximum_remaining_s',.001),('landing_time_uncertainty_s',float('nan'))]:
            cfg=load_config()['source_phase'];cfg[key]=value
            with self.assertRaises(ValueError):SourceLandingWindow(cfg)
    def scenario(self,ascending=False):
        estimator=SourceLandingWindow(load_config()['source_phase']);outputs=[]
        for i in range(5):
            t=i*40_000_000;z=.3+(.5 if ascending else -.5)*i*.04
            p=dict(left_ankle=[.2,0,z],right_ankle=[0,0,0])
            s=dict(timestamp_ns=t,feet=dict(left=dict(status='swing',ankle_height_above_support_proxy_m=z),right=dict(status='support_candidate')))
            outputs.append(estimator.update(t,p,1.,s))
        return estimator,outputs,p,s

    def test_window_from_same_side_descent_not_alternating_clock(self):
        _,out,_,_=self.scenario();self.assertTrue(all(x['status']=='unknown' for x in out[:4]))
        e=out[-1]['event'];self.assertEqual(e['side'],'left');self.assertEqual(e['stance_side'],'right')
        np.testing.assert_allclose(e['ankle_velocity_source_world_m_s'],[0,0,-.5],atol=1e-12)
        self.assertLessEqual(e['earliest_source_ns'],600_000_000);self.assertGreaterEqual(e['latest_source_ns'],600_000_000)

    def test_ascending_missing_and_gaps_do_not_create_landing(self):
        self.assertEqual(self.scenario(True)[1][-1]['status'],'unknown')
        estimator,_,p,s=self.scenario();s['timestamp_ns']=200_000_000
        self.assertEqual(estimator.update(200_000_000,p,1.,s,current=False)['status'],'unknown')
        s['timestamp_ns']=500_000_000
        self.assertEqual(estimator.update(500_000_000,p,1.,s)['status'],'unknown')

    def test_unconfirmed_stance_and_duplicate_source_are_rejected(self):
        estimator,_,p,s=self.scenario();s['timestamp_ns']=200_000_000;s['feet']['right']['status']='unknown'
        self.assertEqual(estimator.update(200_000_000,p,1.,s)['status'],'unknown')
        with self.assertRaises(ValueError):estimator.update(200_000_000,p,1.,s)

    def test_low_descending_swing_can_remain_known_without_inventing_stance(self):
        estimator=SourceLandingWindow(load_config()['source_phase'])
        for i in range(5):
            t=i*40_000_000;z=.12-.5*i*.04
            p=dict(left_ankle=[.2,0,z],right_ankle=[0,0,0])
            support=dict(timestamp_ns=t,feet=dict(left=dict(status='unknown',ankle_height_above_support_proxy_m=z),right=dict(status='support_candidate')))
            result=estimator.update(t,p,1.,support)
        self.assertEqual(result['status'],'hypothesis');self.assertEqual(result['event']['side'],'left')
        self.assertLess(result['event']['latest_source_ns']-t,150_000_000)


class LandingPlannerTests(unittest.TestCase):
    def test_hardware_and_malformed_optimizer_bounds_are_rejected(self):
        for key,value in [('hardware_enabled',True),('minimum_landing_time_s',1.),('solver_max_iterations',0),('constraint_tolerance',float('nan'))]:
            cfg=load_config();cfg[key]=value
            with self.assertRaises(ValueError):LandingPlanner('test',cfg)
    def packets(self):
        robot=dict(timestamp_ns=0,clock_domain='offline_simulation',target_model_sha256='test',
            com_world_m=[0,0,.7],com_velocity_world_m_s=[.2,0,0],angular_momentum_world_kg_m2_s=[0,0,0],mass_kg=33.,gravity_m_s2=9.81,
            loaded_contact_points_by_foot_world_m=dict(right=[[-.1,-.05,0],[.1,-.05,0],[.1,.05,0],[-.1,.05,0]],left=[]))
        event=dict(phase='observed_descending_swing',side='left',stance_side='right',source_timestamp_ns=0,observation_robot_ns=0,
            earliest_robot_ns=100_000_000,latest_robot_ns=250_000_000,nominal_robot_ns=180_000_000,
            nominal_landing_world_xy_m=[.08,0],desired_world_velocity_m_s=[.2,0],target_heading_rad=0.)
        return robot,event

    def test_exact_flow_and_momentum_sign_match_centroidal_force_balance(self):
        c=np.array([.1,.2]);v=np.array([.2,-.1]);m=33.;g=9.81;h=.7;rate=np.array([2.,-3.]);p=np.array([0,.02]);w=np.sqrt(g/h)
        cmp=p+np.array([rate[1],-rate[0]])/(m*g)
        _,vn=propagate(c,v,cmp,w,1e-7);acc=(vn-v)/1e-7
        force=m*np.r_[acc,g];torque=np.cross(np.r_[p-c,-h],force)
        np.testing.assert_allclose(torque[:2],rate,atol=3e-6)

    def test_feasible_plan_preserves_side_window_corridor_and_constraints(self):
        robot,event=self.packets();result=LandingPlanner('test').solve(robot,event,0)
        self.assertEqual(result['status'],'model_feasible',result)
        p=result['plan'];self.assertTrue(.1<=p['landing_after_s']<=.25)
        self.assertLessEqual(np.max(np.abs(np.array(p['landing_world_xy_m'])-event['nominal_landing_world_xy_m'])),.080001)
        self.assertLess(p['equality_residual'],1e-5);self.assertLess(p['inequality_violation'],1e-5)
        np.testing.assert_allclose(np.array(p['angular_momentum_at_landing_world_kg_m2_s'])+.15*np.array(p['angular_momentum_rate_after_world_nm']),[0,0],atol=1e-10)

    def test_world_translation_and_yaw_preserve_physics(self):
        robot,event=self.packets();base=LandingPlanner('test').solve(robot,event,0)['plan'];R=planar_rotation(.8);shift=np.array([20,-15])
        robot['com_world_m'][:2]=(R@robot['com_world_m'][:2]+shift).tolist();robot['com_velocity_world_m_s'][:2]=(R@robot['com_velocity_world_m_s'][:2]).tolist()
        robot['loaded_contact_points_by_foot_world_m']['right']=[[*((R@p[:2])+shift),p[2]] for p in robot['loaded_contact_points_by_foot_world_m']['right']]
        event['nominal_landing_world_xy_m']=(R@event['nominal_landing_world_xy_m']+shift).tolist();event['desired_world_velocity_m_s']=(R@event['desired_world_velocity_m_s']).tolist();event['target_heading_rad']=.8
        p=LandingPlanner('test').solve(robot,event,0)['plan'];self.assertIsNotNone(p)
        np.testing.assert_allclose(p['landing_world_xy_m'],R@base['landing_world_xy_m']+shift,atol=2e-6)
        self.assertAlmostEqual(base['landing_after_s'],p['landing_after_s'])

    def test_unknown_expired_stale_or_wrong_side_cannot_become_new_step(self):
        robot,event=self.packets();planner=LandingPlanner('test')
        self.assertIsNone(planner.solve(robot,None,0)['plan'])
        for change in [dict(latest_robot_ns=-1),dict(observation_robot_ns=1),dict(stance_side='left')]:
            self.assertIsNone(planner.solve(robot,dict(event,**change),0)['plan'])
        self.assertIsNone(planner.solve(robot,event,100_000_000)['plan'])
        robot['target_model_sha256']='wrong'
        with self.assertRaises(ValueError):planner.solve(robot,event,0)

    def test_unreachable_velocity_cannot_pass_by_ignoring_friction(self):
        robot,event=self.packets();robot['com_velocity_world_m_s']=[10,0,0]
        r=LandingPlanner('test').solve(robot,event,0);self.assertEqual(r['status'],'no_feasible_candidate')

    def test_degenerate_stance_can_satisfy_fixed_capture_without_solver_rank_failure(self):
        robot,event=self.packets();robot['loaded_contact_points_by_foot_world_m']['right']=[[0,0,0]]
        robot['com_velocity_world_m_s']=[0,0,0];event['desired_world_velocity_m_s']=[0,0];event['nominal_landing_world_xy_m']=[0,0]
        self.assertEqual(LandingPlanner('test').solve(robot,event,0,mode='fixed')['status'],'model_feasible')


@unittest.skipUnless(importlib.util.find_spec('mujoco'),'Requires robot environment')
class AngularMomentumTests(unittest.TestCase):
    def test_centroidal_matrix_matches_sum_of_body_momenta_without_mutation(self):
        from motion_control.twist import TwistPlant
        from motion_control.support_feedback import SimulationSupportFeedback
        from motion_control.stop_transition import load_config as stop_config
        p=TwistPlant();p.data.qvel[:]=np.random.default_rng(42).normal(0,.1,p.model.nv);before=p.data.qvel.copy()
        adapter=SimulationSupportFeedback(p,stop_config());r=adapter.read(0);d=adapter.data;total=np.zeros(3)
        for body in range(1,p.model.nbody):
            jp=np.zeros((3,p.model.nv));jr=np.zeros_like(jp);p.mj.mj_jacBodyCom(p.model,d,jp,jr,body)
            R=d.ximat[body].reshape(3,3);inertia=R@np.diag(p.model.body_inertia[body])@R.T
            total+=inertia@(jr@d.qvel)+np.cross(d.xipos[body]-r['com_world_m'],p.model.body_mass[body]*(jp@d.qvel))
        np.testing.assert_allclose(total,r['angular_momentum_world_kg_m2_s'],atol=1e-12);np.testing.assert_array_equal(p.data.qvel,before)


if __name__=='__main__':unittest.main()
