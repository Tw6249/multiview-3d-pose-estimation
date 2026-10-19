import copy
import importlib.util
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from motion_control.recovery_rollout import (RecoveryRollout,load_config,scheduled_reference,
    snapshot,branch,boundary_reference,assess_history)
from motion_control.twist import DEFAULT_REF


class RecoveryContractTests(unittest.TestCase):
    def test_reject_live_profile_and_invalid_ranges(self):
        for key,value in [('hardware_enabled',True),('online_eligible',True),('control_dt_seconds',.01),
                          ('candidates',3),('device','auto'),('reference_rate_per_second',[1]*34),
                          ('pose_blend_range_seconds',[0,1]),('entry_feedback_mix_range',[0,2]),
                          ('horizon_seconds',1),('terminal_window_seconds',float('nan'))]:
            cfg=load_config();cfg[key]=value
            with self.assertRaises(ValueError,msg=key):RecoveryRollout(cfg)

    def test_never_accept_device_clock_or_boolean_timestamp(self):
        planner=RecoveryRollout()
        for stamp,clock in [(0,'robot'),(True,'offline_simulation'),(-1,'offline_simulation')]:
            with self.assertRaises(ValueError):planner.plan(None,None,DEFAULT_REF,stamp,clock)

    def test_candidate_grid_is_reproducible_and_inside_bounds(self):
        p=RecoveryRollout();a=p.candidate_parameters();np.testing.assert_array_equal(a,p.candidate_parameters())
        self.assertEqual(a.shape,(32,5))
        keys=('delay_range_seconds','pose_blend_range_seconds','brake_range_seconds','entry_feedback_mix_range','velocity_gain_range')
        for i,key in enumerate(keys):self.assertTrue(np.all((a[:,i]>=p.cfg[key][0])&(a[:,i]<=p.cfg[key][1])))

    def test_withdrawal_has_bounded_end_and_preserves_input(self):
        entry=DEFAULT_REF.copy();entry[0]=.5;entry[9]=1.;old=entry.copy();params=[.2,1.,.3,1.,1.]
        np.testing.assert_array_equal(scheduled_reference(0.,entry,params),entry)
        np.testing.assert_array_equal(scheduled_reference(2.,entry,params),DEFAULT_REF)
        self.assertEqual(scheduled_reference(.31,entry,params)[0],0.)
        np.testing.assert_array_equal(entry,old)

    def test_stationary_fall_or_intermediate_fall_cannot_pass(self):
        cfg=load_config();standing=np.tile([.78,.01,.01,.02],(200,1))
        self.assertTrue(assess_history(standing,cfg)['feasible'])
        fallen=standing.copy();fallen[:,0]=.1
        self.assertFalse(assess_history(fallen,cfg)['feasible'])
        standing[50,1]=.7
        self.assertFalse(assess_history(standing,cfg)['feasible'])

    def test_missing_or_nonfinite_dynamics_cannot_be_scored_as_feasible(self):
        for h in ([],[[.8,0,0,0]],np.tile([.8,0,float('nan'),0],(200,1))):
            with self.assertRaises(ValueError):assess_history(h,load_config())


@unittest.skipUnless(importlib.util.find_spec('mujoco'),'Requires robot environment')
class RecoverySimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from motion_control.twist import TwistPlant,TwistPolicy
        cls.plant=TwistPlant();cls.policy=TwistPolicy()
        for _ in range(50):cls.plant.step(cls.policy.step(DEFAULT_REF,*cls.plant.state()))

    def setUp(self):self.p,self.pol=branch(self.plant,self.policy,snapshot(self.plant))

    def test_branches_clone_integration_and_policy_without_mutating_parent(self):
        before=snapshot(self.p);old=[x.copy() for x in self.pol.history]
        a,ap=branch(self.p,self.pol,before);b,bp=branch(self.p,self.pol,before)
        for _ in range(5):
            a.step(ap.step(DEFAULT_REF,*a.state()));b.step(bp.step(DEFAULT_REF,*b.state()))
        np.testing.assert_array_equal(snapshot(a),snapshot(b));np.testing.assert_array_equal(snapshot(self.p),before)
        np.testing.assert_array_equal(self.pol.history,old)
        # Compare branch to the original's next ticks, including warmstart state.
        for _ in range(5):self.p.step(self.pol.step(DEFAULT_REF,*self.p.state()))
        np.testing.assert_array_equal(snapshot(a),snapshot(self.p))

    def test_feedback_velocity_uses_heading_not_tilted_body(self):
        r=Rotation.from_euler('xyz',[.2,-.1,.8]);self.p.data.qpos[3:7]=r.as_quat()[[3,0,1,2]]
        self.p.data.qvel[:3]=[.3,-.2,.1];before=snapshot(self.p)
        out=boundary_reference(self.p,DEFAULT_REF,1.)
        expected=Rotation.from_euler('z',.8).as_matrix().T@self.p.data.qvel[:3]
        np.testing.assert_allclose(out[:2],expected[:2]);np.testing.assert_array_equal(snapshot(self.p),before)

    def tiny_planner(self):
        cfg=load_config();cfg.update(device='cpu',candidates=4,horizon_seconds=.12,terminal_window_seconds=.02,
            delay_range_seconds=[0,0],pose_blend_range_seconds=[.02,.02],brake_range_seconds=[.02,.02])
        return RecoveryRollout(cfg)

    def test_planning_is_robot_only_nonmutating_and_cpu_verified(self):
        planner=self.tiny_planner();before=snapshot(self.p);old=copy.deepcopy(self.pol.history)
        plan,diag=planner.plan(self.p,self.pol,DEFAULT_REF,1_000_000_000)
        self.assertIsNotNone(plan);self.assertTrue(diag['cpu_verifications'][0]['feasible'])
        self.assertFalse(diag['source_future_used']);self.assertFalse(diag['online_eligible'])
        np.testing.assert_array_equal(snapshot(self.p),before);np.testing.assert_array_equal(self.pol.history,old)
        # Existing loaded feet at t=0 are not falsely recorded as new landings.
        self.assertEqual(diag['candidates'][0]['landings'],[])

    def test_prediction_cannot_override_failed_actual_controller_verification(self):
        planner=self.tiny_planner()
        with patch.object(planner,'verify',return_value={'feasible':False}):
            plan,diag=planner.plan(self.p,self.pol,DEFAULT_REF,0)
        self.assertIsNone(plan);self.assertFalse(diag['accepted']);self.assertGreater(len(diag['cpu_verifications']),0)

    def test_already_fallen_state_does_not_load_or_run_policy_search(self):
        planner=self.tiny_planner();self.p.data.qpos[2]=.1
        plan,diag=planner.plan(self.p,self.pol,DEFAULT_REF,0)
        self.assertIsNone(plan);self.assertEqual(diag['candidate_count'],0);self.assertIsNone(planner.policy)

    def test_model_sensitivity_never_modifies_nominal_physics(self):
        from scripts.benchmark_recovery_sensitivity import altered_model
        mass=self.p.model.body_mass.copy();inertia=self.p.model.body_inertia.copy();friction=self.p.model.geom_friction.copy()
        changed=altered_model(self.p,self.pol,1.1,.5)
        np.testing.assert_array_equal(self.p.model.body_mass,mass);np.testing.assert_array_equal(self.p.model.body_inertia,inertia)
        np.testing.assert_array_equal(self.p.model.geom_friction,friction)
        np.testing.assert_allclose(changed.model.body_mass,mass*1.1);np.testing.assert_allclose(changed.model.body_inertia,inertia*1.1)
        np.testing.assert_allclose(changed.model.geom_friction[:,0],friction[:,0]*.5)
        np.testing.assert_array_equal(snapshot(self.p),snapshot(changed));self.assertNotEqual(changed.sha256,self.p.sha256)


if __name__=='__main__':unittest.main()
