"""Heterogeneous source-robot poses -> target tasks and G1 locomotion intents.

Task references are NEVER motor commands. Whole-body realization additionally
requires target proprioception, contact estimation and a validated controller.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from pose_sim.geometry import rigid_transform
from .contracts import timestamp,vector
from .replay import read_rows,write_rows
from .unitree import G1Description,locomotion_request


class TaskMapper:
    def __init__(self, config):
        self.cfg,self.anchors = config,{}
        if config['schema_version'] != 1:
            raise ValueError('Unsupported transfer schema')
        if config.get('required_source_perception') != 'markerless':
            raise ValueError('G1 transfer requires markerless source perception')
        tasks = [m['target_task'] for m in config['reference_mappings']]
        if len(tasks) != len(set(tasks)):
            raise ValueError('Target task has competing source mappings')
        for mapping in config['reference_mappings']:
            if not np.isfinite(mapping['translation_scale']) or mapping['translation_scale'] <= 0:
                raise ValueError('Translation scale must be finite and positive')
            rigid_transform(mapping['T_target_neutral'])
            t = np.eye(4); t[:3,:3] = mapping['R_target_source']; rigid_transform(t)
        self.last_timestamp = None
        self.source_binding = None

    def map(self, observation):
        if observation.get('perception_mode') != self.cfg['required_source_perception']:
            raise ValueError('Source perception does not meet the required markerless contract')
        if observation['source_id'] != self.cfg['source_id'] or observation['units'] != 'SI':
            raise ValueError('Source robot identity/units mismatch')
        binding = (observation['source_model_sha256'],observation['world_frame'],observation.get('calibration_sha256'),
                   observation.get('detector_model_sha256'),observation.get('perception_mode'))
        if self.source_binding is not None and binding != self.source_binding:
            raise ValueError('Source calibration/model changed; reinitialize mapping anchors')
        self.source_binding = binding
        ts = timestamp(observation['timestamp_ns'])
        if self.last_timestamp is not None and ts <= self.last_timestamp:
            raise ValueError('Nonmonotonic source pose')
        self.last_timestamp = ts
        tasks = {}
        for mapping in self.cfg['reference_mappings']:
            link = observation['links'].get(mapping['source_link'],{})
            if not link.get('valid',False) or not link.get('orientation_observed',False):
                tasks[mapping['target_task']] = {'valid':False,'reason':'source link unobserved'}
                continue
            t = rigid_transform(link['T_world_link'])
            anchor = self.anchors.setdefault(mapping['source_link'],t.copy())
            delta = np.linalg.inv(anchor)@t
            rotation = np.asarray(mapping['R_target_source'])
            aligned = np.eye(4)
            aligned[:3,:3] = rotation@delta[:3,:3]@rotation.T
            aligned[:3,3] = mapping['translation_scale']*(rotation@delta[:3,3])
            target = np.asarray(mapping['T_target_neutral'])@aligned
            tasks[mapping['target_task']] = {'valid':True,'T_reference_task':target.tolist()}
        return {'schema_version':1,'source_id':observation['source_id'],'target_id':self.cfg['target_id'],
                'timestamp_ns':ts,'reference_frame':'target_initial_base','tasks':tasks,
                'contact_state':'unknown','motor_command':False,
                'requires':'target feedback, contacts and validated whole-body controller'}


class LocomotionIntent:
    def __init__(self, config):
        self.cfg = config
        if config.get('required_source_perception') != 'markerless':
            raise ValueError('G1 transfer requires markerless source perception')
        self.previous = None
        self.source_stamp = None
        self.desired = np.zeros(3)
        self.velocity = np.zeros(3)
        self.last_tick = None
        self.last_seen = None
        self.source_binding = None
        self.reason = 'waiting for two source observations'
        self.limits = vector(config['velocity_limits'],3,'velocity limits')
        self.acceleration = vector(config['acceleration_limits'],3,'acceleration limits')
        self.deadband = vector(config['velocity_deadband'],3,'velocity deadband')
        if np.any(self.limits <= 0) or np.any(self.acceleration <= 0) or np.any(self.deadband < 0):
            raise ValueError('Invalid locomotion limits')
        for key in ('max_source_age_ms','max_source_speed_m_s','max_source_yaw_rate_rad_s'):
            if not np.isfinite(config[key]) or config[key] <= 0: raise ValueError('Invalid source timing/rate budget')

    def observe(self, observation):
        if observation.get('perception_mode') != self.cfg['required_source_perception']:
            raise ValueError('Source perception does not meet the required markerless contract')
        if observation['source_id'] != self.cfg['source_id'] or observation['units'] != 'SI':
            raise ValueError('Wrong source robot')
        binding = (observation['source_model_sha256'],observation['world_frame'],observation.get('calibration_sha256'),
                   observation.get('detector_model_sha256'),observation.get('perception_mode'))
        if self.source_binding is not None and binding != self.source_binding:
            raise ValueError('Source model or calibration changed')
        self.source_binding = binding
        ts = timestamp(observation['timestamp_ns'])
        if self.last_seen is not None and ts <= self.last_seen:
            self.previous = None; self.source_stamp = None; self.reason = 'out-of-order source'
            return
        self.last_seen = ts
        link = observation['links'].get(self.cfg['locomotion_source_link'],{})
        if not link.get('valid',False) or not link.get('orientation_observed',False):
            self.previous = None; self.source_stamp = None; self.reason = 'source base unobserved'
            return
        current = rigid_transform(link['T_world_link'])
        self.source_stamp = ts
        if self.previous is None:
            self.previous = (ts,current); self.desired[:] = 0
            return
        previous_ts,previous_t = self.previous
        dt = (ts-previous_ts)/1e9
        self.previous = (ts,current)
        if dt <= 0 or dt*1000 > self.cfg['max_source_age_ms']:
            self.desired[:] = 0; self.reason = 'source gap: reset derivative'; return
        # Body-local displacement and yaw change. Source X-forward/Z-up must be calibrated.
        delta = np.linalg.inv(previous_t)@current
        linear = delta[:3,3]/dt
        yaw = np.arctan2(delta[1,0],delta[0,0])/dt
        if np.linalg.norm(linear) > self.cfg['max_source_speed_m_s'] or abs(yaw) > self.cfg['max_source_yaw_rate_rad_s']:
            self.source_stamp = None; self.previous = None; self.reason = 'source pose jump'; return
        raw = np.array([linear[0],linear[1],yaw])
        self.desired = np.clip(np.where(np.abs(raw) >= self.deadband,raw,0),-self.limits,self.limits)
        self.reason = 'tracking source base velocity'

    def tick(self, now_ns):
        timestamp(now_ns)
        if self.last_tick is not None and now_ns <= self.last_tick:
            raise ValueError('Nonmonotonic control clock')
        dt = 0 if self.last_tick is None else (now_ns-self.last_tick)/1e9
        self.last_tick = now_ns
        fresh = self.source_stamp is not None and 0 <= now_ns-self.source_stamp <= self.cfg['max_source_age_ms']*1e6
        if fresh:
            self.velocity += np.clip(self.desired-self.velocity,-self.acceleration*dt,self.acceleration*dt)
        else:
            # Request zero base velocity; physical braking is owned by the native controller.
            self.velocity[:] = 0
            self.previous = None
        return {'timestamp_ns':now_ns,'valid_until_ns':now_ns+round(self.cfg['command_duration_s']*1e9),
                'source_timestamp_ns':self.source_stamp if fresh else None,
                'clock_domain':'replay_monotonic','reason':self.reason if fresh else 'missing/stale source: stop request',
                'request':locomotion_request(self.velocity,self.cfg['command_duration_s'],self.cfg['mode'])}


def run(session, config_path, output):
    output,session = Path(output),Path(session)
    if output.exists(): raise FileExistsError('Refusing to overwrite transfer evidence')
    cfg = json.loads(Path(config_path).read_text(encoding='utf-8'))
    target = G1Description(cfg['target_variant_for_offline_model'])
    mapper,intents = TaskMapper(cfg),LocomotionIntent(cfg)
    rows = read_rows(session/'link_poses.jsonl')
    if not rows: raise ValueError('Empty link-pose source')
    tasks,commands = [],[]
    index = 0
    for now in range(rows[0]['timestamp_ns'],rows[-1]['timestamp_ns']+400_000_001,20_000_000):
        while index < len(rows) and rows[index]['timestamp_ns'] <= now:
            row = rows[index]
            tasks.append(mapper.map(row)); intents.observe(row); index += 1
        commands.append(intents.tick(now))
    output.mkdir(parents=True)
    write_rows(output/'task_references.jsonl',tasks)
    write_rows(output/'g1_locomotion_requests.jsonl',commands)
    report = {'source_id':cfg['source_id'],'target_id':cfg['target_id'],'target_variant':target.variant,
              'target_urdf_sha256':target.sha256,'target_joint_count':target.size,
              'reference_count':len(tasks),'command_count':len(commands),'dry_run':True,
              'motor_commands_generated':False,'network_calls':0,
              'hardware_variant_confirmed':cfg['hardware_variant_confirmed'],
              'nonzero_velocity_requests':sum(any(abs(v)>1e-8 for v in c['request']['parameters']['velocity']) for c in commands),
              'end_zero_velocity':not any(commands[-1]['request']['parameters']['velocity']),
              'limitations':['Synthetic/offline link poses do not validate arbitrary unmarked robot perception.',
                             'Whole-body balance, contacts and dynamics controller are not implemented.',
                             'Velocity requests use native locomotion semantics, not source leg-angle copying.',
                             'Firmware, DOF/waist/hand variant and physical interface remain unconfirmed.']}
    (output/'transfer_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('session',type=Path)
    p.add_argument('--mapping',type=Path,default=Path('config/transfer_source_to_g1.json'))
    p.add_argument('--output',type=Path)
    a = p.parse_args()
    print(json.dumps(run(a.session,a.mapping,a.output or a.session/'g1_transfer'),indent=2))
