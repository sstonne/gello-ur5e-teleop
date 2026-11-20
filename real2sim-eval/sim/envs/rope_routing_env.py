"""State-based routing pilot: learn approach, close, lift, place and release.

Uses the installed UR5e/PhysTwin contact engine. No particle attachments,
scripted grasp at reset, or RGB rendering during learning.
"""
from pathlib import Path
import os
import gymnasium as gym
import numpy as np
import torch
from omegaconf import OmegaConf
from .env import BaseEnv


def channel_crossing(centerline, clip_pose, radius=.0125):
    """Continuous rope segment crosses both ends of the open 35mm-wide channel."""
    local=(np.asarray(centerline)-clip_pose[:3,3])@clip_pose[:3,:3]
    clearance=.0175-radius
    hits=[]
    for plane in (-.0175,.0175):
        found=[]
        for i,(a,b) in enumerate(zip(local[:-1],local[1:])):
            dy=b[1]-a[1]
            if abs(dy)<1e-9:continue
            t=(plane-a[1])/dy
            if 0<=t<=1:
                p=a+t*(b-a)
                if abs(p[0])<=clearance and radius-.003 <= p[2] <= .030:
                    found.append(i+t)
        hits.append(found)
    if not hits[0] or not hits[1]:return False
    for lo in hits[0]:
        for hi in hits[1]:
            low,high=sorted([lo,hi]);inside=local[int(np.ceil(low)):int(np.floor(high))+1]
            if len(inside) and np.any((np.abs(inside[:,0])>clearance)|(inside[:,2]<radius-.003)|(inside[:,2]>.030)):
                continue
            return True
    return False


class RopeRoutingEnv(gym.Env):
    metadata={'render_modes':['rgb_array']}
    def __init__(self,project,rank=0,episode_steps=256,release_required=True):
        self.cfg=OmegaConf.load(str(project))
        self.cfg.env.robot.init_gripper=[1.]
        self.cfg.env.robot.max_joint_speed_rad_s=1.
        from ..utils.robot.ur5e_robotiq import UR5eRobotiq
        robot=UR5eRobotiq(self.cfg)
        pose=np.array(self.cfg.gs.object.pose).reshape(4,4)
        axis=pose[:3,0].copy();axis[2]=0;axis/=np.linalg.norm(axis)
        self.rotation=np.column_stack([axis,[axis[1],-axis[0],0],[0,0,-1]])
        self.table_z=float(self.cfg.physics.table_height)
        clip_pose=np.array(next(m.pose for m in self.cfg.gs.meshes if m.name=='clip')).reshape(4,4)
        anchors=np.array([pose[:3,3],clip_pose[:3,3]])
        self.workspace_low=np.r_[anchors[:,:2].min(0)-.30,self.table_z+.155]
        self.workspace_high=np.r_[anchors[:,:2].max(0)+.30,self.table_z+.50]
        target=np.eye(4);target[:3,:3]=self.rotation;target[:3,3]=pose[:3,3]+[0,0,.28]
        joints,ok=robot.solve_ik(robot.initial_q,target)
        if not ok:raise RuntimeError('Cannot initialize open approach pose')
        self.cfg.env.robot.init_joints_deg=np.degrees(joints).tolist()
        self.cfg.env.robot.init_eef_xyz=target[:3,3].tolist()
        from scipy.spatial.transform import Rotation
        quat=Rotation.from_matrix(self.rotation).as_quat();self.cfg.env.robot.init_quat=quat[[3,0,1,2]].tolist()
        self.sim=BaseEnv(self.cfg,self.cfg.exp_root,local_rank=rank)
        self.clip=np.array(next(m.pose for m in self.cfg.gs.meshes if m.name=='clip')).reshape(4,4)
        # This keeps physics and robot state updates; only appearance deformation is skipped.
        self.full_update=self.sim.renderer.update_rendervar
        self.sim.renderer.update_rendervar=lambda *args,**kwargs:None
        self.action_space=gym.spaces.Box(-1,1,(4,),np.float32)
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,(173,),np.float32)
        self.episode_steps=episode_steps
        self.release_required=release_required
        self.bins=None

    def _state(self):return self.sim.renderer.get_state()
    def _array(self,t):return t.detach().cpu().numpy()
    def _line(self,key='x'):
        x=self._array(self._state()[key])
        return np.array([x[b].mean(0) for b in self.bins])
    def _tcp(self):return self._array(self._state()['eef_xyz'])[0]+self.rotation[:,2]*.148
    def _obs(self):
        s=self._state()
        return np.r_[self._line().ravel(),self._line('v').ravel(),self._array(s['eef_xyz']).ravel(),self._array(s['eef_quat']).ravel(),self._array(s['eef_gripper']).ravel(),self.sim.renderer.qpos_curr_xarm,self.clip[:3,3],self.clip[:3,:3].ravel(),float(self.ever_grasp),float(self.ever_lift),self.elapsed/self.episode_steps].astype(np.float32)
    def _apply(self,xyz,openness):
        s=self._state()
        a=torch.tensor(np.r_[xyz,self.rotation.ravel(),openness],dtype=torch.float32,device=s['x'].device)[None]
        self.sim.step({'action':a,'do_velocity_control':True})

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        self.sim._reset_simulation(seed=int(self.np_random.integers(0,1000000)))
        self.render_reference_x=self.sim.renderer.state['x'].clone()
        self.sim.renderer.relations=None
        self.sim.renderer.weights=None
        x=self._array(self._state()['x'])
        # Fixed membership follows physical particles throughout deformation.
        direction=np.array(self.cfg.gs.object.pose).reshape(4,4)[:3,0]
        ordered=np.argsort(x@direction)
        self.bins=np.array_split(ordered,24)
        self.pick_index=getattr(self,'reset_pick_index',4)
        point=self._line()[self.pick_index]
        spread=getattr(self,'approach_spread',.015)
        xyz=np.r_[point[:2]+self.np_random.uniform(-spread,spread,2),self.table_z+getattr(self,'approach_height',.30)]
        # Approach alignment is above rope, jaws open; no pre-grasp or scripted contact.
        for _ in range(18):self._apply(xyz,1.)
        self.elapsed=0;self.ever_grasp=False;self.ever_lift=False;self.stable=0
        self.mean_rope_speed=0.
        self.previous_potential=self._potential()
        return self._obs(),self._info(False)

    def _potential(self):
        line=self._line();held=line[self.pick_index]
        if not self.ever_grasp:return -np.linalg.norm(self._tcp()-held)
        if not self.ever_lift:return float(min(held[2]-self.table_z,.09))
        # Seat the selected rope region in the channel, then open and retract.
        goal=self.clip[:3,3]+self.clip[:3,:3]@np.array([0,0,.015])
        return -np.linalg.norm(held-goal)

    def _info(self,success):
        return dict(is_success=bool(success),grasped=bool(self.ever_grasp),lifted=bool(self.ever_lift),in_clip=bool(channel_crossing(self._line(),self.clip)),stable_steps=self.stable,mean_rope_speed_m_s=float(self.mean_rope_speed))

    def step(self,action):
        action=np.asarray(action,dtype=np.float32)
        if action.shape!=(4,) or not np.isfinite(action).all():raise ValueError('Expected four finite actions')
        action=np.clip(action,-1,1)
        s=self._state();xyz=self._array(s['eef_xyz'])[0]
        xyz=np.clip(xyz+action[:3]*.01,self.workspace_low,self.workspace_high)
        opening=np.clip(float(s['eef_gripper'][0,0])+action[3]*.10,0,1)
        line_before=self._line().copy()
        self._apply(xyz,opening);self.elapsed+=1
        # The current PhysTwin adapter exposes an aliased before/after buffer,
        # so renderer 'v' can be zero even while particles move. Measure motion
        # independently for success; preserve the trained observation interface.
        self.mean_rope_speed=float(np.linalg.norm((self._line()-line_before)*float(self.cfg.physics.fps),axis=1).mean())
        s=self._state()
        if not torch.isfinite(s['x']).all():raise FloatingPointError('PhysTwin became nonfinite')
        reward=-.01-.002*float(np.square(action).sum())
        old_stage=(self.ever_grasp,self.ever_lift)
        grasp=bool(self.sim.physics.dynamics_module.grasped) and np.linalg.norm(self._tcp()-self._line()[self.pick_index])<.04
        close=float(s['eef_gripper'][0,0])<.85
        if grasp and close and not self.ever_grasp:
            self.ever_grasp=True;reward+=3.
        held=self._line()[self.pick_index]
        if self.ever_grasp and grasp and held[2]-self.table_z>.075 and not self.ever_lift:
            self.ever_lift=True;reward+=5.
        potential=self._potential()
        if old_stage==(self.ever_grasp,self.ever_lift):reward+=25*(potential-self.previous_potential)
        self.previous_potential=potential
        inside=channel_crossing(self._line(),self.clip)
        released=float(s['eef_gripper'][0,0])>.90 and not grasp
        retracted=np.linalg.norm(self._tcp()-held)>.06
        quiet=self.mean_rope_speed<.08
        valid=self.ever_lift and inside and quiet and (not self.release_required or (released and retracted))
        self.stable=self.stable+1 if valid else 0
        success=self.stable>=10
        if success:reward+=30.
        return self._obs(),float(reward),bool(success),bool(self.elapsed>=self.episode_steps and not success),self._info(success)

    def render(self):
        # LBS expects motion since the last appearance update, not current-current.
        # Physics advances without appearance updates during headless learning.
        renderer=self.sim.renderer
        current=renderer.state['x']
        try:
            renderer.state['x']=self.render_reference_x
            self.full_update(current.clone())
        finally:
            renderer.state['x']=current
        self.render_reference_x=current.clone()
        im=self.sim.get_obs()['image_list'][0]
        return (im[:3].permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8)

    def close(self):self.sim.close()
