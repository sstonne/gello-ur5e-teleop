"""Dense grasp/lift curriculum on the unchanged Desk contact physics."""
import numpy as np
import gymnasium as gym
import torch
from .rope_routing_env import RopeRoutingEnv


def lift_reward(distance, aligned, openness, grasp, lift_fraction, action):
    # Bounded stages: holding and lifting outweigh merely hovering/closing.
    reach=.20*(1-np.tanh(20*distance))
    pinch=.15*aligned*(1-openness)
    hold=.45+.45*np.clip(lift_fraction,0,1) if grasp else 0.
    return float(-1+max(reach+pinch,hold)-.002*np.square(action).sum())


class RopeLiftEnv(RopeRoutingEnv):
    def __init__(self,*args,difficulty='near',**kwargs):
        self.difficulty=difficulty
        self.reset_pick_index=12
        self.approach_height={'pinch':.185,'near':.215,'far':.30}[difficulty]
        self.approach_spread={'pinch':.004,'near':.008,'far':.03}[difficulty]
        self.initial_height=.025
        super().__init__(*args,**kwargs)
        self.reset_pick_index=int(self.cfg.env.robot.get('training_pick_index',12))
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,(17,),np.float32)

    def _grasp(self):
        return bool(self.sim.physics.dynamics_module.grasped)

    def _held(self):
        return self._line()[self.pick_index-1:self.pick_index+2].mean(0)

    def _obs(self):
        held=self._held();tcp=self._tcp();s=self._state()
        return np.r_[(held-tcp)*10,self._line('v')[self.pick_index],float(s['eef_gripper'][0,0]),float(self._grasp()),(held[2]-self.initial_height)/.06,tcp,held,self.stable/10,self.elapsed/self.episode_steps].astype(np.float32)

    def reset(self,**kwargs):
        super().reset(**kwargs)
        self.initial_height=float(self._held()[2])
        self.start_xy=self._held()[:2].copy()
        self.peak_lift=0.;self.previous_grasp=False
        if self._grasp():raise RuntimeError('Curriculum reset must not pre-grasp rope')
        return self._obs(),self._info(False)

    def _info(self,success):
        info=super()._info(success)
        info.update(lift_m=float(self._held()[2]-self.initial_height),current_grasp=self._grasp())
        return info

    def step(self,action):
        action=np.asarray(action,dtype=np.float32)
        if action.shape!=(4,) or not np.isfinite(action).all():raise ValueError('Expected four finite actions')
        action=np.clip(action,-1,1)
        s=self._state();xyz=self._array(s['eef_xyz'])[0]
        xyz=np.clip(xyz+action[:3]*.01,self.workspace_low,self.workspace_high)
        if self.difficulty=='pinch':
            xyz[:2]=np.clip(xyz[:2],self.start_xy-.025,self.start_xy+.025)
            xyz[2]=min(xyz[2],self.table_z+.32)
        openness=np.clip(float(s['eef_gripper'][0,0])+action[3]*.10,0,1)
        self._apply(xyz,openness);self.elapsed+=1
        s=self._state()
        if not torch.isfinite(s['x']).all() or not torch.isfinite(s['v']).all():raise FloatingPointError('Nonfinite rope physics')
        held=self._held();offset=self._tcp()-held
        # Existing bilateral contact flag plus local proximity: no reward for a remote snag.
        grasp=self._grasp() and np.linalg.norm(offset)<.045 and float(s['eef_gripper'][0,0])<.85
        lift=float(held[2]-self.initial_height)
        self.peak_lift=max(self.peak_lift,lift)
        self.ever_grasp |= grasp
        self.ever_lift |= grasp and lift>=.06
        aligned=float(np.exp(-np.square(offset/[.015,.02,.015]).sum()))
        reward=lift_reward(np.linalg.norm(offset),aligned,float(s['eef_gripper'][0,0]),grasp,lift/.06,action)
        if self.previous_grasp and not grasp:reward-=.5
        self.previous_grasp=grasp
        # Local segment must remain lifted, near the fingers, and reasonably still.
        local_speed=np.linalg.norm(self._line('v')[self.pick_index-1:self.pick_index+2],axis=1).mean()
        valid=grasp and lift>=.06 and local_speed<.10
        self.stable=self.stable+1 if valid else 0
        success=self.stable>=10
        if success:reward+=10
        return self._obs(),reward,success,self.elapsed>=self.episode_steps and not success,self._info(success)
