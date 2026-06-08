from __future__ import annotations

import sys
import pathlib
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import numpy as np
import torch.nn.functional as F
import dill
from omegaconf import OmegaConf
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# Import model classes from local planner directory
from sugar_il.policy.generator import Generator
from sugar_il.model.encoder.generator_state_encoder import GeneratorStateObsEncoder

DOWNSAMPLE_RATE = 5

@dataclass
class GeneratorObs:

    obj_pos_b: torch.Tensor          # (num_envs, n_obs_steps, 3)
    obj_ori_b: torch.Tensor          # (num_envs, n_obs_steps, 4) quaternion wxyz
    joint_pos: torch.Tensor          # (num_envs, n_obs_steps, 29)
    project_gravity: torch.Tensor    # (num_envs, n_obs_steps, 3)
    target_obj_pos_b: torch.Tensor   # (num_envs, n_obs_steps, 3)
    target_obj_ori_b: torch.Tensor   # (num_envs, n_obs_steps, 4) quaternion wxyz
    last_command: torch.Tensor



class GeneratorWrapper:

    
    NB = 14  # Number of bodies
    
    def __init__(self, policy, cfg,
                use_last_action: bool,
                use_target: bool,
                device: str = "cuda"):

        self.policy = policy
        self.cfg = cfg
        self.device = device
        self.n_action_steps = cfg.n_action_steps
        self.n_obs_steps = cfg.n_obs_steps
        self.horizon = self.n_obs_steps + self.n_action_steps - 1
        self.action_dim = policy.action_dim
        self.use_last_action = use_last_action
        self.use_target = use_target

    
    @classmethod
    def load(
            cls,
            checkpoint_path: str,
            device: str = "cuda",
            ) -> "GeneratorWrapper":

        
        # Register resolvers
        OmegaConf.register_new_resolver("eval", eval, replace=True)
        
        print(f"[Generator] Loading checkpoint from: {checkpoint_path}")
        
        # Load checkpoint
        with open(checkpoint_path, 'rb') as f:
            payload = torch.load(f, pickle_module=dill, map_location='cpu')
        
        cfg = payload['cfg']
        use_target = cfg.use_target
        use_last_action = cfg.use_last_action
        OmegaConf.resolve(cfg)
        
        # Extract policy config
        policy_cfg = cfg.policy
        
        # Build shape_meta (需要包含obs和action)
        shape_meta = {
            'obs': {
                'n_obs_steps': cfg.n_obs_steps
            },
            'action': {
                'shape': [policy_cfg.shape_meta.action.shape[0]],
                'horizon': cfg.n_obs_steps + cfg.n_action_steps - 1  # 总horizon
            }
        }

        obs_encoder = GeneratorStateObsEncoder(
            shape_meta=shape_meta,
            feature_dim=policy_cfg.obs_encoder.feature_dim,
            use_last_action=use_last_action,
            use_target=use_target
        )
        
        # Create noise scheduler
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=policy_cfg.noise_scheduler.num_train_timesteps,
            beta_start=policy_cfg.noise_scheduler.beta_start,
            beta_end=policy_cfg.noise_scheduler.beta_end,
            beta_schedule=policy_cfg.noise_scheduler.beta_schedule,
            clip_sample=policy_cfg.noise_scheduler.clip_sample,
            prediction_type=policy_cfg.noise_scheduler.prediction_type
        )
        
        # Create policy
        policy = Generator(
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,
            obs_encoder=obs_encoder,
            num_inference_steps=policy_cfg.num_inference_steps,
            n_layer=policy_cfg.n_layer,
            n_head=policy_cfg.n_head,
            p_drop_attn=policy_cfg.p_drop_attn,
            use_attn_mask=policy_cfg.use_attn_mask,
        )
        
        # Load model state dict
        policy.load_state_dict(payload['state_dicts']['model'])
        
        policy.eval()
        policy.to(device)
        
        print(f"[Generator] Model loaded successfully!")
        print(f"  n_obs_steps: {cfg.n_obs_steps}")
        print(f"  n_action_steps: {cfg.n_action_steps}")
        print(f"  horizon: {cfg.n_obs_steps + cfg.n_action_steps - 1}")
        print(f"  action_dim: {policy.action_dim}")
        
        return cls(policy, cfg, 
                   use_last_action=use_last_action,
                   use_target=use_target,
                   device = device)
    
    @staticmethod
    def quat_to_6d_rotation_row(quat: torch.Tensor) -> torch.Tensor:
        """
        Convert quaternion (wxyz) to 6D rotation representation.
        
        Args:
            quat: (..., 4) quaternion in wxyz format (torch.Tensor)
        
        Returns:
            (..., 6) first two rows of rotation matrix
        """

        w = quat[..., 0]
        x = quat[..., 1]
        y = quat[..., 2]
        z = quat[..., 3]
        
        # First row of rotation matrix
        r00 = 1 - 2*(y*y + z*z)
        r01 = 2*(x*y - z*w)
        r02 = 2*(x*z + y*w)
        
        # Second row of rotation matrix
        r10 = 2*(x*y + z*w)
        r11 = 1 - 2*(x*x + z*z)
        r12 = 2*(y*z - x*w)
        
        return torch.stack([r00, r01, r02, r10, r11, r12], dim=-1)
        
    @staticmethod
    def rot6drow_to_quaternion(rot_6d: torch.Tensor) -> torch.Tensor:
        """
        Convert 6D rotation representation to quaternion (wxyz).
        Args:
            rot_6d: (..., 6) tensor, first two rows of rotation matrix
        Returns:
            (..., 4) quaternion in (w, x, y, z) format
        """
        a1 = rot_6d[..., 0:3]
        a2 = rot_6d[..., 3:6]

        b1 = F.normalize(a1, dim=-1)
        b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)

        R = torch.stack((b1, b2, b3), dim=-2)  # (..., 3, 3)
        """
        Convert rotation matrix to quaternion (wxyz).
        """
        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

        qw = torch.sqrt(torch.clamp(trace + 1.0, min=1e-8)) / 2.0
        qx = (R[..., 2, 1] - R[..., 1, 2]) / (4.0 * qw)
        qy = (R[..., 0, 2] - R[..., 2, 0]) / (4.0 * qw)
        qz = (R[..., 1, 0] - R[..., 0, 1]) / (4.0 * qw)

        quat = torch.stack((qw, qx, qy, qz), dim=-1)
        return F.normalize(quat, dim=-1)
    
    @staticmethod
    def quat_to_6d_rotation_col(quat: torch.Tensor) -> torch.Tensor:
        """
        
        Args:
            quat: (..., 4) , wxyz
        
        Returns:
            (..., 6) 
        """
        w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

        r00 = 1 - 2 * (y**2 + z**2)
        r10 = 2 * (x * y + z * w)
        r20 = 2 * (x * z - y * w)

        r01 = 2 * (x * y - z * w)
        r11 = 1 - 2 * (x**2 + z**2)
        r21 = 2 * (y * z + x * w)


        return torch.stack([r00, r01, r10, r11, r20, r21], dim=-1)


    
    def _prepare_obs_dict(self, obs: GeneratorObs) -> Dict[str, torch.Tensor]:
        """
        Convert GeneratorObs to model input format.
        
        Converts quaternions to 6D rotation.
        Input already has time dimension: (num_envs, n_obs_steps, dim)
        """
        # Convert quaternions to 6D rotation: (num_envs, n_obs_steps, 4) -> (num_envs, n_obs_steps, 6)
        obj_ori_6d = self.quat_to_6d_rotation_col(obs.obj_ori_b)
        target_obj_ori_6d = self.quat_to_6d_rotation_col(obs.target_obj_ori_b)
        # Already have time dimension: (num_envs, n_obs_steps, dim)

        obs_dict = {
                'obj_pos_b': obs.obj_pos_b[:,::DOWNSAMPLE_RATE],               # (B, T, 3)
                'obj_ori_b': obj_ori_6d[:,::DOWNSAMPLE_RATE],                  # (B, T, 6)
        }
        if self.use_target:
            obs_dict['target_obj_pos_b'] = obs.target_obj_pos_b[:,::DOWNSAMPLE_RATE] # (B, T, 3)
            obs_dict['target_obj_ori_b'] = target_obj_ori_6d[:,::DOWNSAMPLE_RATE]    # (B, T, 6)

        if self.use_last_action:
            obs_dict['last_action'] = obs.last_command[:,::DOWNSAMPLE_RATE]
        else:
            obs_dict['joint_pos'] = obs.joint_pos[:,::DOWNSAMPLE_RATE]
            obs_dict['project_gravity'] = obs.project_gravity[:,::DOWNSAMPLE_RATE]
        
        
        return obs_dict
    
    def _parse_action(self, action: torch.Tensor):
        """
        Parse raw action tensor into structured output.
        
        Model outputs action of shape (B, horizon, action_dim) where horizon = n_obs_steps + n_action_steps - 1.
        We only return the last n_action_steps.
        
        New action structure (action_dim = 36):
            - ref_joint_pos: 29
            - ref_anchor_lin_vel_b: 3
            - ref_anchor_ang_vel_b: 3
            - contact_label: 1
        
        Args:
            action: (num_envs, horizon, action_dim) full action output
        
        Returns:
            GeneratorOutput with parsed components (only last n_action_steps)
        """
        B, horizon, dim = action.shape
        
        # Only take the last n_action_steps
        # horizon = n_obs_steps + n_action_steps - 1
        # we want indices [n_obs_steps-1, horizon) which is the last n_action_steps
        action = action[:, self.n_obs_steps - 1:, :]  # (B, n_action_steps, action_dim)
        na = action.shape[1]
        
        # idx = 0
        
        action_interp = F.interpolate(
            action.transpose(1, 2), 
            size=(na-1) * DOWNSAMPLE_RATE+1,  
            mode='linear', 
            align_corners=True
        ).transpose(1, 2)

        return action_interp
    
    @torch.no_grad()
    def predict(self, obs: GeneratorObs):
        """
        Run planner inference.
        
        Args:
            obs: GeneratorObs containing observation tensors with shape (num_envs, n_obs_steps, dim)
        
        Returns:
            GeneratorOutput with predicted actions (only last n_action_steps)
        """
        obs_dict = self._prepare_obs_dict(obs)
        action = self.policy.predict_action(obs_dict)  # (B, horizon, action_dim)
        return self._parse_action(action)
    
