from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING
import pickle
import time


from isaaclab.assets import Articulation
from isaaclab.assets.rigid_object.rigid_object import RigidObject
from isaaclab.sensors import ContactSensor
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers.config import SPHERE_MARKER_CFG
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
    quat_from_matrix,
    subtract_frame_transforms,
    quat_apply_inverse,
    matrix_from_quat
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import trimesh
import torch
import numpy as np
from pathlib import Path
from typing import Optional


from sugar_il.wrapper.sugar_il_wrapper import GeneratorWrapper, GeneratorObs, DOWNSAMPLE_RATE

class MotionLoader:
    def __init__(self,
                motion_folder: str,
                body_indexes: Sequence[int],
                motion_names: list[str] | None = None,
                eval_max_time=None, 
                device: str = "cpu"):
        data_list = []
        time_step_total_permotion = []
        contact_label_list = []

        if motion_names is not None:
            motion_list = sorted([Path(motion_folder) / name for name in motion_names])
        else:
            motion_list = sorted(list(Path(motion_folder).glob("data_*")))
        num_motion = len(motion_list)
        for i in range(num_motion):
            with np.load(f'{motion_list[i]}/robot_50hz.npz') as f:
                robot_data = {k: v for k, v in f.items()}
            with open(f'{motion_list[i]}/obj_motion_global_50hz.pkl', 'rb') as f:
                object_data = pickle.load(f)

            data_list.append((robot_data, object_data))
            time_step_total_permotion.append(robot_data["joint_pos"].shape[0])
            contact_label_list.append(np.load(f'{motion_list[i]}/contact_labels_50hz.npy'))
        
        max_time_step = max(time_step_total_permotion)
        if eval_max_time is not None:
            max_time_step = eval_max_time 

        # robot
        self.joint_pos = torch.zeros((num_motion, max_time_step, data_list[0][0]["joint_pos"].shape[1]), dtype=torch.float32, device=device)
        self.joint_vel = torch.zeros((num_motion, max_time_step, data_list[0][0]["joint_pos"].shape[1]), dtype=torch.float32, device=device)

        self._body_pos_w = torch.zeros((num_motion, max_time_step, 35, data_list[0][0]['body_pos_w'].shape[2]), dtype=torch.float32, device=device)
        self._body_quat_w = torch.zeros((num_motion, max_time_step, 35, data_list[0][0]['body_quat_w'].shape[2]), dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.zeros((num_motion, max_time_step, 35, data_list[0][0]['body_lin_vel_w'].shape[2]), dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.zeros((num_motion, max_time_step, 35, data_list[0][0]['body_ang_vel_w'].shape[2]), dtype=torch.float32, device=device)
        

        # object
        self.obj_pos = torch.zeros((num_motion, max_time_step, 3), dtype=torch.float32, device=device)
        self.obj_quat = torch.zeros((num_motion, max_time_step, 4), dtype=torch.float32, device=device)
        self.obj_lin_vel = torch.zeros((num_motion, max_time_step, 3), dtype=torch.float32, device=device)
        self.obj_ang_vel = torch.zeros((num_motion, max_time_step, 3), dtype=torch.float32, device=device)

        # contact label
        self.contact_label = torch.zeros((num_motion,max_time_step),dtype=torch.bool, device=device)

        self._body_indexes = body_indexes
        self.time_step_total_permotion = torch.tensor(time_step_total_permotion, dtype=torch.int64, device=device)
        self.num_motion = num_motion

        for i in range(num_motion):
            robot_data, object_data = data_list[i]
            t = time_step_total_permotion[i]
            self.joint_pos[i, :t] = torch.tensor(robot_data["joint_pos"], dtype=torch.float32, device=device)
            self.joint_vel[i, :t] = torch.tensor(robot_data["joint_vel"], dtype=torch.float32, device=device)

            # The rollout saved 14 used bodies, which is inconsistent with the original 35.
            if robot_data["body_pos_w"].shape[1] == len(self._body_indexes):
                self._body_pos_w[i, :t, self._body_indexes] = torch.tensor(robot_data["body_pos_w"], dtype=torch.float32, device=device)
                self._body_quat_w[i, :t, self._body_indexes] = torch.tensor(robot_data["body_quat_w"], dtype=torch.float32, device=device)
                self._body_lin_vel_w[i, :t, self._body_indexes] = torch.tensor(robot_data["body_lin_vel_w"], dtype=torch.float32, device=device)
                self._body_ang_vel_w[i, :t, self._body_indexes] = torch.tensor(robot_data["body_ang_vel_w"], dtype=torch.float32, device=device)
            else:
                self._body_pos_w[i, :t] = torch.tensor(robot_data["body_pos_w"], dtype=torch.float32, device=device)
                self._body_quat_w[i, :t] = torch.tensor(robot_data["body_quat_w"], dtype=torch.float32, device=device)
                self._body_lin_vel_w[i, :t] = torch.tensor(robot_data["body_lin_vel_w"], dtype=torch.float32, device=device)
                self._body_ang_vel_w[i, :t] = torch.tensor(robot_data["body_ang_vel_w"], dtype=torch.float32, device=device)

            # Slight length discrepancy exists between object and robot data due to separate acquisition. Aligning them to the same length.
            self.obj_pos[i, :t] = torch.tensor(object_data["obj_trans"], dtype=torch.float32, device=device)[:t]    
            self.obj_quat[i, :t] = quat_from_matrix(torch.tensor(object_data["obj_rot"], dtype=torch.float32, device=device))[:t]
            self.obj_lin_vel[i, :t] = torch.tensor(object_data["obj_lin_vel"], dtype=torch.float32, device=device)[:t]
            self.obj_ang_vel[i, :t] = torch.tensor(object_data["obj_ang_vel"], dtype=torch.float32, device=device)[:t]

            self.contact_label[i,:t] = torch.tensor(contact_label_list[i][:t],dtype=torch.bool, device=device)
            
            
    @property
    def body_pos_w(self) -> torch.Tensor:
        if self._body_pos_w.shape[2] == len(self._body_indexes):
            return self._body_pos_w
        else:
            return self._body_pos_w[:, :, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        if self._body_quat_w.shape[2] == len(self._body_indexes):
            return self._body_quat_w
        else:
            return self._body_quat_w[:, :, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        if self._body_lin_vel_w.shape[2] == len(self._body_indexes):
            return self._body_lin_vel_w
        else:
            return self._body_lin_vel_w[:, :, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        if self._body_ang_vel_w.shape[2] == len(self._body_indexes):
            return self._body_ang_vel_w
        else:
            return self._body_ang_vel_w[:, :, self._body_indexes]



class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.count = 0

        self.obj: RigidObject = env.scene[cfg.obj_name]

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name) 
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)    # index of index
        self.key_body_indexes = [self.cfg.body_names.index(name) for name in self.cfg.key_body_names]
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        
        def parse_motion_id(folder_name: str) -> int:
            """Extracts motion_id from the filename. Supports two formats:
            - data_{motion_id} (e.g., data_005)
            - data_{motion_id}_{env_id} (e.g., data_005_010)
            """
            parts = folder_name.split('_')
            if len(parts) >= 2:
                return int(parts[1])
            raise ValueError(f"Invalid folder name format: {folder_name}")

        teacher_motion_names = None
        if self.cfg.teacher_motion_folder is not None:
            motion_files = sorted(list(Path(self.cfg.motion_folder).glob("data_*")))
            motion_id_list = []
            for f in motion_files:
                motion_id = parse_motion_id(f.name)
                motion_id_list.append(motion_id)
            
            teacher_files = sorted(list(Path(self.cfg.teacher_motion_folder).glob("data_*")))
            teacher_motion_dict = {}  # motion_id -> folder_name
            for f in teacher_files:
                mid = parse_motion_id(f.name)
                teacher_motion_dict[mid] = f.name

            teacher_motion_names = []
            for mid in motion_id_list:
                if mid in teacher_motion_dict:
                    teacher_motion_names.append(teacher_motion_dict[mid])
                else:
                    print(f"[MotionCommand] Warning: motion_id {mid} not found in teacher_motion_folder")
                    teacher_motion_names.append(None)
            
            valid_indices = [i for i, name in enumerate(teacher_motion_names) if name is not None]
            teacher_motion_names = [teacher_motion_names[i] for i in valid_indices]
            motion_file_names = [motion_files[i].name for i in valid_indices]
            
            print(f"[MotionCommand] Teacher motion enabled: {len(teacher_motion_names)} motions ")
            print(f"  motion_id_list: {motion_id_list[:10]}... (total {len(motion_id_list)})")
        else:
            motion_file_names = None

        self.motion = MotionLoader(
            self.cfg.motion_folder,
            self.body_indexes,
            motion_names=motion_file_names,
            eval_max_time=self.cfg.eval_max_time+5 if self.cfg.eval_mode else None,
            device=self.device
        )
        if self.cfg.teacher_motion_folder is not None:
            self.teacher_motion = MotionLoader(
                self.cfg.teacher_motion_folder,
                self.body_indexes,
                motion_names=teacher_motion_names,
                device=self.device
            )
            assert not ((self.teacher_motion.time_step_total_permotion - self.motion.time_step_total_permotion).abs() > 2.0).any(), print('teacher motion is not aligned with student motion!!!!!!!!!')
        else:
            self.teacher_motion = None

        self.motion_id = torch.randint(0, self.motion.num_motion, (self.num_envs,), device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        self.sit_contact_time = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.obj_static_time = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.robot_static_time = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.start_init_env_count = int(self.num_envs * self.cfg.start_init_env_ratio)
        
        if self.cfg.rollout_traj or self.cfg.eval_mode:
            self.start_init_env_count = self.num_envs
            
        print(f"[MotionCommand] start_init_env_count: {self.start_init_env_count} / {self.num_envs} ({self.cfg.start_init_env_ratio*100:.1f}%)")    
        
        # self._use_motion_data: marks whether each env reads from ref_motion (True) or from the pool (False)
        # Set in _sample_init_state and used in _get_init_state.
        self._use_motion_data = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.init_pool = {
            'joint_pos': self.motion.joint_pos.clone(),  # num_motion, t, dof
            'joint_vel': self.motion.joint_vel.clone(),  # num_motion, t, dof
            'root_pos_w': self.motion.body_pos_w[:, :, 0].clone(),  # Stores positions without env_origin, num_motion, t, 3
            'root_quat_w': self.motion.body_quat_w[:, :, 0].clone(),  # num_motion, t, 4
            'root_lin_vel_w': self.motion.body_lin_vel_w[:, :, 0].clone(),  # num_motion, t, 3
            'root_ang_vel_w': self.motion.body_ang_vel_w[:, :, 0].clone(),  # num_motion, t, 3

            'obj_pos_w': self.motion.obj_pos.clone(),  # num_motion, t, 3
            'obj_quat_w': self.motion.obj_quat.clone(),  # num_motion, t, 4
            'obj_lin_vel_w': self.motion.obj_lin_vel.clone(),  # num_motion, t, 3
            'obj_ang_vel_w': self.motion.obj_ang_vel.clone(),  # num_motion, t, 3

            'last_action': torch.zeros_like(self.motion.joint_pos),  # num_motion, t, dof (initialized to zero)

            'flag': torch.zeros((self.motion.joint_pos.shape[0], self.motion.joint_pos.shape[1]), dtype=torch.bool, device=self.device),    # num_motion, t
        }

        self.init_pool['flag'][:,0] = True 

        
        # Candidate-state validation: save candidates validation_k steps early; only states not terminated during validation can be written into the pool.
        self.validation_k = 48  # Validation window
        self.candidate_states = {
            'joint_pos': torch.zeros(self.num_envs, self.motion.joint_pos.shape[-1], device=self.device),
            'joint_vel': torch.zeros(self.num_envs, self.motion.joint_vel.shape[-1], device=self.device),
            'root_pos_w': torch.zeros(self.num_envs, 3, device=self.device),
            'root_quat_w': torch.zeros(self.num_envs, 4, device=self.device),
            'root_lin_vel_w': torch.zeros(self.num_envs, 3, device=self.device),
            'root_ang_vel_w': torch.zeros(self.num_envs, 3, device=self.device),
            'obj_pos_w': torch.zeros(self.num_envs, 3, device=self.device),
            'obj_quat_w': torch.zeros(self.num_envs, 4, device=self.device),
            'obj_lin_vel_w': torch.zeros(self.num_envs, 3, device=self.device),
            'obj_ang_vel_w': torch.zeros(self.num_envs, 3, device=self.device),
            'last_action': torch.zeros(self.num_envs, self.motion.joint_pos.shape[-1], device=self.device),
        }
        self.candidate_motion_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.candidate_time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.candidate_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # Whether the candidate is valid


        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

        self.metrics["error_obj_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_obj_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_obj_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_obj_ang_vel"] = torch.zeros(self.num_envs, device=self.device)

        
        # Record the timestep at the last reset for each env, used to compute episode length.
        self.last_reset_timestep = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.last_reset_motion_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # Target object pose in world coordinates.
        self.obj_target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.obj_target_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.obj_target_quat_w[:, 0] = 1.0  # Initialize as the identity quaternion
        
        # Visualization markers for the target object pose.
        target_marker_cfg = FRAME_MARKER_CFG.copy()
        target_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)  # Slightly smaller for distinction
        target_marker_cfg.prim_path = "/Visuals/Command/target"
        self.target_visualizer = VisualizationMarkers(target_marker_cfg)
        

        # =====================================================================
        # Rollout trajectory collection initialization
        # =====================================================================
        if self.cfg.rollout_traj:
            self._init_rollout_collection()

        # =====================================================================
        # Generator initialization
        # =====================================================================
        self.generator: Optional[GeneratorWrapper] = None
        if self.cfg.use_generator:
            self._init_generator()

    def _init_generator(self):
        if not self.cfg.generator_checkpoint_path:
            raise ValueError("[Generator] generator_checkpoint_path is required when use_generator=True")
        
        print(f"[Generator] Initializing Generator...")
        self.generator = GeneratorWrapper.load(
            checkpoint_path=self.cfg.generator_checkpoint_path,
            device=self.device,
        )
        print(f"[Generator] Generator initialized successfully!")
        
        # Read generator configuration parameters
        n_obs_steps = self.generator.n_obs_steps
        n_action_steps = self.generator.n_action_steps
        future_frames = self.cfg.future_frames
        
        
        if self.cfg.generator_call_interval is not None:
            self.generator_call_interval = self.cfg.generator_call_interval
        else:
            self.generator_call_interval = n_action_steps - future_frames + 1
        
        print(f"[Generator] n_obs_steps={n_obs_steps}, n_action_steps={n_action_steps}, "
              f"future_frames={future_frames}, call_interval={self.generator_call_interval}")
        
        # ========== 1. Generator observation history buffer ==========
        self.generator_obs_buffer = {
            'obj_pos_b': torch.zeros(self.num_envs, (n_obs_steps-1)*DOWNSAMPLE_RATE+1, 3, device=self.device),
            'obj_ori_b': torch.zeros(self.num_envs, (n_obs_steps-1)*DOWNSAMPLE_RATE+1, 4, device=self.device),
            'joint_pos': torch.zeros(self.num_envs, (n_obs_steps-1)*DOWNSAMPLE_RATE+1, 29, device=self.device),
            'project_gravity': torch.zeros(self.num_envs, (n_obs_steps-1)*DOWNSAMPLE_RATE+1, 3, device=self.device),
            'target_obj_pos_b': torch.zeros(self.num_envs, (n_obs_steps-1)*DOWNSAMPLE_RATE+1, 3, device=self.device),
            'target_obj_ori_b': torch.zeros(self.num_envs, (n_obs_steps-1)*DOWNSAMPLE_RATE+1, 4, device=self.device),
            'last_command': torch.zeros(self.num_envs, (n_obs_steps-1)*DOWNSAMPLE_RATE+1, 36, device=self.device),
        }
        
        # Initialize quaternions as identity quaternions
        self.generator_obs_buffer['obj_ori_b'][:, :, 0] = 1.0
        self.generator_obs_buffer['target_obj_ori_b'][:, :, 0] = 1.0
        self.generator_obs_buffer['project_gravity'][:, :, 2] = -1.0
        
        # ========== 2. Generated command buffer ==========
        # action_dim = 36:
        self.generated_command_buffer = {
            'action': torch.zeros(self.num_envs, (n_action_steps-1)*DOWNSAMPLE_RATE+1, 36, device=self.device)
        }
        
        
        print(f"[Generator] Buffers initialized. Ready for inference.")


    def _update_generator_obs_buffer(self, env_ids: Optional[torch.Tensor] = None):
        """
        Update the generator observation buffer.
        
        Shift the observation history left by one step and append the current observation at the end.
        
        Args:
            env_ids: The specified environment IDs. If None, update all environments.
        """
        if self.generator is None:
            return
        
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            
        # Get the observation for the current frame
        current_obs = self._get_generator_obs(env_ids)  # Returns shape (len(env_ids), 1, dim)


        # Shift left and append the new observation
        # shift: buffer[:, 1:, :] -> buffer[:, :-1, :]
        # append: current_obs -> buffer[:, -1, :]
        for key in self.generator_obs_buffer.keys():
            # Shift left
            self.generator_obs_buffer[key][env_ids, :-1] = self.generator_obs_buffer[key][env_ids, 1:].clone()
            # Append the current observation (squeeze the time dimension)
            self.generator_obs_buffer[key][env_ids, -1] = getattr(current_obs, key).squeeze(1)

    def _fill_generator_obs_buffer(self, env_ids: torch.Tensor):
        """
        Fill the entire generator observation buffer with the current observation.
        
        Used during reset, when no history is available and all time steps should be filled with the current observation.
        
        Args:
            env_ids: The environment IDs to fill.
        """
        if self.generator is None or len(env_ids) == 0:
            return
        
        # Get the observation for the current frame
        current_obs = self._get_generator_obs(env_ids)  # Returns shape (len(env_ids), 1, dim)
        
        n_obs_steps = self.generator.n_obs_steps
        
        # Fill all time steps with the current observation
        for key in self.generator_obs_buffer.keys():
            # Expand the time dimension and fill the buffer
            obs_val = getattr(current_obs, key).squeeze(1)  # (len(env_ids), dim)
            self.generator_obs_buffer[key][env_ids] = obs_val.unsqueeze(1).expand(-1, (n_obs_steps-1)*DOWNSAMPLE_RATE+1, -1).clone()

    def _call_generator(self, env_ids: Optional[torch.Tensor] = None):
        """
        Run generator inference for the specified environments.
        
        Args:
            env_ids: The environment IDs to run the generator on. If None, run inference for all environments.
        """
        if self.generator is None:
            return
        
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return
        
        # Build GeneratorObs for the selected env_ids only
        obs = GeneratorObs(
            obj_pos_b=self.generator_obs_buffer['obj_pos_b'][env_ids],
            obj_ori_b=self.generator_obs_buffer['obj_ori_b'][env_ids],
            joint_pos=self.generator_obs_buffer['joint_pos'][env_ids],
            project_gravity=self.generator_obs_buffer['project_gravity'][env_ids],
            target_obj_pos_b=self.generator_obs_buffer['target_obj_pos_b'][env_ids],
            target_obj_ori_b=self.generator_obs_buffer['target_obj_ori_b'][env_ids],
            last_command=self.generator_obs_buffer['last_command'][env_ids]
        )
        
        # Run generator inference for the selected env_ids only
        output = self.generator.predict(obs)
        
        # Update the generated_command buffer for the selected env_ids only
        self.generated_command_buffer['action'][env_ids] = output
        
        

    def _get_generator_obs(self, env_ids: Optional[torch.Tensor] = None) -> GeneratorObs:
        """
        Get the generator observation.
        
        Note: this method returns the single-frame observation for the current step, with shape (num_envs, 1, dim).
        
        Args:
            env_ids: The specified environment IDs. If None, get observations for all environments.
        
        Returns:
            GeneratorObs: A data structure containing all observations required by the generator, with shape (num_envs, 1, dim)
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        num_envs = len(env_ids)
        
        # ========== 1. Object pose in the anchor frame ==========
        obj_pos_b, obj_ori_b = subtract_frame_transforms(
            self.robot_anchor_pos_w[env_ids],
            self.robot_anchor_quat_w[env_ids],
            self.obj_pos_w[env_ids],
            self.obj_quat_w[env_ids],
        )  # obj_pos_b: (num_envs, 3), obj_ori_b: (num_envs, 4)
        
        # ========== 2. Joint positions ==========
        joint_pos = self.robot_joint_pos[env_ids]  # (num_envs, 29)
        
        # 3.3wts: during deployment, the gravity projection is taken from the base frame
        gravity_w = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(num_envs, 3)

        project_gravity = quat_apply(quat_inv(self.robot_base_quat_w[env_ids]), gravity_w)  # (num_envs, 3)
        
        # ========== 4. Offset from the current object pose to the target object pose ==========
        target_obj_pos_b, target_obj_ori_b = subtract_frame_transforms(
            self.robot_anchor_pos_w[env_ids],
            self.robot_anchor_quat_w[env_ids],
            self.obj_target_pos_w[env_ids],
            self.obj_target_quat_w[env_ids],
        )  # target_obj_pos_o: (num_envs, 3), target_obj_ori_o: (num_envs, 4)
        
        # Add the time dimension: (num_envs, dim) -> (num_envs, 1, dim)
        return GeneratorObs(
            obj_pos_b=obj_pos_b.unsqueeze(1),
            obj_ori_b=obj_ori_b.unsqueeze(1),
            joint_pos=joint_pos.unsqueeze(1),
            project_gravity=project_gravity.unsqueeze(1),
            target_obj_pos_b=target_obj_pos_b.unsqueeze(1),
            target_obj_ori_b=target_obj_ori_b.unsqueeze(1),
            last_command=self.last_command[env_ids].clone(),
        )
    
    def _init_rollout_collection(self):
        """Initialize rollout trajectory collection."""
        print(f"[Rollout] Initializing rollout trajectory collection...")
        print(f"[Rollout] num_envs={self.num_envs}, num_motions={self.motion.num_motion}")
        print(f"[Rollout] rollout_start_distance={self.cfg.rollout_start_distance}")
        
        # Generate a timestamp or use the provided directory
        if getattr(self.cfg, 'rollout_dir', None) is not None:
            self.rollout_base_dir = self.cfg.rollout_dir
            os.makedirs(self.rollout_base_dir, exist_ok=True)
            print(f"[Rollout] Using provided rollout directory: {self.rollout_base_dir}")
        else:
            self.rollout_run_name = time.strftime('%Y%m%d_%H%M%S')
            self.rollout_base_dir = f"refined_data/{self.rollout_run_name}/raw_npz"
            os.makedirs(self.rollout_base_dir, exist_ok=True)
            print(f"[Rollout] Using generated rollout directory: {self.rollout_base_dir}")
        
        # 1. Assign reference trajectories to each env in order (round-robin)
        # Ensure that each motion is assigned at least once
        self.motion_id[:] = torch.arange(self.num_envs, device=self.device) % self.motion.num_motion
        self.time_steps[:] = 0
        
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._record_reference_targets(env_ids)
        
        print(f"[Rollout] Motion assignment: {self.motion_id[:min(16, self.num_envs)].tolist()}...")
        
        # 2. Initialize the completion flag array
        self.rollout_completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 2.1 Initialize rollout start timestep tracking (each env keeps one start point, incremented after each rollout)
        self.rollout_start_timesteps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # 2.2 Record how many rollouts have been saved for each env (used for file naming)
        self.rollout_save_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # 3. Initialize termination-reason tracking
        # Termination reasons: 'trajectory_complete', 'anchor_ori', 'ee_body_pos', 'obj_pos', 'obj_ori', 'other'
        self.rollout_end_reasons = [''] * self.num_envs  # List of strings
        
        # 4. Initialize trajectory buffers (one list per env)
        # Estimate the maximum length using the maximum motion timestep
        max_timesteps = self.motion.time_step_total_permotion.max().item()

        num_bodies = len(self.cfg.body_names)
        num_dof = self.motion.joint_pos.shape[-1]
        
        # Trajectory data structure
        self.rollout_trajectories = {
            # new action
            'ref_joint_pos': [[] for _ in range(self.num_envs)],         # (29,)
            # 'ref_anchor_lin_vel_b': [[] for _ in range(self.num_envs)],  # (3,)
            # 'ref_anchor_ang_vel_b': [[] for _ in range(self.num_envs)],  # (3,)
            'ref_root_lin_vel_b': [[] for _ in range(self.num_envs)],  # (3,)
            'ref_root_ang_vel_b': [[] for _ in range(self.num_envs)],  # (3,)
            'ref_contact_label': [[] for _ in range(self.num_envs)],        # (1,) bool

            
            # Anchor (torso) state - the reference frame used for reward and observation computation
            'anchor_pos_w': [[] for _ in range(self.num_envs)],         # (3,)
            'anchor_quat_w': [[] for _ in range(self.num_envs)],         # (4,)
            'anchor_lin_vel_w': [[] for _ in range(self.num_envs)],     # (3,)
            'anchor_ang_vel_w': [[] for _ in range(self.num_envs)],     # (3,)
            'anchor_lin_vel_b': [[] for _ in range(self.num_envs)],     # (3,)
            'anchor_ang_vel_b': [[] for _ in range(self.num_envs)],     # (3,)
            'project_gravity': [[] for _ in range(self.num_envs)],      # (3,) gravity projection in the anchor frame
            # Root state - used to set the robot state during replay
            'root_pos_w': [[] for _ in range(self.num_envs)],           # (3,)
            'root_quat_w': [[] for _ in range(self.num_envs)],           # (4,)
            'root_lin_vel_w': [[] for _ in range(self.num_envs)],       # (3,)
            'root_ang_vel_w': [[] for _ in range(self.num_envs)],       # (3,)
            'root_lin_vel_b': [[] for _ in range(self.num_envs)],       # (3,)
            'root_ang_vel_b': [[] for _ in range(self.num_envs)],       # (3,)
            # Body state in the anchor frame
            'body_pos_b': [[] for _ in range(self.num_envs)],           # (num_bodies, 3)
            'body_quat_b': [[] for _ in range(self.num_envs)],           # (num_bodies, 4)
            # Body state in the world frame
            'body_pos_w': [[] for _ in range(self.num_envs)],           # (num_bodies, 3)
            'body_quat_w': [[] for _ in range(self.num_envs)],           # (num_bodies, 4)
            'body_lin_vel_w': [[] for _ in range(self.num_envs)],       # (num_bodies, 3)
            'body_ang_vel_w': [[] for _ in range(self.num_envs)],       #
            'joint_pos': [[] for _ in range(self.num_envs)],            # (num_dof,)
            'joint_vel': [[] for _ in range(self.num_envs)],            # (num_dof,)
            'action': [[] for _ in range(self.num_envs)],               # (action_dim,)
            'obj_pos_w': [[] for _ in range(self.num_envs)],            # (3,)
            'obj_quat_w': [[] for _ in range(self.num_envs)],            # (4,)
            # Object pose in the anchor frame
            'obj_pos_b': [[] for _ in range(self.num_envs)],            # (3,)
            'obj_quat_b': [[] for _ in range(self.num_envs)],            # (4,)
            'obj_lin_vel_w': [[] for _ in range(self.num_envs)],        # (3,)
            'obj_ang_vel_w': [[] for _ in range(self.num_envs)],        #  (3,)
            'obj2body_pos': [[] for _ in range(self.num_envs)],         # (num_bodies, 3)
            'obj2body_quat': [[] for _ in range(self.num_envs)],         # (num_bodies, 4)
            'target_obj_pos_b': [[] for _ in range(self.num_envs)],     # (3,)
            'target_obj_quat_b': [[] for _ in range(self.num_envs)],     # (4,)
            'hands_contact_label': [[] for _ in range(self.num_envs)],        # (1,) bool
            'foot_contact_label': [[] for _ in range(self.num_envs)],        # (1,) bool
            'sit_contact_label': [[] for _ in range(self.num_envs)],        # (1,) bool

        }
        
        # 5. Compute the expected total number of rollouts (assuming all rollouts finish by timeout)
        # For each env, compute how many rollout windows its assigned motion can produce
        self.rollout_expected_total = 0
        for env_id in range(self.num_envs):
            motion_id = self.motion_id[env_id].item()
            motion_length = self.motion.time_step_total_permotion[motion_id].item()
            # Valid start range: 0 to motion_length - 101 (because new_start >= motion_length - 100 marks completion)
            # Start points: 0, rollout_start_distance, 2*rollout_start_distance, ...
            # Compute how many valid start points there are
            if motion_length <= 100:
                num_rollouts = 0
            else:
                max_valid_start = motion_length - 101
                num_rollouts = (max_valid_start // self.cfg.rollout_start_distance) + 1
            self.rollout_expected_total += num_rollouts
        
        print(f"[Rollout] Trajectory buffers initialized. Max timesteps: {max_timesteps}")
        print(f"[Rollout] Data dimensions: num_bodies={num_bodies}, num_dof={num_dof}")
        print(f"[Rollout] Expected total rollouts (assuming all timeouts): {self.rollout_expected_total}")
        print(f"[Rollout] rollout_window_length={self.cfg.rollout_window_length}")

    def _collect_rollout_step(self):
        """Collect rollout trajectory data for the current step."""
        if not self.cfg.rollout_traj:
            return
        # Collect data only for unfinished envs
        active_envs = (~self.rollout_completed).nonzero(as_tuple=False).squeeze(-1)
        if len(active_envs) == 0:
            return
        
        num_bodies = len(self.cfg.body_names)
        env_origins = self._env.scene.env_origins[active_envs]  # Get env-origin offsets
        
        # new action
        ref_joint_pos = self.joint_pos[active_envs].clone()
        ref_root_lin_vel_b = self.root_lin_vel_b[active_envs].clone()
        ref_root_ang_vel_b = self.root_ang_vel_b[active_envs].clone()
        ref_contact_label = self.contact_label[active_envs].clone()
        
        # ========== 1.1 Anchor-related data (torso, the reference frame for reward/obs computation) ==========
        anchor_pos_w = self.robot_anchor_pos_w[active_envs].clone() - env_origins
        anchor_ori_w = self.robot_anchor_quat_w[active_envs].clone()
        base_quat_w = self.robot_base_quat_w[active_envs].clone()
        anchor_lin_vel_w = self.robot_anchor_lin_vel_w[active_envs].clone()
        anchor_ang_vel_w = self.robot_anchor_ang_vel_w[active_envs].clone()
        
        anchor_lin_vel_b = self.robot_anchor_lin_vel_b[active_envs].clone()
        anchor_ang_vel_b = self.robot_anchor_ang_vel_b[active_envs].clone()
        
        # ========== 1.2 Root-related data ==========
        root_pos_w = self.robot.data.root_pos_w[active_envs].clone() - env_origins
        root_ori_w = self.robot.data.root_quat_w[active_envs].clone()
        root_lin_vel_w = self.robot.data.root_lin_vel_w[active_envs].clone()
        root_ang_vel_w = self.robot.data.root_ang_vel_w[active_envs].clone()

        root_lin_vel_b = self.robot.data.root_lin_vel_b[active_envs].clone()
        root_ang_vel_b = self.robot.data.root_ang_vel_b[active_envs].clone()
        
        # ========== 1.3 Anchor-frame gravity projection ==========
        # Project the gravity vector from the world frame into the anchor frame
        # wts 3.3: during deployment, the gravity projection comes from the base frame
        gravity_w = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(len(active_envs), 3)
        project_gravity = quat_apply(quat_inv(base_quat_w), gravity_w)  # Transform into the anchor local frame
        
        # ========== 2. Body-related data in the anchor frame ==========
        body_pos_b, body_ori_b = subtract_frame_transforms(
            self.robot_anchor_pos_w[active_envs, None, :].repeat(1, num_bodies, 1),
            self.robot_anchor_quat_w[active_envs, None, :].repeat(1, num_bodies, 1),
            self.robot_body_pos_w[active_envs],
            self.robot_body_quat_w[active_envs],
        )
        
        # ========== 2.1 Body-related data in the world frame, minus env_origins ==========
        body_pos_w = self.robot_body_pos_w[active_envs].clone() - env_origins.unsqueeze(1)
        body_ori_w = self.robot_body_quat_w[active_envs].clone()

        body_lin_vel_w = self.robot_body_lin_vel_w[active_envs].clone()
        body_ang_vel_w = self.robot_body_ang_vel_w[active_envs].clone()
        obj_lin_vel_w = self.obj_lin_vel_w[active_envs].clone()
        obj_ang_vel_w = self.obj_ang_vel_w[active_envs].clone()
        
        # ========== 3. Joint-related data ==========
        joint_pos = self.robot_joint_pos[active_envs].clone()
        joint_vel = self.robot_joint_vel[active_envs].clone()
        
        # ========== 4. Action data ==========
        action = self._env.action_manager.action[active_envs].clone()
        
        # ========== 5. Object-related data (subtract env_origins to convert to relative coordinates) ==========
        obj_pos_w = self.obj_pos_w[active_envs].clone() - env_origins
        obj_ori_w = self.obj_quat_w[active_envs].clone()
        
        # ========== 5.1 Object pose in the anchor frame ==========
        obj_pos_b, obj_ori_b = subtract_frame_transforms(
            self.robot_anchor_pos_w[active_envs],
            self.robot_anchor_quat_w[active_envs],
            self.obj_pos_w[active_envs],
            self.obj_quat_w[active_envs],
        )
        
        # ========== 6. Obj2Body data ==========
        obj2body_pos, obj2body_ori = subtract_frame_transforms(
            self.robot_body_pos_w[active_envs],
            self.robot_body_quat_w[active_envs],
            self.obj_pos_w[active_envs, None, :].repeat(1, num_bodies, 1),
            self.obj_quat_w[active_envs, None, :].repeat(1, num_bodies, 1),
        )
        
        # ========== 7. Target object data ==========
        goal_pos_w = self.obj_target_pos_w[active_envs]
        goal_quat_w = self.obj_target_quat_w[active_envs]
        
        target_obj_pos_b, target_obj_ori_b = subtract_frame_transforms(
            self.robot_anchor_pos_w[active_envs],
            self.robot_anchor_quat_w[active_envs],
            goal_pos_w,
            goal_quat_w,
        )
        
        # ========== 8. Contact labels ==========

        # Both-hands label
        left_hand_sensor = self._env.scene.sensors["left_hand_forces"]
        right_hand_sensor = self._env.scene.sensors["right_hand_forces"]
        
        left_hand_forces = left_hand_sensor.data.force_matrix_w_history[active_envs]
        right_hand_forces = right_hand_sensor.data.force_matrix_w_history[active_envs]
        
        left_hand_max_force = torch.max(torch.norm(left_hand_forces[:, :, 0, 0, :], dim=-1), dim=1)[0]
        right_hand_max_force = torch.max(torch.norm(right_hand_forces[:, :, 0, 0, :], dim=-1), dim=1)[0]
        hands_contact_label = (left_hand_max_force > 0.1) & (right_hand_max_force > 0.1)

        # Single-foot label
        left_foot_sensor = self._env.scene.sensors["left_foot_forces"]
        right_foot_sensor = self._env.scene.sensors["right_foot_forces"]
        
        left_foot_forces = left_foot_sensor.data.force_matrix_w_history[active_envs]
        right_foot_forces = right_foot_sensor.data.force_matrix_w_history[active_envs]
        
        left_foot_max_force = torch.max(torch.norm(left_foot_forces[:, :, 0, 0, :], dim=-1), dim=1)[0]
        right_foot_max_force = torch.max(torch.norm(right_foot_forces[:, :, 0, 0, :], dim=-1), dim=1)[0]
        foot_contact_label = (left_foot_max_force > 0.1) | (right_foot_max_force > 0.1)

        # Hip/pelvis contact label
        left_hip_sensor = self._env.scene.sensors["left_hip_forces"]
        right_hip_sensor = self._env.scene.sensors["right_hip_forces"]
        pelvis_sensor = self._env.scene.sensors["pelvis_forces"]
        
        left_hip_forces = left_hip_sensor.data.force_matrix_w_history[active_envs]
        right_hip_forces = right_hip_sensor.data.force_matrix_w_history[active_envs]
        pelvis_forces = pelvis_sensor.data.force_matrix_w_history[active_envs]
        
        left_hip_max_force = torch.max(torch.norm(left_hip_forces[:, :, 0, 0, :], dim=-1), dim=1)[0]
        right_hip_max_force = torch.max(torch.norm(right_hip_forces[:, :, 0, 0, :], dim=-1), dim=1)[0]
        pelvis_max_force = torch.max(torch.norm(pelvis_forces[:, :, 0, 0, :], dim=-1), dim=1)[0]

        sit_contact_label = (left_hip_max_force > 0.1) | (right_hip_max_force > 0.1) | (pelvis_max_force > 0.1) 
        


        # ========== Store into buffers ==========
        for i, env_id in enumerate(active_envs.tolist()):
            # new action
            self.rollout_trajectories['ref_joint_pos'][env_id].append(ref_joint_pos[i].cpu().numpy())
            self.rollout_trajectories['ref_root_lin_vel_b'][env_id].append(ref_root_lin_vel_b[i].cpu().numpy())
            self.rollout_trajectories['ref_root_ang_vel_b'][env_id].append(ref_root_ang_vel_b[i].cpu().numpy())
            self.rollout_trajectories['ref_contact_label'][env_id].append(ref_contact_label[i].cpu().numpy())

            
            # Anchor data
            self.rollout_trajectories['anchor_pos_w'][env_id].append(anchor_pos_w[i].cpu().numpy())
            self.rollout_trajectories['anchor_quat_w'][env_id].append(anchor_ori_w[i].cpu().numpy())
            self.rollout_trajectories['anchor_lin_vel_w'][env_id].append(anchor_lin_vel_w[i].cpu().numpy())
            self.rollout_trajectories['anchor_ang_vel_w'][env_id].append(anchor_ang_vel_w[i].cpu().numpy())
            self.rollout_trajectories['anchor_lin_vel_b'][env_id].append(anchor_lin_vel_b[i].cpu().numpy())
            self.rollout_trajectories['anchor_ang_vel_b'][env_id].append(anchor_ang_vel_b[i].cpu().numpy())
            self.rollout_trajectories['project_gravity'][env_id].append(project_gravity[i].cpu().numpy())
            # Root data
            self.rollout_trajectories['root_pos_w'][env_id].append(root_pos_w[i].cpu().numpy())
            self.rollout_trajectories['root_quat_w'][env_id].append(root_ori_w[i].cpu().numpy())
            self.rollout_trajectories['root_lin_vel_w'][env_id].append(root_lin_vel_w[i].cpu().numpy())
            self.rollout_trajectories['root_ang_vel_w'][env_id].append(root_ang_vel_w[i].cpu().numpy())

            self.rollout_trajectories['root_lin_vel_b'][env_id].append(root_lin_vel_b[i].cpu().numpy())
            self.rollout_trajectories['root_ang_vel_b'][env_id].append(root_ang_vel_b[i].cpu().numpy())
            # Body data
            self.rollout_trajectories['body_pos_b'][env_id].append(body_pos_b[i].cpu().numpy())
            self.rollout_trajectories['body_quat_b'][env_id].append(body_ori_b[i].cpu().numpy())
            self.rollout_trajectories['body_pos_w'][env_id].append(body_pos_w[i].cpu().numpy())
            self.rollout_trajectories['body_quat_w'][env_id].append(body_ori_w[i].cpu().numpy())
            self.rollout_trajectories['body_lin_vel_w'][env_id].append(body_lin_vel_w[i].cpu().numpy())
            self.rollout_trajectories['body_ang_vel_w'][env_id].append(body_ang_vel_w[i].cpu().numpy())
            # Joint/action data
            self.rollout_trajectories['joint_pos'][env_id].append(joint_pos[i].cpu().numpy())
            self.rollout_trajectories['joint_vel'][env_id].append(joint_vel[i].cpu().numpy())
            self.rollout_trajectories['action'][env_id].append(action[i].cpu().numpy())
            # Object data
            self.rollout_trajectories['obj_pos_w'][env_id].append(obj_pos_w[i].cpu().numpy())
            self.rollout_trajectories['obj_quat_w'][env_id].append(obj_ori_w[i].cpu().numpy())
            self.rollout_trajectories['obj_pos_b'][env_id].append(obj_pos_b[i].cpu().numpy())
            self.rollout_trajectories['obj_quat_b'][env_id].append(obj_ori_b[i].cpu().numpy())
            self.rollout_trajectories['obj_lin_vel_w'][env_id].append(obj_lin_vel_w[i].cpu().numpy())
            self.rollout_trajectories['obj_ang_vel_w'][env_id].append(obj_ang_vel_w[i].cpu().numpy())
            # Obj2Body data
            self.rollout_trajectories['obj2body_pos'][env_id].append(obj2body_pos[i].cpu().numpy())
            self.rollout_trajectories['obj2body_quat'][env_id].append(obj2body_ori[i].cpu().numpy())
            # Target data
            self.rollout_trajectories['target_obj_pos_b'][env_id].append(target_obj_pos_b[i].cpu().numpy())
            self.rollout_trajectories['target_obj_quat_b'][env_id].append(target_obj_ori_b[i].cpu().numpy())
            # Contact data
            self.rollout_trajectories['hands_contact_label'][env_id].append(hands_contact_label[i].cpu().numpy())
            self.rollout_trajectories['foot_contact_label'][env_id].append(foot_contact_label[i].cpu().numpy())
            self.rollout_trajectories['sit_contact_label'][env_id].append(sit_contact_label[i].cpu().numpy())

    def _record_rollout_termination(self, env_ids: Sequence[int], termination_reasons: dict = None):
        """Record rollout termination information, supporting multiple rollouts.
        
        New logic:
        - After each termination, regardless of the reason, increase the rollout start by rollout_start_distance steps.
        - Save a trajectory file only for timeouts that reach the end of the rollout window or trajectory.
        - Do not save trajectories terminated for other reasons, but still advance the start point.
        - Mark the env as completed when the new start point is >= trajectory length - 100.
        
        Args:
            env_ids: List of terminated environment IDs.
            termination_reasons: Optional termination-reason dictionary {term_name: mask}
        """
        if not self.cfg.rollout_traj:
            return
        
        # Return immediately if all envs are already completed
        if self.rollout_completed.all():
            return
        
        env_ids_tensor = env_ids if isinstance(env_ids, torch.Tensor) else torch.tensor(env_ids, device=self.device)
        
        for env_id in env_ids_tensor.tolist():
            # Skip envs that are already completed
            if self.rollout_completed[env_id]:
                continue
            
            # Check whether trajectory data exists; skip if nothing has been collected
            if len(self.rollout_trajectories['anchor_pos_w'][env_id]) == 0:
                continue
            
            # Determine the termination reason
            motion_id = self.motion_id[env_id].item()
            timestep = self.time_steps[env_id].item()
            max_timestep = self.motion.time_step_total_permotion[motion_id].item()
            current_start = self.rollout_start_timesteps[env_id].item()
            
            # Compute the number of steps since the rollout start
            steps_since_start = timestep - current_start
            
            # Determine whether this is a timeout (window complete or trajectory end)
            window_timeout = steps_since_start >= self.cfg.rollout_window_length - 1
            trajectory_end = timestep >= max_timestep - 1
            is_timeout = window_timeout or trajectory_end
            
            if is_timeout:
                # Timeout: save the trajectory; both window completion and trajectory end use trajectory_complete
                self.rollout_end_reasons[env_id] = 'trajectory_complete'
                self._save_single_trajectory(env_id, current_start, timestep, max_timestep)
                
                # NOTE: _save_single_trajectory increments rollout_save_count internally, so sum() is the post-save value here
                save_count = self.rollout_save_count[env_id].item()
                total_saved = self.rollout_save_count.sum().item()
                progress_pct = 100 * total_saved / self.rollout_expected_total if hasattr(self, 'rollout_expected_total') and self.rollout_expected_total > 0 else 0
                print(f"[Rollout] ({progress_pct:.1f}%) Env {env_id} saved rollout #{save_count} (motion={motion_id}, t={current_start}-{timestep})")
            else:
                # Non-timeout: do not save; silently clear the trajectory
                for key in self.rollout_trajectories:
                    self.rollout_trajectories[key][env_id] = []
            
            # Advance the rollout start regardless of timeout status
            new_start = current_start + self.cfg.rollout_start_distance
            self.rollout_start_timesteps[env_id] = new_start
            
            # Check whether the new start exceeds the limit (>= trajectory length - 100)
            if new_start >= max_timestep - 100:
                self.rollout_completed[env_id] = True
                self.rollout_start_timesteps[env_id] = 0
                completed_count = self.rollout_completed.sum().item()
                save_count = self.rollout_save_count[env_id].item()
                print(f"[Rollout] Progress: {completed_count}/{self.num_envs} "
                      f"({100*completed_count/self.num_envs:.1f}%) - Env {env_id} completed "
                      f"(motion={motion_id}, saved {save_count} rollouts)")
        
        # Check whether all envs are completed
        if self.rollout_completed.all():
            total_saved = self.rollout_save_count.sum().item()
            msg = f"\n[Rollout] ====== All {self.num_envs} envs completed, total {total_saved} trajectories saved to {self.rollout_base_dir} ======"
            print(msg)
            # Raise SystemExit so the program exits cleanly at this point
            raise SystemExit(msg)

    def _save_single_trajectory(self, env_id: int, rollout_start: int = 0, rollout_end: int = 0, max_timestep: int = 0):
        """Save the trajectory of a single env to an npz file.
        
        Args:
            env_id: Environment ID.
            rollout_start: Start timestep of this rollout, used for file naming and metadata.
            rollout_end: End timestep of this rollout.
            max_timestep: Maximum timestep length of the original trajectory.
        """
        motion_id = self.motion_id[env_id].item()
        reason = self.rollout_end_reasons[env_id]
        
        # Create the directory structure: rollout_data/{time}/trajectory_complete/
        save_dir = os.path.join(self.rollout_base_dir, reason)
        os.makedirs(save_dir, exist_ok=True)
        
        # Build the filename including the rollout start and end timesteps
        rollout_idx = self.rollout_save_count[env_id].item()
        filename = f"motion_{motion_id}_env_{env_id}_t{rollout_start}-{rollout_end}_idx_{rollout_idx}.npz"
        filepath = os.path.join(save_dir, filename)
        
        # Convert trajectory data to NumPy arrays
        traj_data = {}
        for key, value_list in self.rollout_trajectories.items():
            if len(value_list[env_id]) > 0:
                traj_data[key] = np.stack(value_list[env_id], axis=0)
        
        # Add metadata
        traj_data['motion_id'] = motion_id
        traj_data['env_id'] = env_id
        traj_data['end_reason'] = reason
        traj_data['fps'] = 50  # Control frequency: 50 Hz
        traj_data['num_timesteps'] = len(self.rollout_trajectories['anchor_pos_w'][env_id])
        traj_data['key_body_indexes'] = np.array(self.key_body_indexes)
        traj_data['rollout_start'] = rollout_start  # Rollout start timestep
        traj_data['rollout_end'] = rollout_end  # Rollout end timestep
        traj_data['rollout_idx'] = rollout_idx  # Rollout index
        traj_data['original_max_timestep'] = max_timestep  # Original trajectory maximum length
        
        # Save
        np.savez(filepath, **traj_data)
        
        # Increment the save counter
        self.rollout_save_count[env_id] += 1
        
        # Clear the saved trajectory data to free memory
        for key in self.rollout_trajectories:
            self.rollout_trajectories[key][env_id] = []

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

        self.metrics["error_obj_pos"] = torch.norm(self.obj_pos_w - self.obj_ref_pos_w, dim=-1)
        self.metrics["error_obj_rot"] = quat_error_magnitude(self.obj_quat_w, self.obj_ref_quat_w)
        self.metrics["error_obj_lin_vel"] = torch.norm(self.obj_lin_vel_w - self.obj_ref_lin_vel_w, dim=-1)
        self.metrics["error_obj_ang_vel"] = torch.norm(self.obj_ang_vel_w - self.obj_ref_ang_vel_w, dim=-1)

    def _update_pool_metrics(self):
        """Update pool- and motion-related metrics at every step to avoid dilution from averaging."""
        return
    
    def _sample_init_state(self, env_ids: Sequence[int]):
        """Unified entry point for sampling initialization states.
        
        Determine the sampling strategy from init_with_ref and set the _use_motion_data flag:
        
        Protected env (first 25%/80%):
            - motion_id: evenly distributed in order (round-robin) and kept fixed
            - timestep: 
                - init_with_ref=False: fixed at 0
                - init_with_ref=True: randomly selected from [0, end-48]
            - Data source: motion
        
        Pool env (remaining 75%/20%):
            - motion_id and timestep: randomly sample a valid state from the pool each time
            - Data source: pool
        """
        if len(env_ids) == 0:
            return
        
        env_ids_tensor = env_ids if isinstance(env_ids, torch.Tensor) else torch.tensor(env_ids, device=self.device)
        
        # Split protected envs and pool envs
        start_init_mask = env_ids_tensor < self.start_init_env_count
        start_init_env_ids = env_ids_tensor[start_init_mask]
        free_env_ids = env_ids_tensor[~start_init_mask]
        
        if len(start_init_env_ids) > 0:
            # Sample motion_id by round-robin assignment
            if self.cfg.eval_mode and self.cfg.eval_random_motion:
                self.motion_id[start_init_env_ids] = torch.randint(0, self.motion.num_motion, (len(start_init_env_ids),), device=self.device)
            else:
                self.motion_id[start_init_env_ids] = start_init_env_ids % self.motion.num_motion
                
            # Sample timestep
            if self.cfg.rollout_traj:
                # Note: in rollout mode, use rollout_start_timesteps as the start point
                self.time_steps[start_init_env_ids] = self.rollout_start_timesteps[start_init_env_ids]
            else:
                # Outside rollout mode, always start from 0
                self.time_steps[start_init_env_ids] = 0
            
            # Mark the data source as motion
            self._use_motion_data[start_init_env_ids] = True
        
        # Free envs: sample from random timesteps
        if len(free_env_ids) > 0:
            if self.cfg.init_with_ref:
                # 1. Sample (motion_id, timestep) uniformly from all available frames

                self.motion_id[free_env_ids] = torch.randint(0, self.motion.num_motion, (len(free_env_ids),), device=self.device)
                max_timesteps = torch.clamp(self.motion.time_step_total_permotion[self.motion_id[free_env_ids]] - 48, min=1)
                self.time_steps[free_env_ids] = (torch.rand(len(free_env_ids), device=self.device) * max_timesteps).long()

                # 2. Determine probability of using Reference state (vs Pool state)
                if self.count < self.cfg.pool_warmup_steps:
                    ref_prob = 1.0
                elif self.count < self.cfg.pool_minref_steps:
                    # Linear decay from 1.0 to pool_minref_ratio
                    alpha = (self.count - self.cfg.pool_warmup_steps) / (self.cfg.pool_minref_steps - self.cfg.pool_warmup_steps)
                    ref_prob = 1.0 - alpha * (1.0 - self.cfg.pool_minref_ratio)
                else:
                    ref_prob = self.cfg.pool_minref_ratio

                # 3. Sample decision: True -> prefer Ref, False -> prefer Pool
                use_ref = torch.rand(len(free_env_ids), device=self.device) < ref_prob

                # 4. Check validity in pool (must fallback to ref if pool is empty/invalid at this state)
                pool_valid = self.init_pool['flag'][self.motion_id[free_env_ids], self.time_steps[free_env_ids]].bool()

                # 5. Final decision: Use Motion Data (Ref) if preferred OR if pool is invalid
                self._use_motion_data[free_env_ids] = use_ref | (~pool_valid)
            else:
                # Use pool initialization (states with timestep=0 in the pool are valid from the start)
                
                # Get indices of all valid states in the pool
                valid_indices = self.init_pool['flag'].nonzero(as_tuple=False)
                
                if len(valid_indices) > 0:
                    # Randomly sample valid states
                    random_idx = torch.randint(0, len(valid_indices), (len(free_env_ids),), device=self.device)
                    selected_pairs = valid_indices[random_idx]
                    
                    self.motion_id[free_env_ids] = selected_pairs[:, 0]
                    self.time_steps[free_env_ids] = selected_pairs[:, 1]
                else:
                    # If the pool is empty (should not happen because timestep 0 exists)
                    raise RuntimeError("Init pool is empty! No valid states to sample for pool envs.")
                    
                    
                # Mark the data source as pool
                self._use_motion_data[free_env_ids] = False
    
    def _get_init_state(self, env_ids: torch.Tensor) -> dict:
        """Get initialization states using a unified path based on the _use_motion_data flag.
        
        Args:
            env_ids: Indices of envs whose states should be fetched.
            
        Returns:
            A dictionary containing the initial robot and object states, sized for num_envs.
        """
        dof = self.motion.joint_pos.shape[-1]
        
        # Initialize state buffers with size num_envs
        state = {
            'root_pos_w': torch.zeros(self.num_envs, 3, device=self.device),
            'root_quat_w': torch.zeros(self.num_envs, 4, device=self.device),
            'root_lin_vel_w': torch.zeros(self.num_envs, 3, device=self.device),
            'root_ang_vel_w': torch.zeros(self.num_envs, 3, device=self.device),
            'joint_pos': torch.zeros(self.num_envs, dof, device=self.device),
            'joint_vel': torch.zeros(self.num_envs, dof, device=self.device),
            'obj_pos_w': torch.zeros(self.num_envs, 3, device=self.device),
            'obj_quat_w': torch.zeros(self.num_envs, 4, device=self.device),
            'obj_lin_vel_w': torch.zeros(self.num_envs, 3, device=self.device),
            'obj_ang_vel_w': torch.zeros(self.num_envs, 3, device=self.device),
            'last_action': torch.zeros(self.num_envs, dof, device=self.device),
        }
        
        # Fetch data using absolute indices
        motion_ids = self.motion_id[env_ids]
        timesteps = self.time_steps[env_ids]
        use_motion = self._use_motion_data[env_ids]
        env_origins = self._env.scene.env_origins[env_ids]
        
        # Envs read from motion data (absolute indices)
        motion_env_ids = env_ids[use_motion]
        if len(motion_env_ids) > 0:
            m_ids = motion_ids[use_motion]
            m_ts = timesteps[use_motion]
            m_origins = env_origins[use_motion]
            
            state['root_pos_w'][motion_env_ids] = self.motion.body_pos_w[m_ids, m_ts, 0] + m_origins
            state['root_quat_w'][motion_env_ids] = self.motion.body_quat_w[m_ids, m_ts, 0]
            state['root_lin_vel_w'][motion_env_ids] = self.motion.body_lin_vel_w[m_ids, m_ts, 0]
            state['root_ang_vel_w'][motion_env_ids] = self.motion.body_ang_vel_w[m_ids, m_ts, 0]
            state['joint_pos'][motion_env_ids] = self.motion.joint_pos[m_ids, m_ts]
            state['joint_vel'][motion_env_ids] = self.motion.joint_vel[m_ids, m_ts]
            state['obj_pos_w'][motion_env_ids] = self.motion.obj_pos[m_ids, m_ts] + m_origins
            state['obj_quat_w'][motion_env_ids] = self.motion.obj_quat[m_ids, m_ts]
            state['obj_lin_vel_w'][motion_env_ids] = self.motion.obj_lin_vel[m_ids, m_ts]
            state['obj_ang_vel_w'][motion_env_ids] = self.motion.obj_ang_vel[m_ids, m_ts]
            # last_action is zero for motion initialization already, so no action is needed
        
        # Envs read from the pool (absolute indices)
        pool_env_ids = env_ids[~use_motion]
        if len(pool_env_ids) > 0:
            p_ids = motion_ids[~use_motion]
            p_ts = timesteps[~use_motion]
            p_origins = env_origins[~use_motion]
            
            state['root_pos_w'][pool_env_ids] = self.init_pool['root_pos_w'][p_ids, p_ts] + p_origins
            state['root_quat_w'][pool_env_ids] = self.init_pool['root_quat_w'][p_ids, p_ts]
            state['root_lin_vel_w'][pool_env_ids] = self.init_pool['root_lin_vel_w'][p_ids, p_ts]
            state['root_ang_vel_w'][pool_env_ids] = self.init_pool['root_ang_vel_w'][p_ids, p_ts]
            state['joint_pos'][pool_env_ids] = self.init_pool['joint_pos'][p_ids, p_ts]
            state['joint_vel'][pool_env_ids] = self.init_pool['joint_vel'][p_ids, p_ts]
            state['obj_pos_w'][pool_env_ids] = self.init_pool['obj_pos_w'][p_ids, p_ts] + p_origins
            state['obj_quat_w'][pool_env_ids] = self.init_pool['obj_quat_w'][p_ids, p_ts]
            state['obj_lin_vel_w'][pool_env_ids] = self.init_pool['obj_lin_vel_w'][p_ids, p_ts]
            state['obj_ang_vel_w'][pool_env_ids] = self.init_pool['obj_ang_vel_w'][p_ids, p_ts]
            state['last_action'][pool_env_ids] = self.init_pool['last_action'][p_ids, p_ts]
        
        return state
 
    def _record_reference_targets(self, env_ids: Sequence[int]):
        
        """Record the target pose of the reference trajectory.
        
        Directly use the target pose from the motion data as the reference target pose.
        """
        if len(env_ids) == 0:
            return

        motion_ids = self.motion_id[env_ids]
        last_frame_indices = self.motion.time_step_total_permotion[motion_ids] - 1
        
        goal_pos_w = self.motion.obj_pos[motion_ids, last_frame_indices] + self._env.scene.env_origins[env_ids]
        goal_quat_w = self.motion.obj_quat[motion_ids, last_frame_indices]
        
        self.obj_target_pos_w[env_ids] = goal_pos_w
        self.obj_target_quat_w[env_ids] = goal_quat_w
        
        self._update_target_visualization()
        
    def _resample_command(self, env_ids: Sequence[int]):
        # This is part of the reset logic
        if len(env_ids) == 0:
            return
        
        env_ids_tensor = env_ids if isinstance(env_ids, torch.Tensor) else torch.tensor(env_ids, device=self.device)
        # ====== Rollout mode: record termination reasons ======
        if self.cfg.rollout_traj:
            self._record_rollout_termination(env_ids_tensor)
        
        # Mark candidates of reset envs as invalid
        if len(env_ids) > 0:
            self.candidate_valid[env_ids] = False
        
        # Step 1: sample motion_id/time_steps and set the data-source flag
        self._sample_init_state(env_ids_tensor)
        
        # Record new reset information
        self.last_reset_timestep[env_ids] = self.time_steps[env_ids]
        self.last_reset_motion_id[env_ids] = self.motion_id[env_ids]
        
        
        # Record the target object pose for training the Command Generator
        self._record_reference_targets(env_ids)
        
        
        # Step 2: get initialization states through the unified path (returns arrays sized for num_envs)
        state = self._get_init_state(env_ids_tensor)
        
        # Extract states into local variables
        root_pos = state['root_pos_w']
        root_ori = state['root_quat_w']
        root_lin_vel = state['root_lin_vel_w']
        root_ang_vel = state['root_ang_vel_w']
        joint_pos = state['joint_pos']
        joint_vel = state['joint_vel']
        obj_pos = state['obj_pos_w']
        obj_quat = state['obj_quat_w']
        obj_vel = state['obj_lin_vel_w']
        obj_ang_vel = state['obj_ang_vel_w']
        last_action = state['last_action']

        # Step 3: add initialization noise only for envs with timestep=0
        start_up_mask = self.time_steps[env_ids] == 0
        start_up_env_ids = env_ids[start_up_mask]
        if start_up_mask.any():
            range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
            ranges = torch.tensor(range_list, device=self.device)
            rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(start_up_env_ids), 6), device=self.device)
            root_pos[start_up_env_ids] += rand_samples[:, 0:3]
            orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
            root_ori[start_up_env_ids] = quat_mul(orientations_delta, root_ori[start_up_env_ids])

            joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)

        # Step 4: write to the simulator
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )
        
        self.obj.write_root_state_to_sim(torch.cat([
            obj_pos[env_ids],
            obj_quat[env_ids],
            obj_vel[env_ids],
            obj_ang_vel[env_ids]
            ], dim=-1),
            env_ids=env_ids,)

        # Restore last_action
        self._env.action_manager.action[env_ids] = last_action[env_ids]
        
        # ========== Generator reset logic ==========
        # Fill the entire observation buffer with the current observation during reset
        # Because time_steps=0 after reset and 0 % call_interval == 0, the generator will be triggered automatically
        if self.generator is not None and len(env_ids) > 0:
            env_ids_tensor = env_ids if isinstance(env_ids, torch.Tensor) else torch.tensor(env_ids, device=self.device)
            self._fill_generator_obs_buffer(env_ids_tensor)

    def _update_target_visualization(self):
        """Update visualization markers for random targets."""
        
        # Update the frame marker to display the target pose
        self.target_visualizer.visualize(
            translations=self.obj_target_pos_w,
            orientations=self.obj_target_quat_w
        )
        

    def _update_command(self):
        # Generator logic
        
        if self.generator is not None:
            # 1. Update the observation buffer
            self._update_generator_obs_buffer()
            
            # 2. Call the generator
            # Condition: for each env, time_steps % call_interval == 0
            env_ids = (self.time_steps % self.generator_call_interval == 0).nonzero(as_tuple=False).squeeze(-1)
            self._call_generator(env_ids)
            
        # Sparsely update the init pool with random samples, using a validation mechanism
        update_interval = 24 * 100  # 2400 控制步
        warmup_steps = self.cfg.pool_warmup_steps    # 12000 控制步 warmup

        self.count += 1

        # Stage 1: write valid candidates into the pool at the validation step
        if (self.count % update_interval == 0) and self.count > warmup_steps:
            valid_envs = self.candidate_valid.nonzero(as_tuple=False).squeeze(-1)
            if len(valid_envs) > 0:
                motion_ids = self.candidate_motion_id[valid_envs]
                time_steps = self.candidate_time_steps[valid_envs]
                
                # Filter out timestep 0 to protect the original reference-trajectory start point
                updatable_mask = time_steps > 0
                valid_envs = valid_envs[updatable_mask]
                motion_ids = motion_ids[updatable_mask]
                time_steps = time_steps[updatable_mask]
                
                if len(valid_envs) > 0:
                    # Deduplicate: for identical (motion_id, timestep), keep only the last env
                    # Use a dict so that pool storage and source tracking refer to the same env
                    unique_keys = {}  # {(motion_id, timestep): index_in_valid_envs}
                    for i in range(len(valid_envs)):
                        key = (motion_ids[i].item(), time_steps[i].item())
                        unique_keys[key] = i  # Later entries overwrite earlier ones
                    
                    # Get the deduplicated indices
                    unique_indices = torch.tensor(list(unique_keys.values()), dtype=torch.long, device=self.device)
                    valid_envs = valid_envs[unique_indices]
                    motion_ids = motion_ids[unique_indices]
                    time_steps = time_steps[unique_indices]
                    
                    self.init_pool['joint_pos'][motion_ids, time_steps] = self.candidate_states['joint_pos'][valid_envs]
                    self.init_pool['joint_vel'][motion_ids, time_steps] = self.candidate_states['joint_vel'][valid_envs]
                    self.init_pool['root_pos_w'][motion_ids, time_steps] = self.candidate_states['root_pos_w'][valid_envs]
                    self.init_pool['root_quat_w'][motion_ids, time_steps] = self.candidate_states['root_quat_w'][valid_envs]
                    self.init_pool['root_lin_vel_w'][motion_ids, time_steps] = self.candidate_states['root_lin_vel_w'][valid_envs]
                    self.init_pool['root_ang_vel_w'][motion_ids, time_steps] = self.candidate_states['root_ang_vel_w'][valid_envs]
                    self.init_pool['obj_pos_w'][motion_ids, time_steps] = self.candidate_states['obj_pos_w'][valid_envs]
                    self.init_pool['obj_quat_w'][motion_ids, time_steps] = self.candidate_states['obj_quat_w'][valid_envs]
                    self.init_pool['obj_lin_vel_w'][motion_ids, time_steps] = self.candidate_states['obj_lin_vel_w'][valid_envs]
                    self.init_pool['obj_ang_vel_w'][motion_ids, time_steps] = self.candidate_states['obj_ang_vel_w'][valid_envs]
                    self.init_pool['last_action'][motion_ids, time_steps] = self.candidate_states['last_action'][valid_envs]
                    self.init_pool['flag'][motion_ids, time_steps] = True
               
            
            # Clear candidates
            self.candidate_valid[:] = False
            

        # Stage 2: save candidate snapshots k steps before the validation step and start recording trajectories
        if (self.count % update_interval == update_interval - self.validation_k) and self.count > (warmup_steps - self.validation_k):
            self.candidate_states['joint_pos'][:] = self.robot_joint_pos
            self.candidate_states['joint_vel'][:] = self.robot_joint_vel
            self.candidate_states['root_pos_w'][:] = self.robot_body_pos_w[:, 0] - self._env.scene.env_origins
            self.candidate_states['root_quat_w'][:] = self.robot_body_quat_w[:, 0]
            self.candidate_states['root_lin_vel_w'][:] = self.robot_body_lin_vel_w[:, 0]
            self.candidate_states['root_ang_vel_w'][:] = self.robot_body_ang_vel_w[:, 0]
            self.candidate_states['obj_pos_w'][:] = self.obj_pos_w - self._env.scene.env_origins
            self.candidate_states['obj_quat_w'][:] = self.obj_quat_w
            self.candidate_states['obj_lin_vel_w'][:] = self.obj_lin_vel_w
            self.candidate_states['obj_ang_vel_w'][:] = self.obj_ang_vel_w
            self.candidate_states['last_action'][:] = self._env.action_manager.action
            
            self.candidate_motion_id[:] = self.motion_id
            self.candidate_time_steps[:] = self.time_steps
            self.candidate_valid[:] = True  # Candidates for all envs start as valid
            
        
            
        # Increment timesteps
        self.time_steps += 1

        # ====== Rollout mode: collect trajectory data every step ======
        if self.cfg.rollout_traj:
            self._collect_rollout_step()

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)


    def _set_debug_vis_impl(self, debug_vis: bool):

        if debug_vis:
            if not hasattr(self, "current_body_visualizers"):
                self.current_obj_visualizer = VisualizationMarkers(
                    self.cfg.object_visualizer_cfg.replace(prim_path="/Visuals/Command/current/object")
                )
                self.goal_obj_visualizer = VisualizationMarkers(
                    self.cfg.object_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/object")
                )

                
                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )
                
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)


        else:
            if hasattr(self, "current_body_visualizers"):

                self.current_obj_visualizer.set_visibility(False)
                self.goal_obj_visualizer.set_visibility(False)

                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        
        self.current_obj_visualizer.visualize(self.obj_pos_w, self.obj_quat_w)
        self.goal_obj_visualizer.visualize(self.obj_ref_pos_w, self.obj_ref_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(
                self.body_pos_w[:, i], self.body_quat_w[:, i]
            )


    def get_future_index(self,future_frames=None): # Indices used for the main data; we no longer need to load raw data and refined data simultaneously, so one unified index is enough.
        # Used for the teacher when training the refiner and for distillation, with length refiner_future_frames.
        
        if future_frames is None:
            num_future_frames = self.cfg.future_frames
        else:
            num_future_frames = future_frames
        # Get motion length for each env
        motion_lengths = self.motion.time_step_total_permotion[self.motion_id]  # (num_envs,)
        
        # Calculate timestep indices: current + 0, 1, 2, 3, 4 (including the current frame)
        future_offsets = torch.arange(num_future_frames, device=self.device)  # (5,)
        future_timesteps = self.time_steps.unsqueeze(-1) + future_offsets.unsqueeze(0)  # (num_envs, 5)
        
        # Clamp to the valid range for each motion
        future_timesteps = future_timesteps.clamp(min=0).clamp_max(motion_lengths.unsqueeze(-1) - 1)
        batch_motion_ids = self.motion_id.unsqueeze(-1).expand(-1, num_future_frames)  # (num_envs, 5)
        return (batch_motion_ids, future_timesteps)

    @property
    def command(self) -> torch.Tensor: 
        return None

    @property
    def generated_command(self) -> Optional[dict]:
        """
        Get the generator command for the next future_frames steps.
        """
        if self.generator is None:
            return None
        
        future_frames = 1
        
        idx = (self.time_steps - 1) % self.generator_call_interval  # (num_envs,) long tensor
        
        # idx: (num_envs,), take future_frames frames for each env starting at idx
        # Build indices: e (num_envs, 1), t (num_envs, future_frames)
        e = torch.arange(self.num_envs, device=self.device).unsqueeze(1)
        t = idx.unsqueeze(1) + torch.arange(future_frames, device=self.device)
        
        buf = self.generated_command_buffer
        # import ipdb; ipdb.set_trace()
        self.last_command = buf['action'][e, t]
        return {
            'action':  buf['action'][e, t],
        }

    @property
    def teacher_joint_pos_vel_future(self) -> torch.Tensor:
        batch_motion_ids, future_timesteps = self.get_future_index()
        future_joint_pos = self.teacher_motion.joint_pos[batch_motion_ids, future_timesteps]
        future_joint_vel = self.teacher_motion.joint_vel[batch_motion_ids, future_timesteps]
        return torch.cat([future_joint_pos, future_joint_vel], dim=-1)

    @property
    def joint_pos_vel_future(self) -> torch.Tensor:
        batch_motion_ids, future_timesteps = self.get_future_index()
        
        # Index into motion data
        future_joint_pos = self.motion.joint_pos[batch_motion_ids, future_timesteps]  # (num_envs, 5, num_joints)
        future_joint_vel = self.motion.joint_vel[batch_motion_ids, future_timesteps]  # (num_envs, 5, num_joints)
        
        # Concatenate pos and vel: (num_envs, 5, num_joints * 2)
        future_joint_pos_vel = torch.cat([future_joint_pos, future_joint_vel], dim=-1)
        
        return future_joint_pos_vel

    @property
    def joint_pos(self) -> torch.Tensor:

        return self.motion.joint_pos[self.motion_id, self.time_steps]
    
    @property
    def joint_pos_future(self) -> torch.Tensor:
        """Get current + future frames of joint positions.
        
        Returns:
            torch.Tensor: Shape (num_envs, future_frames, num_joints)
        """
        batch_motion_ids, future_timesteps = self.get_future_index()

        future_joint_pos = self.motion.joint_pos[batch_motion_ids, future_timesteps]  # (num_envs, 5, num_joints)
        
        return future_joint_pos

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.motion_id, self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.motion_id, self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_pos_w_future(self) -> torch.Tensor:
        batch_motion_ids, future_timesteps = self.get_future_index()
        future_body_pos_w = self.motion.body_pos_w[batch_motion_ids, future_timesteps]  # (num_envs, 5, num_bodies, 3)
        future_body_pos_w = future_body_pos_w + self._env.scene.env_origins[:, None, None, :]  # Add env origins
        return future_body_pos_w

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.motion_id, self.time_steps]
    
    @property
    def body_quat_w_future(self) -> torch.Tensor:
        batch_motion_ids, future_timesteps = self.get_future_index()
        future_body_quat_w = self.motion.body_quat_w[batch_motion_ids, future_timesteps]  # (num_envs, 5, num_bodies, 4)
        return future_body_quat_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.motion_id, self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.motion_id, self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.motion_id, self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins
    
    @property
    def anchor_pos_w_final(self) -> torch.Tensor:
        motion_lengths = self.motion.time_step_total_permotion[self.motion_id] 
        return self.motion.body_pos_w[self.motion_id, motion_lengths-1, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.motion_id, self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_quat_w_final(self) -> torch.Tensor:
        motion_lengths = self.motion.time_step_total_permotion[self.motion_id] 
        return self.motion.body_quat_w[self.motion_id, motion_lengths-1, self.motion_anchor_body_index]
    
    
    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.motion_id, self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.motion_id, self.time_steps, self.motion_anchor_body_index]
    
    @property
    def anchor_lin_vel_b(self) -> torch.Tensor:
        # Directly use the corresponding values from refined data
        anchor_lin_vel_w = self.anchor_lin_vel_w
        anchor_quat_w = self.anchor_quat_w

        # Core transformation
        anchor_lin_vel_b = quat_apply_inverse(anchor_quat_w, anchor_lin_vel_w)
        
        return anchor_lin_vel_b
    
    @property
    def root_lin_vel_b(self) -> torch.Tensor:
        root_lin_vel_w = self.motion.body_lin_vel_w[self.motion_id, self.time_steps, 0]
        root_quat_w = self.motion.body_quat_w[self.motion_id, self.time_steps, 0]

        root_lin_vel_b = quat_apply_inverse(root_quat_w, root_lin_vel_w)
        return root_lin_vel_b

    @property
    def root_ang_vel_b(self) -> torch.Tensor:
        root_ang_vel_w = self.motion.body_ang_vel_w[self.motion_id, self.time_steps,0]
        root_quat_w = self.motion.body_quat_w[self.motion_id, self.time_steps, 0]

        root_ang_vel_b = quat_apply_inverse(root_quat_w, root_ang_vel_w)
        return root_ang_vel_b

    @property
    def anchor_ang_vel_b(self) -> torch.Tensor:
        # Directly use the corresponding values from refined data

        anchor_ang_vel_w = self.anchor_ang_vel_w
        anchor_quat_w = self.anchor_quat_w

        anchor_ang_vel_b = quat_apply_inverse(
            anchor_quat_w, 
            anchor_ang_vel_w
        )
        return anchor_ang_vel_b
    
    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]
    
    @property
    def robot_base_quat_w(self) -> torch.Tensor:
        return self.robot.data.root_quat_w
    
    @property
    def robot_base_pos_w(self) -> torch.Tensor:
        return self.robot.data.root_pos_w

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]
    
    @property
    def robot_anchor_lin_vel_b(self) -> torch.Tensor:
        """Robot anchor linear velocity in the body frame: (num_envs, 3)."""
        # Use quat_apply_inverse to project world-frame velocity into the local frame defined by the current body orientation
        return quat_apply_inverse(self.robot_anchor_quat_w, self.robot_anchor_lin_vel_w)
    
    @property
    def robot_anchor_ang_vel_b(self) -> torch.Tensor:
        """Robot anchor angular velocity in the body frame: (num_envs, 3)."""
        # The same operation applies to angular velocity
        return quat_apply_inverse(self.robot_anchor_quat_w, self.robot_anchor_ang_vel_w)


    # Object-related properties
    @property
    def obj_pos_w(self) -> torch.Tensor:
        return self.obj.data.root_pos_w
    
    @property    
    def obj_quat_w(self) -> torch.Tensor:
        return self.obj.data.root_quat_w

    @property
    def obj_lin_vel_w(self) -> torch.Tensor:
        return self.obj.data.root_lin_vel_w

    @property
    def obj_ang_vel_w(self) -> torch.Tensor:
        return self.obj.data.root_ang_vel_w

    @property
    def obj_ref_pos_w(self) -> torch.Tensor:
        return self.motion.obj_pos[self.motion_id, self.time_steps] + self._env.scene.env_origins
    
    @property
    def obj_ref_quat_w(self) -> torch.Tensor:
        return self.motion.obj_quat[self.motion_id, self.time_steps]

    @property
    def obj_ref_lin_vel_w(self) -> torch.Tensor:
        return self.motion.obj_lin_vel[self.motion_id, self.time_steps]

    @property
    def obj_ref_ang_vel_w(self) -> torch.Tensor:
        return self.motion.obj_ang_vel[self.motion_id, self.time_steps]

    @property
    def contact_label(self) -> torch.Tensor:
        return self.motion.contact_label[self.motion_id, self.time_steps]


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING
    obj_name: str = MISSING
    obj_mesh_scale: float = MISSING
    motion_folder: str = MISSING
    teacher_motion_folder: str = MISSING  # Optional teacher-motion data folder, used for teacher observations in bcppo

    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    joint_position_range: tuple[float, float] = (-0.1, 0.1)
    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    
    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    object_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    object_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
    
    
    # Future-observation frame-count configuration, including the current frame
    future_frames: int = MISSING
    
    # Whether to use reference initialization
    init_with_ref: bool = False
    
    # Rollout trajectory collection mode: used to collect simulation trajectories and save them as npz files
    rollout_traj: bool = False
    
    use_generator: bool = MISSING  
    generator_checkpoint_path: str = MISSING  
    generator_call_interval: int = None,

    pool_warmup_steps: int = MISSING
    start_init_env_ratio: float = MISSING
    pool_minref_steps: int = None
    pool_minref_ratio: float = None
    
    key_body_names: list[str] = MISSING 
    
    eval_mode: bool = False
    eval_max_time: int = 1501
    eval_random_motion: bool = False
    
    rollout_dir: str = None
    
    # Rollout trajectory collection configuration
    rollout_start_distance: int = None  
    rollout_window_length: int = None  
    
    