from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg

##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import sugar_rl.tasks.locomanip.mdp as mdp
from sugar_rl.assets.robots.unitree import UNITREE_G1_29DOF_MIMIC_ACTION_SCALE
from sugar_rl.assets.robots.unitree import UNITREE_G1_29DOF_MIMIC_CFG as ROBOT_CFG
from dataclasses import MISSING

import torch


@configclass
class BaseRobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    
    # objects
    obj: RigidObjectCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )

    left_foot_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link", 
            history_length=3, 
            track_air_time=True, 
            force_threshold=0.1,
            debug_vis=True,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Obj"], 
        )
    right_foot_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link", 
            history_length=3, 
            track_air_time=True, 
            force_threshold=0.1,
            debug_vis=True,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Obj"], 
        )
    
    left_hand_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_rubber_hand", 
            history_length=3, 
            track_air_time=True, 
            force_threshold=0.1,
            debug_vis=True,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Obj"], 
        )
    right_hand_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_rubber_hand", 
            history_length=3, 
            track_air_time=True, 
            force_threshold=0.1,
            debug_vis=True,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Obj"], 
        )

    left_hip_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_hip_pitch_link", 
            history_length=3, 
            track_air_time=True, 
            force_threshold=0.1,
            debug_vis=True,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Obj"], 
        )
    right_hip_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_hip_pitch_link", 
            history_length=3, 
            track_air_time=True, 
            force_threshold=0.1,
            debug_vis=True,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Obj"], 
        )
    pelvis_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/pelvis", 
            history_length=3, 
            track_air_time=True, 
            force_threshold=0.1,
            debug_vis=True,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Obj"], 
        )

##
# MDP settings
##


@configclass
class BaseCommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        obj_name="obj",
        obj_mesh_scale=1.0,
        
        motion_folder=None,  
        teacher_motion_folder=None,
        
        use_generator=True,
        generator_checkpoint_path=None,
        
        eval_mode=True,
        eval_random_motion=False,
        
        generator_call_interval=20,
        anchor_body_name="torso_link",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        pose_range={ 
            "x": (-0.0, 0.0),
            "y": (-0.0, 0.0),
            "z": (-0.0, 0.0),
            "roll": (-0.0, 0.0),
            "pitch": (-0.0, 0.0),
            "yaw": (-0.0, 0.0),
        },
        joint_position_range=(-0.0, 0.0),
        body_names=[
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ],
        key_body_names=[
            'torso_link',
            'left_ankle_roll_link',
            'right_ankle_roll_link',
            'left_wrist_yaw_link',
            'right_wrist_yaw_link',
        ],
        # Future observation frames
        future_frames=8,
        start_init_env_ratio=1.0,
        pool_warmup_steps=10000000*24,
        init_with_ref=False
    )


@configclass
class BaseActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[
            ".*_hip_.*",
            ".*_knee_.*",
            ".*_ankle_.*",
            "waist_.*",
            ".*_shoulder_.*",
            ".*_elbow_.*",
            ".*_wrist_.*",
            ], 
            scale=UNITREE_G1_29DOF_MIMIC_ACTION_SCALE,
    )

@configclass
class BaseObservationsCfg:
    @configclass

    @configclass
    class TrackerCfg(ObsGroup):
        """Tracker observations using generated command."""
        
        generated_command = ObsTerm(func=mdp.generated_command, params={"command_name": "motion"})
        
        base_ang_vel_history = ObsTerm(func=mdp.base_ang_vel, history_length=5, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos_history = ObsTerm(func=mdp.joint_pos_rel, history_length=5, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_history = ObsTerm(func=mdp.joint_vel_rel, history_length=5, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions_history = ObsTerm(func=mdp.last_action, history_length=5)
        project_gravity = ObsTerm(func=mdp.project_gravity, params={"command_name": "motion"},history_length=5, noise=Unoise(n_min=-0.2, n_max=0.2))
        obj_pos_b = ObsTerm(func=mdp.obj_pos_b, params={"command_name": "motion"},noise=Unoise(n_min=-0.02, n_max=0.02)) 
        obj_ori_b = ObsTerm(func=mdp.obj_ori_b, params={"command_name": "motion"},noise=Unoise(n_min=-0.1, n_max=0.1)) 
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True


    @configclass
    class PrivilegedCfg(ObsGroup):

        # future ref [0 1 2 3 4]
        joint_pos_vel_future = ObsTerm(func=mdp.joint_pos_vel_future, params={"command_name": "motion"}) 
        motion_anchor_pos_b_future = ObsTerm(func=mdp.motion_anchor_pos_b_future, params={"command_name": "motion"}) 
        motion_anchor_ori_b_future = ObsTerm(func=mdp.motion_anchor_ori_b_future, params={"command_name": "motion"}) 
        ref_obj_pos_b_future = ObsTerm(func=mdp.obj_motion_pos_future, params={"command_name": "motion"})  
        ref_obj_ori_b_future = ObsTerm(func=mdp.obj_motion_ori_future, params={"command_name": "motion"})  
        ref_obj_lin_vel_b_future = ObsTerm(func=mdp.ref_obj_lin_vel_b_future, params={"command_name": "motion"})  
        ref_obj_ang_vel_b_future = ObsTerm(func=mdp.ref_obj_ang_vel_b_future, params={"command_name": "motion"})  
        
        
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        obj_pos_b = ObsTerm(func=mdp.obj_pos_b, params={"command_name": "motion"})
        obj_ori_b = ObsTerm(func=mdp.obj_ori_b, params={"command_name": "motion"}) 
        obj_lin_vel_b = ObsTerm(func=mdp.obj_lin_vel_b, params={"command_name": "motion"})
        obj_ang_vel_b = ObsTerm(func=mdp.obj_ang_vel_b, params={"command_name": "motion"})

    
    policy: TrackerCfg = TrackerCfg()
    critic: PrivilegedCfg = PrivilegedCfg()
    

@configclass
class BaseEventCfg:
    """Configuration for events."""

    # # startup
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )
    # # startup
    obj_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("obj", body_names=".*"),
            "static_friction_range": (0.3, 0.8),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    obj_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("obj"),
            "operation": "scale",
            "mass_distribution_params": (0.5, 2.0),
            "distribution": "log_uniform",
            "recompute_inertia": True,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    
@configclass
class BaseRewardsCfg:
    """Reward terms for the MDP."""

    # -- base
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )


@configclass
class BaseTerminationsCfg:
    """Termination terms for the MDP."""

    timeout = DoneTerm(
        func=mdp.eval_timeout,
        params={"command_name": "motion"},
        time_out=False,
    )