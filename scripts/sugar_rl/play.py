# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--replay", action="store_true", default=False, help="Use RobotPlayEnvCfg instead of RobotDistillPlayEnvCfg for distill tasks.")
parser.add_argument("--generator_checkpoint", type=str, default=None, help="Path to the generator checkpoint.")
parser.add_argument("--eval_random_motion", action="store_true", default=False, help="Enable random motion evaluation.")
parser.add_argument("--eval_mode", action="store_true", default=False, help="Enable evaluation mode.")
parser.add_argument("--eval_max_time", type=int, default=None, help="Maximum evaluation time steps.")
parser.add_argument("--rollout_dir", type=str, default=None, help="Directory to save rollout data.")
parser.add_argument("--motion_folder", type=str, default=None, help="Path to motion folder for the environment.")
parser.add_argument("--teacher_motion_folder", type=str, default=None, help="Path to teacher motion folder for the environment.")
parser.add_argument("--teacher_ckpt", type=str, default=None, help="Path to teacher checkpoint for PPO-BC algorithm.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

import rsl_rl.algorithms
import builtins
from sugar_rl.utils.rsl_rl_bcppo import BCPPO
setattr(builtins, "BCPPO", BCPPO)
setattr(rsl_rl.algorithms, "BCPPO", BCPPO)

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

import sugar_rl.tasks  # noqa: F401
from sugar_rl.utils.parser_cfg import parse_env_cfg


def main():
    """Play with RSL-RL agent."""
    
    # Handle --replay parameter: force use RobotPlayEnvCfg
    if args_cli.replay:
        task_spec = gym.spec(args_cli.task)
        if task_spec and "play_env_cfg_entry_point" in task_spec.kwargs:
            # Extract the module path and force use RobotPlayEnvCfg
            original_cfg = task_spec.kwargs["play_env_cfg_entry_point"]
            module_path = original_cfg.rsplit(":", 1)[0]  # Get module path before ':'
            print(f"[INFO] Original play_env_cfg_entry_point: {original_cfg}")
            task_spec.kwargs["play_env_cfg_entry_point"] = f"{module_path}:RobotPlayEnvCfg"
            print(f"[INFO] --replay flag detected: Forcing use of RobotPlayEnvCfg")
            # raise SystemExit("Restart the script to apply the --replay flag changes.")
    
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )

    if args_cli.generator_checkpoint:
        if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion") and hasattr(env_cfg.commands.motion, "generator_checkpoint_path"):
            print(f"[INFO] Overriding generator checkpoint path: {args_cli.generator_checkpoint}")
            env_cfg.commands.motion.generator_checkpoint_path = args_cli.generator_checkpoint

    if args_cli.eval_random_motion:
        if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion") and hasattr(env_cfg.commands.motion, "eval_random_motion"):
            print(f"[INFO] Overriding eval_random_motion: True")
            env_cfg.commands.motion.eval_random_motion = True

    if args_cli.eval_mode:
        if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion") and hasattr(env_cfg.commands.motion, "eval_mode"):
            print(f"[INFO] Overriding eval_mode: True")
            env_cfg.commands.motion.eval_mode = True

    if args_cli.eval_max_time is not None:
        if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion") and hasattr(env_cfg.commands.motion, "eval_max_time"):
            print(f"[INFO] Overriding eval_max_time: {args_cli.eval_max_time}")
            env_cfg.commands.motion.eval_max_time = args_cli.eval_max_time
            
    if args_cli.rollout_dir is not None:
        if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion"):
            print(f"[INFO] Overriding rollout_dir: {args_cli.rollout_dir}")
            env_cfg.commands.motion.rollout_dir = args_cli.rollout_dir
            
    if getattr(args_cli, "motion_folder", None) is not None:
        if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion"):
            print(f"[INFO] Overriding motion_folder: {args_cli.motion_folder}")
            env_cfg.commands.motion.motion_folder = args_cli.motion_folder

    if getattr(args_cli, "teacher_motion_folder", None) is not None:
        if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion"):
            print(f"[INFO] Overriding teacher_motion_folder: {args_cli.teacher_motion_folder}")
            env_cfg.commands.motion.teacher_motion_folder = args_cli.teacher_motion_folder
    
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    if getattr(args_cli, "teacher_ckpt", None) is not None:
        if hasattr(agent_cfg, "algorithm") and hasattr(agent_cfg.algorithm, "teacher_ckpt"):
            print(f"[INFO] Overriding teacher_ckpt: {args_cli.teacher_ckpt}")
            agent_cfg.algorithm.teacher_ckpt = args_cli.teacher_ckpt

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # >>> 插入以下代码以打印真实的关节顺序 <<<
    print("\n" + "="*60)
    print("ISAACLAB TRUE JOINT ORDERS (CRITICAL FOR SIM2SIM)")
    print("="*60)
    robot = env.unwrapped.scene["robot"]

    # 1. 观测空间的关节顺序 (通常是 URDF 物理顺序)
    obs_joint_names = robot.data.joint_names
    print("1. Observation Joint Names (Physical/URDF Order):")
    for i, name in enumerate(obs_joint_names):
        print(f"  [{i:2d}] {name}")

    # 2. 动作空间的关节顺序 (正则展开后的真实顺序)
    action_term = env.unwrapped.action_manager._terms.get("JointPositionAction")
    if action_term:
        action_joint_names = action_term._joint_names
        print("\n2. Action Joint Names (Regex Expanded Order):")
        for i, name in enumerate(action_joint_names):
            print(f"  [{i:2d}] {name}")
    else:
        print("\n[WARNING] Could not find 'JointPositionAction' in action_manager._terms")
    print("="*60 + "\n")
    # >>> 插入结束 <<<
    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    # export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    # export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            
            if timestep == 0:
                import json
                import numpy as np
                try:
                    # 1) Dump frame 0 diagnostics (joint state)
                    robot = env.unwrapped.scene["robot"]
                    joint_names = robot.data.joint_names
                    q_actual = robot.data.joint_pos[0].cpu().numpy()
                    q_default = robot.data.default_joint_pos[0].cpu().numpy()
                    
                    try:
                        q_target = robot.data.joint_pos_target[0].cpu().numpy()
                    except AttributeError:
                        q_target = np.zeros_like(q_actual)

                    try:
                        import sys
                        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "source", "sugar_rl")))
                        from sugar_rl.assets.robots.unitree_config import (
                            UNITREE_G1_29DOF_MIMIC_ACTION_SCALE, STIFFNESS_5020, STIFFNESS_7520_14, STIFFNESS_7520_22, STIFFNESS_4010,
                            DAMPING_5020, DAMPING_7520_14, DAMPING_7520_22, DAMPING_4010
                        )
                        JOINT_KP = {
                            "left_hip_pitch_joint": STIFFNESS_7520_14, "left_hip_roll_joint": STIFFNESS_7520_22, "left_hip_yaw_joint": STIFFNESS_7520_14, "left_knee_joint": STIFFNESS_7520_22, "left_ankle_pitch_joint": 2.0 * STIFFNESS_5020, "left_ankle_roll_joint": 2.0 * STIFFNESS_5020,
                            "right_hip_pitch_joint": STIFFNESS_7520_14, "right_hip_roll_joint": STIFFNESS_7520_22, "right_hip_yaw_joint": STIFFNESS_7520_14, "right_knee_joint": STIFFNESS_7520_22, "right_ankle_pitch_joint": 2.0 * STIFFNESS_5020, "right_ankle_roll_joint": 2.0 * STIFFNESS_5020,
                            "waist_yaw_joint": STIFFNESS_7520_14, "waist_roll_joint": 2.0 * STIFFNESS_5020, "waist_pitch_joint": 2.0 * STIFFNESS_5020,
                            "left_shoulder_pitch_joint": STIFFNESS_5020, "left_shoulder_roll_joint": STIFFNESS_5020, "left_shoulder_yaw_joint": STIFFNESS_5020, "left_elbow_joint": STIFFNESS_5020, "left_wrist_roll_joint": STIFFNESS_5020, "left_wrist_pitch_joint": STIFFNESS_4010, "left_wrist_yaw_joint": STIFFNESS_4010,
                            "right_shoulder_pitch_joint": STIFFNESS_5020, "right_shoulder_roll_joint": STIFFNESS_5020, "right_shoulder_yaw_joint": STIFFNESS_5020, "right_elbow_joint": STIFFNESS_5020, "right_wrist_roll_joint": STIFFNESS_5020, "right_wrist_pitch_joint": STIFFNESS_4010, "right_wrist_yaw_joint": STIFFNESS_4010,
                        }
                        JOINT_KD = {
                            "left_hip_pitch_joint": DAMPING_7520_14, "left_hip_roll_joint": DAMPING_7520_22, "left_hip_yaw_joint": DAMPING_7520_14, "left_knee_joint": DAMPING_7520_22, "left_ankle_pitch_joint": 2.0 * DAMPING_5020, "left_ankle_roll_joint": 2.0 * DAMPING_5020,
                            "right_hip_pitch_joint": DAMPING_7520_14, "right_hip_roll_joint": DAMPING_7520_22, "right_hip_yaw_joint": DAMPING_7520_14, "right_knee_joint": DAMPING_7520_22, "right_ankle_pitch_joint": 2.0 * DAMPING_5020, "right_ankle_roll_joint": 2.0 * DAMPING_5020,
                            "waist_yaw_joint": DAMPING_7520_14, "waist_roll_joint": 2.0 * DAMPING_5020, "waist_pitch_joint": 2.0 * DAMPING_5020,
                            "left_shoulder_pitch_joint": DAMPING_5020, "left_shoulder_roll_joint": DAMPING_5020, "left_shoulder_yaw_joint": DAMPING_5020, "left_elbow_joint": DAMPING_5020, "left_wrist_roll_joint": DAMPING_5020, "left_wrist_pitch_joint": DAMPING_4010, "left_wrist_yaw_joint": DAMPING_4010,
                            "right_shoulder_pitch_joint": DAMPING_5020, "right_shoulder_roll_joint": DAMPING_5020, "right_shoulder_yaw_joint": DAMPING_5020, "right_elbow_joint": DAMPING_5020, "right_wrist_roll_joint": DAMPING_5020, "right_wrist_pitch_joint": DAMPING_4010, "right_wrist_yaw_joint": DAMPING_4010,
                        }
                    except Exception as e:
                        print("Could not load unitree_config", e)
                        UNITREE_G1_29DOF_MIMIC_ACTION_SCALE = {}
                        JOINT_KP = {}
                        JOINT_KD = {}

                    MJ_IDS = {'left_ankle_pitch_joint': {'actuator_id': 4, 'qpos_adr': 11}, 'left_ankle_roll_joint': {'actuator_id': 5, 'qpos_adr': 12}, 'left_elbow_joint': {'actuator_id': 18, 'qpos_adr': 25}, 'left_hip_pitch_joint': {'actuator_id': 0, 'qpos_adr': 7}, 'left_hip_roll_joint': {'actuator_id': 1, 'qpos_adr': 8}, 'left_hip_yaw_joint': {'actuator_id': 2, 'qpos_adr': 9}, 'left_knee_joint': {'actuator_id': 3, 'qpos_adr': 10}, 'left_shoulder_pitch_joint': {'actuator_id': 15, 'qpos_adr': 22}, 'left_shoulder_roll_joint': {'actuator_id': 16, 'qpos_adr': 23}, 'left_shoulder_yaw_joint': {'actuator_id': 17, 'qpos_adr': 24}, 'left_wrist_pitch_joint': {'actuator_id': 20, 'qpos_adr': 27}, 'left_wrist_roll_joint': {'actuator_id': 19, 'qpos_adr': 26}, 'left_wrist_yaw_joint': {'actuator_id': 21, 'qpos_adr': 28}, 'right_ankle_pitch_joint': {'actuator_id': 10, 'qpos_adr': 17}, 'right_ankle_roll_joint': {'actuator_id': 11, 'qpos_adr': 18}, 'right_elbow_joint': {'actuator_id': 25, 'qpos_adr': 32}, 'right_hip_pitch_joint': {'actuator_id': 6, 'qpos_adr': 13}, 'right_hip_roll_joint': {'actuator_id': 7, 'qpos_adr': 14}, 'right_hip_yaw_joint': {'actuator_id': 8, 'qpos_adr': 15}, 'right_knee_joint': {'actuator_id': 9, 'qpos_adr': 16}, 'right_shoulder_pitch_joint': {'actuator_id': 22, 'qpos_adr': 29}, 'right_shoulder_roll_joint': {'actuator_id': 23, 'qpos_adr': 30}, 'right_shoulder_yaw_joint': {'actuator_id': 24, 'qpos_adr': 31}, 'right_wrist_pitch_joint': {'actuator_id': 27, 'qpos_adr': 34}, 'right_wrist_roll_joint': {'actuator_id': 26, 'qpos_adr': 33}, 'right_wrist_yaw_joint': {'actuator_id': 28, 'qpos_adr': 35}, 'waist_pitch_joint': {'actuator_id': 14, 'qpos_adr': 21}, 'waist_roll_joint': {'actuator_id': 13, 'qpos_adr': 20}, 'waist_yaw_joint': {'actuator_id': 12, 'qpos_adr': 19}}

                    actions_np = actions[0].cpu().numpy()
                    diag_log = []
                    for i, name in enumerate(joint_names):
                        act_idx = i
                        raw_act = float(actions_np[act_idx]) if act_idx < len(actions_np) else 0.0
                        scale_val = float(UNITREE_G1_29DOF_MIMIC_ACTION_SCALE.get(name, 1.0))
                        kp_val = float(JOINT_KP.get(name, 0.0))
                        kd_val = float(JOINT_KD.get(name, 0.0))
                        
                        joint_diag = {
                            "name": name,
                            "q_actual": float(q_actual[i]),
                            "q_default": float(q_default[i]),
                            "q_target": float(q_target[i]),
                            "curr_joint_pos_rel": float(q_actual[i] - q_default[i]),
                            "raw_action": raw_act,
                            "action_urdf": raw_act,
                            "action_scale": scale_val,
                            "kp": kp_val,
                            "kd": kd_val,
                            "actuator_id": MJ_IDS.get(name, {}).get("actuator_id", -1),
                            "qpos_adr": MJ_IDS.get(name, {}).get("qpos_adr", -1),
                        }
                        diag_log.append(joint_diag)

                    with open("isaaclab_diagnostics.log", "w") as f:
                        f.write("--- FRAME 0 DIAGNOSTICS ---\n")
                        f.write(json.dumps(diag_log, indent=2))

                    # 2) Dump frame 0 observation diagnostics
                    raw_obs = env.unwrapped.observation_manager._obs_buffer["policy"]
                    
                    if isinstance(raw_obs, dict):
                        obs_np_dict = {k: v[0].cpu().numpy().tolist() for k, v in raw_obs.items()}
                        obs_diag = {
                            "tracker_obs_shape": [sum(len(v) if isinstance(v, list) else 1 for v in obs_np_dict.values())],
                            "components": obs_np_dict
                        }
                    else:
                        obs_np = raw_obs[0].cpu().numpy()
                        obs_diag = {
                            "tracker_obs_shape": [len(obs_np)],
                            "components": {
                                "generated_command": obs_np[0:36].tolist(),
                                "base_ang_vel_history": obs_np[36:51].tolist(),
                                "joint_pos_history": obs_np[51:196].tolist(),
                                "joint_vel_history": obs_np[196:341].tolist(),
                                "actions_history": obs_np[341:486].tolist(),
                                "project_gravity": obs_np[486:501].tolist(),
                                "obj_pos_b": obs_np[501:504].tolist(),
                                "obj_ori_b": obs_np[504:510].tolist()
                            }
                        }

                    with open("isaaclab_diagnostics.log", "a") as f:
                        f.write("\n\n--- FRAME 0 TRACKER OBSERVATION DIAGNOSTICS ---\n")
                        f.write(json.dumps(obs_diag, indent=2))
                    print("--- FRAME 0 DIAGNOSTICS WRITTEN TO isaaclab_diagnostics.log ---")
                except Exception as e:
                    print(f"Failed to dump diagnostics: {e}")
                
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
