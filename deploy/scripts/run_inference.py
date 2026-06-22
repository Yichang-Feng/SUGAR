"""
Hierarchical Policy Inference Runner — NO VISION, NO LANGUAGE.
基于纯状态输入（本体、物体位姿、目标点）的分层策略推理脚本。

通信架构 (全部基于 ZMQ):
  1. Proprioception -> ZMQ SUB (g1_debug 话题, 端口 5557)
  2. Object State   -> ZMQ SUB (object_state 话题, 端口 5558)  <-- 新增
  3. Actions out    -> ZMQ PUB (pose 话题, 端口 5556，直接发送 joint_pos)
  4. Keyboard       -> ZMQ SUB (端口 5580)
"""

import time
import queue
import os
import threading
import collections
from dataclasses import dataclass

import numpy as np
import torch
import tyro
import zmq
import msgpack

import re
from source.sugar_rl.sugar_rl.assets.robots.unitree_config import (
    UNITREE_G1_29DOF_MIMIC_ACTION_SCALE,
    JOINT_NAMES_29,
    INIT_JOINT_POS,
    STIFFNESS_5020,
    STIFFNESS_7520_14,
    STIFFNESS_7520_22,
    STIFFNESS_4010,
    DAMPING_5020,
    DAMPING_7520_14,
    DAMPING_7520_22,
    DAMPING_4010,
)

from deploy.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber
from deploy.utils.data_collection.keyboard_subscriber import ZMQKeyboardSubscriber, DEFAULT_ZMQ_KEYBOARD_PORT
from deploy.utils.data_collection.telemetry import Telemetry
from deploy.utils.data_collection.transforms import compute_projected_gravity


class HierarchicalPolicyWrapper:
    """
    分层策略加载与推理封装类
    """
    def __init__(self, high_path: str, low_path: str):
        self.high_path = high_path
        self.low_path = low_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        high_path_exp = os.path.expanduser(self.high_path)
        low_path_exp = os.path.expanduser(self.low_path)
        
        print(f"[PolicyWrapper] 加载高层策略模型: {high_path_exp}")
        if high_path_exp.endswith('.ckpt'):
            try:
                from sugar_il.wrapper.sugar_il_wrapper import GeneratorWrapper
                self.high_policy = GeneratorWrapper.load(checkpoint_path=high_path_exp, device=self.device)
                self.is_generator = True
            except ImportError:
                print("[PolicyWrapper] 警告: 未找到 sugar_il，尝试使用常规 torch.load 加载...")
                self.high_policy = torch.load(high_path_exp, map_location=self.device)
                self.is_generator = False
        else:
            self.high_policy = torch.jit.load(high_path_exp, map_location=self.device)
            self.is_generator = False
        
        print(f"[PolicyWrapper] 加载底层策略模型: {low_path_exp}")
        self.low_policy = torch.jit.load(low_path_exp, map_location=self.device)

    def get_high_level_action(
        self,
        high_obs: torch.Tensor = None,
        joint_pos: np.ndarray = None,
        project_gravity: np.ndarray = None,
        obj_pos_b: np.ndarray = None,
        obj_ori_b: np.ndarray = None,
        target_obj_pos_b: np.ndarray = None,
        target_obj_ori_b: np.ndarray = None,
        last_command: np.ndarray = None,
    ) -> np.ndarray:
        if self.is_generator:
            from sugar_il.wrapper.sugar_il_wrapper import GeneratorObs
            
            obj_pos_b_t = torch.from_numpy(obj_pos_b).float().to(self.device).view(1, 1, -1)
            obj_ori_b_t = torch.from_numpy(obj_ori_b).float().to(self.device).view(1, 1, -1)
            joint_pos_t = torch.from_numpy(joint_pos).float().to(self.device).view(1, 1, -1)
            project_gravity_t = torch.from_numpy(project_gravity).float().to(self.device).view(1, 1, -1)
            target_obj_pos_b_t = torch.from_numpy(target_obj_pos_b).float().to(self.device).view(1, 1, -1)
            target_obj_ori_b_t = torch.from_numpy(target_obj_ori_b).float().to(self.device).view(1, 1, -1)
            last_command_t = torch.from_numpy(last_command).float().to(self.device).view(1, 1, -1)
            
            obs = GeneratorObs(
                obj_pos_b=obj_pos_b_t,
                obj_ori_b=obj_ori_b_t,
                joint_pos=joint_pos_t,
                project_gravity=project_gravity_t,
                target_obj_pos_b=target_obj_pos_b_t,
                target_obj_ori_b=target_obj_ori_b_t,
                last_command=last_command_t,
            )
            
            action = self.high_policy.predict(obs)
            return action[0, 0].detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                return self.high_policy(high_obs.to(self.device)).detach().cpu().numpy().flatten()

    def get_low_level_action(self, low_obs: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return self.low_policy(low_obs.to(self.device)).detach().cpu().numpy().flatten()


@dataclass
class InferenceConfig:
    """分层推理配置"""
    action_publish_rate: int = 50
    """控制频率 (Hz)，必须与训练环境严格一致 (例如 0.02s)"""

    # ZMQ 网络配置
    state_zmq_host: str = "localhost"
    state_zmq_port: int = 5557
    
    action_zmq_host: str = "*"
    action_zmq_port: int = 5556

    # 新增：物体与目标状态订阅端口
    object_zmq_host: str = "localhost"
    object_zmq_port: int = 5558
    
    keyboard_zmq_host: str = "localhost"
    keyboard_zmq_port: int = DEFAULT_ZMQ_KEYBOARD_PORT

    # 策略模型路径
    high_level_policy_path: str = "~/SUGAR/demo_ckpts/CarryBox/generator.ckpt"
    low_level_policy_path: str = "~/SUGAR/demo_ckpts/CarryBox/policy.pt"
    
    # 调试模式
    debug_mode: str = "normal"  # "normal" | "zero_action" | "debug_obs" | "debug_scales"
    """
    调试模式:
    - normal: 正常运行
    - zero_action: 发送零动作测试控制器稳定性
    - debug_obs: 打印观测构建过程
    - debug_scales: 打印 action_scale 和初始位置
    """
    # 最大允许的原始动作幅度（在推理后进行裁剪，防止极端输出直接驱动机器人）
    max_raw_action: float = 1.0

def quat_wxyz_to_6d(quat):
    """严格对齐 process_tracker_rollout.py 中的 6D 姿态转换逻辑"""
    quat = np.asarray(quat, dtype=np.float64)
    r, i, j, k = quat
    two_s = 2.0 / np.sum(quat**2)
    rot_mat = np.array([
        [1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r)],
        [two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r)],
        [two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j)]
    ])
    # 取前两列并展平，得到 6 维向量
    return rot_mat[:, :2].flatten()

class ObsHistoryBuffer:
    """维护观测历史，以对齐 base_inference_env_cfg.py 中的 TrackerCfg"""
    def __init__(self, history_length=5, num_joints=29):
        self.history_length = history_length
        self.num_joints = num_joints
        
        # 初始化为零矩阵
        self.base_ang_vel_buf = collections.deque([np.zeros(3)] * history_length, maxlen=history_length)
        self.project_gravity_buf = collections.deque([np.array([0, 0, -1])] * history_length, maxlen=history_length)
        self.joint_pos_buf = collections.deque([np.zeros(num_joints)] * history_length, maxlen=history_length)
        self.joint_vel_buf = collections.deque([np.zeros(num_joints)] * history_length, maxlen=history_length)
        self.actions_buf = collections.deque([np.zeros(num_joints)] * history_length, maxlen=history_length)
        
        # 从配置获取默认关节位置
        self.default_joint_pos = np.zeros(num_joints)
        for i, name in enumerate(JOINT_NAMES_29):
            if name in INIT_JOINT_POS:
                self.default_joint_pos[i] = INIT_JOINT_POS[name]

    def update(self, proprio_msg, last_action):
        """将当前帧本体数据推入历史"""
        base_quat = np.array(proprio_msg["base_quat"], dtype=np.float64)
        gravity = compute_projected_gravity(base_quat)
        
        self.base_ang_vel_buf.append(np.array(proprio_msg.get("base_ang_vel", np.zeros(3))))
        self.project_gravity_buf.append(gravity)
        
        # 计算相对关节位置
        current_q = np.array(proprio_msg["body_q"])
        rel_q = current_q - self.default_joint_pos
        self.joint_pos_buf.append(rel_q)
        
        self.joint_vel_buf.append(np.array(proprio_msg["body_dq"]))
        self.actions_buf.append(last_action)
    
    def _get_hist(self, buf):
        """将 deque 转换为 oldest-to-newest 顺序 [t-4, t-3, t-2, t-1, t] 以与 Isaac Lab 的 flatten_history_dim 行为保持一致"""
        return np.concatenate(list(buf))
    
    def get_low_level_obs(self, high_level_cmd, obj_pos_b, obj_ori_b):
        """
        按照 base_inference_env_cfg.py 中的顺序拼接一维观测向量
        注意：此处顺序极其关键，拼错一个维度将导致策略失效。
        """
        obj_ori_6d = quat_wxyz_to_6d(obj_ori_b)
        obs_components = [
            np.array(high_level_cmd).flatten(),            # generated_command
            self._get_hist(self.base_ang_vel_buf),         # base_ang_vel_history
            self._get_hist(self.joint_pos_buf),            # joint_pos_history
            self._get_hist(self.joint_vel_buf),            # joint_vel_history
            self._get_hist(self.actions_buf),              # actions_history
            self._get_hist(self.project_gravity_buf),      # project_gravity history
            np.array(obj_pos_b).flatten(),                 # obj_pos_b
            obj_ori_6d                  # obj_ori_b
        ]
        
        flat_obs = np.concatenate(obs_components).astype(np.float32)
        return torch.from_numpy(flat_obs).unsqueeze(0)  # (1, obs_dim)

    def get_high_level_obs(self, proprio_msg, obj_pos_b, obj_ori_b, target_point):
        """为高层策略提取全局观测"""
        base_quat = np.array(proprio_msg["base_quat"], dtype=np.float64)
        gravity = compute_projected_gravity(base_quat)
        
        obs_components = [
            gravity,
            np.array(proprio_msg["body_q"]) - self.default_joint_pos,
            np.array(proprio_msg["body_dq"]),
            np.array(obj_pos_b),
            np.array(obj_ori_b),
            np.array(target_point)
        ]
        return torch.from_numpy(np.concatenate(obs_components).astype(np.float32)).unsqueeze(0)


def pack_joint_action_message(joint_pos: np.ndarray, frame_index: int) -> bytes:
    """将预测的关节位置打包，发送给底层的 ZMQ socket"""
    joint_pos = np.asarray(joint_pos, dtype=np.float32)
        
    pose_data = {
        "joint_pos": joint_pos.tolist(),
        "frame_index": frame_index,
    }
    return b"pose" + msgpack.packb(pose_data)


def main(config: InferenceConfig):
    # 1. 网络初始化
    state_subscriber = ZMQStateSubscriber(host=config.state_zmq_host, port=config.state_zmq_port)
    keyboard_listener = ZMQKeyboardSubscriber(port=config.keyboard_zmq_port, host=config.keyboard_zmq_host)
    
    zmq_context = zmq.Context()
    
    # 动作发布器
    action_socket = zmq_context.socket(zmq.PUB)
    action_socket.bind(f"tcp://{config.action_zmq_host}:{config.action_zmq_port}")
    
    # 新增：物体状态订阅器
    obj_socket = zmq_context.socket(zmq.SUB)
    obj_socket.connect(f"tcp://{config.object_zmq_host}:{config.object_zmq_port}")
    obj_socket.setsockopt_string(zmq.SUBSCRIBE, "object_state")

    # 2. 策略与状态初始化
    policy_wrapper = HierarchicalPolicyWrapper(
        high_path=config.high_level_policy_path,
        low_path=config.low_level_policy_path
    )
    obs_buffer = ObsHistoryBuffer(history_length=5, num_joints=29)
    telemetry = Telemetry(window_size=100)
    
    # === 调试信息 ===
    if config.debug_mode == "debug_scales":
        print("\n" + "="*80)
        print("ACTION SCALE DEBUG INFO")
        print("="*80)
        print(f"默认关节位置 (default_joint_pos):\n{obs_buffer.default_joint_pos}")
        print(f"\nJoint Names ({len(JOINT_NAMES_29)}):")
        for i, name in enumerate(JOINT_NAMES_29):
            scale = UNITREE_G1_29DOF_MIMIC_ACTION_SCALE.get(name, 0.0)
            default_pos = obs_buffer.default_joint_pos[i]
            print(f"  [{i:2d}] {name:30s} scale={scale:.6f} default_pos={default_pos:.6f}")
        print("="*80 + "\n")
    
    loop_period = 1.0 / config.action_publish_rate
    pause_loop = False
    frame_counter = 0
    last_action = np.zeros(29)
    
    # 最近一次的物体和目标状态缓存
    last_obj_pos = np.array([0.5, 0.0, 0.5])  # 假设箱子在正前方 0.5m，离地 0.5m
    last_obj_ori = np.array([1, 0, 0, 0])
    last_target = np.array([1.0, 0.0, 0.0])
    last_high_level_cmd = np.zeros(36, dtype=np.float32)

    print(f"🚀 系统已启动. 控制频率: {config.action_publish_rate}Hz. 调试模式: {config.debug_mode}")

    try:
        while True:
            t_start = time.monotonic()
            
            # --- 键盘控制逻辑 ---
            key = keyboard_listener.read_msg()
            if key == "p":
                pause_loop = not pause_loop
                print(f"[{'Paused' if pause_loop else 'Resumed'}]")
            
            # --- 读取最新物体状态 (非阻塞) ---
            try:
                # 假设对方发送的是 msgpack 打包的字典
                while True: 
                    raw_obj_msg = obj_socket.recv(zmq.NOBLOCK)
                    obj_data = msgpack.unpackb(raw_obj_msg[len(b"object_state"):])
                    last_obj_pos = obj_data["obj_pos_b"]
                    last_obj_ori = obj_data["obj_ori_b"]
                    last_target = obj_data["target_point"]
            except zmq.Again:
                pass # 保持上一次的值

            # --- 获取本体感觉 ---
            proprio_msg = state_subscriber.get_msg()
            if proprio_msg is None:
                time.sleep(0.001)
                continue

            # 更新历史 Buffer
            obs_buffer.update(proprio_msg, last_action)

            if pause_loop:
                time.sleep(0.1)
                continue

            with telemetry.timer("total_loop"):
                # --- 构建观测 ---
                high_obs = obs_buffer.get_high_level_obs(proprio_msg, last_obj_pos, last_obj_ori, last_target)

                # --- 策略推理 (可在 zero_action 模式下跳过) ---
                if config.debug_mode == "zero_action":
                    # 跳过所有策略推理，发送默认站立位以验证物理/控制器稳定性
                    high_level_cmd = np.zeros(36, dtype=np.float32)
                    raw_low_action = np.zeros(29, dtype=np.float32)
                    if frame_counter % 50 == 0:
                        print(f"[Frame {frame_counter}] zero_action: 跳过策略，发送默认姿态")
                else:
                    # 高层推理
                    if policy_wrapper.is_generator:
                        base_quat = np.array(proprio_msg["base_quat"], dtype=np.float64)
                        gravity = compute_projected_gravity(base_quat)
                        rel_q = np.array(proprio_msg["body_q"]) - obs_buffer.default_joint_pos
                        target_obj_ori_b = np.array([base_quat[0], -base_quat[1], -base_quat[2], -base_quat[3]], dtype=np.float32)
                        
                        high_level_cmd = policy_wrapper.get_high_level_action(
                            joint_pos=rel_q,
                            project_gravity=gravity,
                            obj_pos_b=last_obj_pos,
                            obj_ori_b=last_obj_ori,
                            target_obj_pos_b=last_target,
                            target_obj_ori_b=target_obj_ori_b,
                            last_command=last_high_level_cmd,
                        )
                        last_high_level_cmd = high_level_cmd.copy()
                    else:
                        high_level_cmd = policy_wrapper.get_high_level_action(high_obs=high_obs)

                    if config.debug_mode == "debug_obs" and frame_counter % 50 == 0:
                        print(f"\n[Frame {frame_counter}] 高层观测:")
                        print(f"  high_obs shape: {high_obs.shape}")
                        print(f"  high_level_cmd (前5): {high_level_cmd[:5]}")

                    # 低层推理
                    low_obs = obs_buffer.get_low_level_obs(high_level_cmd, last_obj_pos, last_obj_ori)
                    raw_low_action = policy_wrapper.get_low_level_action(low_obs)

                # --- 动作后处理：裁剪、缩放与日志 ---
                action_scale = np.array([
                    UNITREE_G1_29DOF_MIMIC_ACTION_SCALE.get(name, 0.0) for name in JOINT_NAMES_29
                ], dtype=np.float32)

                # 记录原始动作统计
                raw_min, raw_max = float(np.min(raw_low_action)), float(np.max(raw_low_action))

                # 裁剪原始动作到合理范围，防止极端输出直接驱动机器人
                raw_low_action_clipped = np.clip(raw_low_action, -config.max_raw_action, config.max_raw_action)

                if config.debug_mode in ("debug_obs", "debug_scales") and frame_counter % 50 == 0:
                    print(f"[Frame {frame_counter}] raw_low_action min/max: {raw_min:.6f}/{raw_max:.6f}")
                    clipped_min, clipped_max = float(np.min(raw_low_action_clipped)), float(np.max(raw_low_action_clipped))
                    print(f"[Frame {frame_counter}] clipped raw_low_action min/max: {clipped_min:.6f}/{clipped_max:.6f}")
                    print(f"[Frame {frame_counter}] action_scale (前5): {action_scale[:5]}")

                # 计算目标关节位置并可选裁剪到安全范围
                target_joint_pos = obs_buffer.default_joint_pos + (raw_low_action_clipped * action_scale)

                # 对目标关节位置做简单边界保护（默认位置 +/- 1.5 rad）
                joint_limit = 1.5
                target_joint_pos = np.clip(target_joint_pos, obs_buffer.default_joint_pos - joint_limit, obs_buffer.default_joint_pos + joint_limit)

                if config.debug_mode in ("debug_obs", "debug_scales") and frame_counter % 50 == 0:
                    print(f"[Frame {frame_counter}] target_joint_pos (前5): {target_joint_pos[:5]}")

                last_action = raw_low_action_clipped  # 保存给下一帧的 obs_buffer 使用

                # --- 发送指令 ---
                zmq_message = pack_joint_action_message(target_joint_pos, frame_counter)
                action_socket.send(zmq_message)
                frame_counter += 1

            # 严格的时间步长控制
            elapsed = time.monotonic() - t_start
            remaining = loop_period - elapsed
            if remaining > 0:
                time.sleep(remaining)
            else:
                telemetry.log_timing_info(context="Control Loop Missed Deadline", threshold=0.0)

    except KeyboardInterrupt:
        print("终止推理循环.")
    finally:
        action_socket.close()
        obj_socket.close()
        zmq_context.term()
        state_subscriber.close()
        keyboard_listener.close()

if __name__ == "__main__":
    config = tyro.cli(InferenceConfig)
    main(config)