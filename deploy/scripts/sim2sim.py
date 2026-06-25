#!/usr/bin/env python3
import os
import sys
import argparse
import pickle
import time
import numpy as np
import torch
import random
import threading

import threading
import zmq
import json

# Configure sys.path to find local packages
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "source", "sugar_il"))
sys.path.append(os.path.join(ROOT_DIR, "source", "sugar_rl"))

import mujoco
import mujoco.viewer

# Load G1 robot configurations
from source.sugar_rl.sugar_rl.assets.robots.unitree_config import (
    INIT_JOINT_POS,
    INIT_POS,
    UNITREE_G1_29DOF_MIMIC_ACTION_SCALE,
    STIFFNESS_5020,
    STIFFNESS_7520_14,
    STIFFNESS_7520_22,
    STIFFNESS_4010,
    DAMPING_5020,
    DAMPING_7520_14,
    DAMPING_7520_22,
    DAMPING_4010,
)

# The order of joints output by the policy action (based on JointPositionActionCfg regex order)
ACTION_JOINT_ORDER = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint"
]
JOINT_NAMES_29 = ACTION_JOINT_ORDER
ACTION_JOINT_NAMES = ACTION_JOINT_ORDER 

# Mappings from joint name to stiffness (Kp) and damping (Kd) constants
JOINT_KP = {
    "left_hip_pitch_joint": STIFFNESS_7520_14,
    "left_hip_roll_joint": STIFFNESS_7520_22,
    "left_hip_yaw_joint": STIFFNESS_7520_14,
    "left_knee_joint": STIFFNESS_7520_22,
    "left_ankle_pitch_joint": 2.0 * STIFFNESS_5020,
    "left_ankle_roll_joint": 2.0 * STIFFNESS_5020,
    
    "right_hip_pitch_joint": STIFFNESS_7520_14,
    "right_hip_roll_joint": STIFFNESS_7520_22,
    "right_hip_yaw_joint": STIFFNESS_7520_14,
    "right_knee_joint": STIFFNESS_7520_22,
    "right_ankle_pitch_joint": 2.0 * STIFFNESS_5020,
    "right_ankle_roll_joint": 2.0 * STIFFNESS_5020,
    
    "waist_yaw_joint": STIFFNESS_7520_14,
    "waist_roll_joint": 2.0 * STIFFNESS_5020,
    "waist_pitch_joint": 2.0 * STIFFNESS_5020,
    
    "left_shoulder_pitch_joint": STIFFNESS_5020,
    "left_shoulder_roll_joint": STIFFNESS_5020,
    "left_shoulder_yaw_joint": STIFFNESS_5020,
    "left_elbow_joint": STIFFNESS_5020,
    "left_wrist_roll_joint": STIFFNESS_5020,
    "left_wrist_pitch_joint": STIFFNESS_4010,
    "left_wrist_yaw_joint": STIFFNESS_4010,
    
    "right_shoulder_pitch_joint": STIFFNESS_5020,
    "right_shoulder_roll_joint": STIFFNESS_5020,
    "right_shoulder_yaw_joint": STIFFNESS_5020,
    "right_elbow_joint": STIFFNESS_5020,
    "right_wrist_roll_joint": STIFFNESS_5020,
    "right_wrist_pitch_joint": STIFFNESS_4010,
    "right_wrist_yaw_joint": STIFFNESS_4010,
}

JOINT_KD = {
    "left_hip_pitch_joint": DAMPING_7520_14,
    "left_hip_roll_joint": DAMPING_7520_22,
    "left_hip_yaw_joint": DAMPING_7520_14,
    "left_knee_joint": DAMPING_7520_22,
    "left_ankle_pitch_joint": 2.0 * DAMPING_5020,
    "left_ankle_roll_joint": 2.0 * DAMPING_5020,
    
    "right_hip_pitch_joint": DAMPING_7520_14,
    "right_hip_roll_joint": DAMPING_7520_22,
    "right_hip_yaw_joint": DAMPING_7520_14,
    "right_knee_joint": DAMPING_7520_22,
    "right_ankle_pitch_joint": 2.0 * DAMPING_5020,
    "right_ankle_roll_joint": 2.0 * DAMPING_5020,
    
    "waist_yaw_joint": DAMPING_7520_14,
    "waist_roll_joint": 2.0 * DAMPING_5020,
    "waist_pitch_joint": 2.0 * DAMPING_5020,
    
    "left_shoulder_pitch_joint": DAMPING_5020,
    "left_shoulder_roll_joint": DAMPING_5020,
    "left_shoulder_yaw_joint": DAMPING_5020,
    "left_elbow_joint": DAMPING_5020,
    "left_wrist_roll_joint": DAMPING_5020,
    "left_wrist_pitch_joint": DAMPING_4010,
    "left_wrist_yaw_joint": DAMPING_4010,
    
    "right_shoulder_pitch_joint": DAMPING_5020,
    "right_shoulder_roll_joint": DAMPING_5020,
    "right_shoulder_yaw_joint": DAMPING_5020,
    "right_elbow_joint": DAMPING_5020,
    "right_wrist_roll_joint": DAMPING_5020,
    "right_wrist_pitch_joint": DAMPING_4010,
    "right_wrist_yaw_joint": DAMPING_4010,
}

# Quaternion helpers
from scipy.spatial.transform import Rotation

def quat_inv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def quat_apply(q, v):
    w = q[0]
    x = q[1:]
    t = 2.0 * np.cross(x, v)
    return v + w * t + np.cross(x, t)

def quat_apply_inverse(q, v):
    w = q[0]
    x = q[1:]
    t = 2.0 * np.cross(x, v)
    return v - w * t + np.cross(x, t)

# Rotation Matrix to Quaternion (wxyz)
def matrix_to_quat(R):
    r = Rotation.from_matrix(R)
    q_xyzw = r.as_quat()
    if R.ndim == 3:
        q_wxyz = np.zeros_like(q_xyzw)
        q_wxyz[:, 0] = q_xyzw[:, 3]
        q_wxyz[:, 1:] = q_xyzw[:, :3]
    else:
        q_wxyz = np.zeros_like(q_xyzw)
        q_wxyz[0] = q_xyzw[3]
        q_wxyz[1:] = q_xyzw[:3]
    return q_wxyz

class ZmqVisionClient:
    def __init__(self):
        self.context = zmq.Context()
        self.image_pub = self.context.socket(zmq.PUB)
        self.image_pub.bind("tcp://127.0.0.1:5555")
        
        self.pose_sub = self.context.socket(zmq.SUB)
        self.pose_sub.bind("tcp://127.0.0.1:5556")
        self.pose_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        
        self.obj_pose = None
        self.tgt_pose = None
        self.running = True
        
        self.thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.thread.start()

    def _recv_loop(self):
        while self.running:
            try:
                msg = self.pose_sub.recv_json()
                if "tag_0" in msg:
                    self.obj_pose = msg["tag_0"]
                if "tag_1" in msg:
                    self.tgt_pose = msg["tag_1"]
            except Exception:
                pass

    def publish_image(self, rgb_image, time_ns=0):
        md = dict(
            dtype=str(rgb_image.dtype),
            shape=rgb_image.shape,
        )
        self.image_pub.send_json(md, zmq.SNDMORE)
        self.image_pub.send(rgb_image.tobytes())

class TrackerPolicy:
    def __init__(self, model_path, device="cuda"):
        self.model_path = model_path
        self.device = device
        
        # If tracker.pt was passed directly, try to locate policy.pt or policy.onnx in exported/
        if model_path.endswith("tracker.pt"):
            exported_pt = os.path.join(os.path.dirname(model_path), "exported", "policy.pt")
            exported_onnx = os.path.join(os.path.dirname(model_path), "exported", "policy.onnx")
            if os.path.exists(exported_pt):
                print(f"[Tracker] Resolving 'tracker.pt' to JIT compiled policy: {exported_pt}")
                model_path = exported_pt
            elif os.path.exists(exported_onnx):
                print(f"[Tracker] Resolving 'tracker.pt' to ONNX policy: {exported_onnx}")
                model_path = exported_onnx
            else:
                raise FileNotFoundError(f"Could not locate exported JIT or ONNX policy under {os.path.dirname(model_path)}/exported/")

        if model_path.endswith(".onnx"):
            import onnxruntime as ort
            providers = ['CPUExecutionProvider']
            if device == "cuda" and 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = ['CUDAExecutionProvider'] + providers
            print(f"[Tracker] Loading ONNX policy from: {model_path} (providers: {providers})")
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.is_onnx = True
        else:
            print(f"[Tracker] Loading JIT policy from: {model_path}")
            self.model = torch.jit.load(model_path, map_location=device)
            self.model.eval()
            self.is_onnx = False
            
    def __call__(self, obs_np):
        # obs_np shape: (510,) -> model input: (1, 510)
        obs_input = np.expand_dims(obs_np, axis=0)
        if self.is_onnx:
            inputs = {self.session.get_inputs()[0].name: obs_input.astype(np.float32)}
            outputs = self.session.run(None, inputs)
            return outputs[0][0]  # shape (29,)
        else:
            obs_t = torch.tensor(obs_input, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                actions_t = self.model(obs_t)
            return actions_t.cpu().numpy()[0]  # shape (29,)

def load_reference_motion(task, motion_id):
    motion_dir = os.path.join(ROOT_DIR, "data", task, f"data_{motion_id:03d}")
    robot_path = os.path.join(motion_dir, "robot_50hz.npz")
    obj_path = os.path.join(motion_dir, "obj_motion_global_50hz.pkl")
    label_path = os.path.join(motion_dir, "contact_labels_50hz.npy")
    
    if not os.path.exists(robot_path) or not os.path.exists(obj_path):
        raise FileNotFoundError(f"Reference files not found in {motion_dir}")
        
    print(f"[DataLoader] Loading reference trajectory from: {motion_dir}")
    robot_data = np.load(robot_path)
    # # >>> 检查 npz 顺序 <<<
    # print(f"\n=== NPZ DATA INSPECTION ===")
    # print(f"Keys in npz: {robot_data.files}")
    # print(f"joint_pos shape: {robot_data['joint_pos'].shape}")
    # print(f"Frame 0 joint_pos values:\n{robot_data['joint_pos'][0]}")
    # print("===========================\n")

    with open(obj_path, 'rb') as f:
        obj_data = pickle.load(f)
        
    if os.path.exists(label_path):
        contact_labels = np.load(label_path)
    else:
        contact_labels = np.zeros(robot_data["joint_pos"].shape[0], dtype=bool)
        
    # Convert object rotation matrix to quaternion
    obj_rot = obj_data["obj_rot"]  # (N, 3, 3)
    obj_quat = matrix_to_quat(obj_rot)  # (N, 4) in wxyz format
    
    # Align lengths
    num_frames = min(robot_data["joint_pos"].shape[0], obj_data["obj_trans"].shape[0], contact_labels.shape[0])
    

    # Map the raw unsorted npz data (in JOINT_NAMES_29_UNSORTED order) to the alphabetically sorted JOINT_NAMES_29
    joint_pos_urdf = robot_data["joint_pos"][:num_frames, :29].astype(np.float32)
    joint_vel_urdf = robot_data["joint_vel"][:num_frames, :29].astype(np.float32)
    
    ref_data = {
        "joint_pos": joint_pos_urdf,
        "joint_pos_urdf": robot_data["joint_pos"][:num_frames],
        "joint_vel": joint_vel_urdf,
        "robot_body_pos_w": robot_data["body_pos_w"][:num_frames],
        "robot_body_quat_w": robot_data["body_quat_w"][:num_frames],
        "robot_body_lin_vel_w": robot_data["body_lin_vel_w"][:num_frames],
        "robot_body_ang_vel_w": robot_data["body_ang_vel_w"][:num_frames],
        "obj_pos": obj_data["obj_trans"][:num_frames],
        "obj_quat": obj_quat[:num_frames],
        "obj_lin_vel": obj_data["obj_lin_vel"][:num_frames],
        "obj_ang_vel": obj_data["obj_ang_vel"][:num_frames],
        "contact_label": contact_labels[:num_frames],
    }
    print(f"[DataLoader] Successfully loaded {num_frames} frames (reordered to alphabetical).")
    return ref_data

def main():
    parser = argparse.ArgumentParser(description="G1 MuJoCo Sim2Sim Deployment Script")
    parser.add_argument("--task", type=str, default="CarryBox", help="Task name, e.g. CarryBox")
    parser.add_argument("--mode", type=str, choices=["full", "tracker"], default="full", 
                        help="full: runs high-level + low-level; tracker: runs low-level tracking ground-truth reference trajectory")
    parser.add_argument("--motion_id", type=int, default=0, help="Reference motion ID from data folder")
    parser.add_argument("--tracker_model", type=str, default=None, 
                        help="Path to tracker ONNX/JIT model (defaults to demo_ckpts/{task}/exported/policy.onnx)")
    parser.add_argument("--generator_model", type=str, default=None, 
                        help="Path to generator model checkpoint (defaults to demo_ckpts/{task}/generator.ckpt)")
    parser.add_argument("--sim_dt", type=float, default=0.002, help="MuJoCo physics simulation timestep (default: 0.002s)")
    parser.add_argument("--control_dt", type=float, default=0.02, help="Control loop timestep (default: 0.02s)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for policy running")
    parser.add_argument("--headless", action="store_true", help="Run simulation in headless mode (no passive viewer window)")
    parser.add_argument("--use_vision", action="store_true", help="Use vision to get object and target pose")
    args, unknown = parser.parse_known_args()

    # Load reference motion data
    ref_data = load_reference_motion(args.task, args.motion_id)
    num_frames = ref_data["joint_pos"].shape[0] + 500
    if args.use_vision:
        num_frames = 9999999

    # Resolve default paths
    if args.tracker_model is None:
        args.tracker_model = os.path.join(ROOT_DIR, "demo_ckpts", args.task, "exported", "policy.onnx")
        if not os.path.exists(args.tracker_model):
            args.tracker_model = os.path.join(ROOT_DIR, "demo_ckpts", args.task, "exported", "policy.pt")
            
    if args.generator_model is None and args.mode == "full":
        args.generator_model = os.path.join(ROOT_DIR, "demo_ckpts", args.task, "generator.ckpt")

    # Load policies
    print(f"[Policy] Running on device: {args.device}")
    tracker_policy = TrackerPolicy(args.tracker_model, device=args.device)
    
    generator = None
    if args.mode == "full":
        from sugar_il.wrapper.sugar_il_wrapper import GeneratorWrapper, GeneratorObs
        generator = GeneratorWrapper.load(args.generator_model, device=args.device)
        print(f"[Generator] Policy loaded successfully.")

    # Load MuJoCo Model
    xml_path = os.path.join(ROOT_DIR, "deploy", "data", "robot_model", "model_data", "g1", "g1_29dof_rev_1_0_box.xml")
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"MuJoCo XML model not found at {xml_path}")
    print(f"[MuJoCo] Loading XML model from: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Configure simulation dt
    model.opt.timestep = args.sim_dt
    decimation = int(round(args.control_dt / args.sim_dt))
    print(f"[MuJoCo] Timestep: {args.sim_dt}s. Control Frequency: {1.0/args.control_dt:.1f}Hz (decimation: {decimation} physics steps)")

    if args.use_vision:
        sim_node = ZmqVisionClient()
        print("[Sim2Sim] Started ZmqVisionClient.")
        renderer = mujoco.Renderer(model, 480, 640)
    else:
        sim_node = None
        renderer = None

    # Retrieve Joint Mappings
    joint_qpos_indices = []
    joint_qvel_indices = []
    actuator_indices = []
    for joint_name in JOINT_NAMES_29:
        joint_id = model.joint(joint_name).id
        actuator_id = model.actuator(joint_name).id
        
        qpos_addr = model.joint(joint_name).qposadr[0]
        dof_addr = model.joint(joint_name).dofadr[0]
        
        joint_qpos_indices.append(qpos_addr)
        joint_qvel_indices.append(dof_addr)
        actuator_indices.append(actuator_id)

    # Get body ids
    pelvis_body_id = model.body("pelvis").id
    torso_body_id = model.body("torso_link").id
    box_body_id = model.body("box").id
    
    box_qpos_adr = model.joint("floating_box_base_joint").qposadr[0]
    box_qvel_adr = model.joint("floating_box_base_joint").dofadr[0]

    # Map Kp and Kd arrays matching the joint order
    kp_array = np.array([JOINT_KP[name] for name in JOINT_NAMES_29])
    kd_array = np.array([JOINT_KD[name] for name in JOINT_NAMES_29])
    action_scale_array = np.array([UNITREE_G1_29DOF_MIMIC_ACTION_SCALE[name] for name in JOINT_NAMES_29])
    
    # Default joint pos mapping
    q_default = np.zeros(29)
    for i, name in enumerate(JOINT_NAMES_29):
        if name in INIT_JOINT_POS:
            q_default[i] = INIT_JOINT_POS[name]

    # Reset helper
    def reset_to_frame(frame_idx):
        p_pelvis = ref_data["robot_body_pos_w"][frame_idx, 0]
        q_pelvis = ref_data["robot_body_quat_w"][frame_idx, 0]
        
        data.qpos[0:3] = p_pelvis
        data.qpos[3:7] = q_pelvis
        data.qvel[0:3] = ref_data["robot_body_lin_vel_w"][frame_idx, 0]
        data.qvel[3:6] = ref_data["robot_body_ang_vel_w"][frame_idx, 0]
        
        for idx, (qpos_adr, dof_adr) in enumerate(zip(joint_qpos_indices, joint_qvel_indices)):
            data.qpos[qpos_adr] = ref_data["joint_pos"][frame_idx, idx]
            data.qvel[dof_adr] = ref_data["joint_vel"][frame_idx, idx]
            
        data.qpos[box_qpos_adr:box_qpos_adr+3] = ref_data["obj_pos"][frame_idx]
        data.qpos[box_qpos_adr+3:box_qpos_adr+7] = ref_data["obj_quat"][frame_idx]
        data.qvel[box_qvel_adr+3:box_qvel_adr+6] = ref_data["obj_ang_vel"][frame_idx]
        
        if args.use_vision and frame_idx == 0:
            data.qpos[box_qpos_adr] += random.uniform(-0.1, 0.1)
            data.qpos[box_qpos_adr+1] += random.uniform(-0.1, 0.1)
            # Make the box upright so the tag stays on top, and place it stably on the ground
            data.qpos[box_qpos_adr+2] = 0.175
            data.qpos[box_qpos_adr+3:box_qpos_adr+7] = [1.0, 0.0, 0.0, 0.0]
            data.qvel[box_qvel_adr:box_qvel_adr+6] = 0.0
        
        data.ctrl[:] = 0
        mujoco.mj_forward(model, data)

    # Initialize state
    reset_to_frame(0)

    # History Queues for Tracker (length 5)
    base_ang_vel_hist = []
    joint_pos_rel_hist = []
    joint_vel_hist = []
    actions_hist = []
    project_gravity_hist = []

    # Get initial values
    R_pelvis = data.xmat[pelvis_body_id].reshape(3, 3)
    initial_base_ang_vel = data.qvel[3:6].copy()
    
    q_actual = np.array([data.qpos[adr] for adr in joint_qpos_indices])
    v_actual = np.array([data.qvel[adr] for adr in joint_qvel_indices])
    initial_joint_pos_rel = q_actual - q_default
    initial_joint_vel = v_actual
    initial_project_gravity = R_pelvis.T @ np.array([0, 0, -1.0])

    initial_action_regex = np.zeros(29)
    for i, name in enumerate(ACTION_JOINT_NAMES):
        alpha_idx = JOINT_NAMES_29.index(name)
        initial_action_regex[i] = initial_joint_pos_rel[alpha_idx] / action_scale_array[alpha_idx]

    # Fill History queues
    for _ in range(5):
        base_ang_vel_hist.append(initial_base_ang_vel)
        joint_pos_rel_hist.append(initial_joint_pos_rel)
        joint_vel_hist.append(initial_joint_vel)
        actions_hist.append(initial_action_regex.copy())
        project_gravity_hist.append(initial_project_gravity)

    # High-level command sequence buffer
    generator_command = None
    generator_predictions = None
    gen_call_interval = 20
    
    # Generator history buffer
    if args.mode == "full":
        n_obs_steps = generator.n_obs_steps
        buffer_len = (n_obs_steps - 1) * 5 + 1
        
        gen_history = {
            'obj_pos_b': [],
            'obj_ori_b': [],
            'joint_pos': [],
            'project_gravity': [],
            'target_obj_pos_b': [],
            'target_obj_ori_b': [],
            'last_command': [],
        }
        
        # Helper to compute single frame generator observation
        def get_pose_from_vision(pose_dict):
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "depth_camera")
            P_cam_world = data.cam_xpos[cam_id]
            R_cam_world = data.cam_xmat[cam_id].reshape(3, 3)
            
            p_tag_cv = np.array(pose_dict["pos"])
            q_tag_cv = np.array(pose_dict["quat"])
            R_tag_cv = Rotation.from_quat([q_tag_cv[1], q_tag_cv[2], q_tag_cv[3], q_tag_cv[0]]).as_matrix()
            
            # OpenCV (Z forward, Y down) to MuJoCo (Z backward, Y up)
            R_cv2m = np.array([
                [1,  0,  0],
                [0, -1,  0],
                [0,  0, -1]])
            p_tag_m = R_cv2m @ p_tag_cv
            R_box2cvtag = np.array([
                [-1,  0,  0],
                [0, 1,  0],
                [0,  0, -1]])
            R_tag_m = R_cv2m @ R_tag_cv @ R_box2cvtag
            
            P_tag_world = P_cam_world + R_cam_world @ p_tag_m
            R_tag_world = R_cam_world @ R_tag_m
            
            # Temporary debug print
            if getattr(get_pose_from_vision, 'print_counter', 0) % 50 == 0:
                print(f"[Verbose Vision Debug] P_cam_world: {np.round(P_cam_world, 3)}")
                print(f"                       p_tag_cv:    {np.round(p_tag_cv, 3)}")
                print(f"                       P_tag_world: {np.round(P_tag_world, 3)}")
            get_pose_from_vision.print_counter = getattr(get_pose_from_vision, 'print_counter', 0) + 1
            
            return P_tag_world, R_tag_world

        def get_single_frame_gen_obs(last_cmd):
            p_obj = data.xpos[box_body_id]
            p_torso = data.xpos[torso_body_id]
            R_torso = data.xmat[torso_body_id].reshape(3, 3)
            
            q_torso = np.zeros(4)
            mujoco.mju_mat2Quat(q_torso, data.xmat[torso_body_id])
            
            if args.use_vision and sim_node.obj_pose is not None:
                P_tag_world, R_tag_world = get_pose_from_vision(sim_node.obj_pose)
                p_obj = P_tag_world - R_tag_world @ np.array([0, 0, 0.1755])
                R_obj = R_tag_world
                q_obj = np.zeros(4)
                mujoco.mju_mat2Quat(q_obj, R_obj.flatten())
            else:
                q_obj = np.zeros(4)
                mujoco.mju_mat2Quat(q_obj, data.xmat[box_body_id])

            obj_pos_b_val = R_torso.T @ (p_obj - p_torso)
            obj_ori_b_val = quat_mul(quat_inv(q_torso), q_obj)
            
            joint_pos_val = np.array([data.qpos[adr] for adr in joint_qpos_indices]) - q_default
            
            R_pelvis_val = data.xmat[pelvis_body_id].reshape(3, 3)
            project_gravity_val = R_pelvis_val.T @ np.array([0, 0, -1.0])
            
            # Target is the last frame of reference motion
            if args.use_vision and sim_node.tgt_pose is not None:
                P_tgt_world, R_tgt_world = get_pose_from_vision(sim_node.tgt_pose)
                # Tag is at 0.001 Z. Box center should be at Z=0.17
                obj_target_pos_w = P_tgt_world + np.array([0, 0, 0.17 - 0.001])
                r = Rotation.from_matrix(R_tgt_world)
                q_xyzw = r.as_quat()
                obj_target_quat_w = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
            else:
                obj_target_pos_w = ref_data["obj_pos"][-1]
                obj_target_quat_w = ref_data["obj_quat"][-1]
            
            target_obj_pos_b_val = R_torso.T @ (obj_target_pos_w - p_torso)
            target_obj_ori_b_val = quat_mul(quat_inv(q_torso), obj_target_quat_w)
            
            return {
                'obj_pos_b': obj_pos_b_val,
                'obj_ori_b': obj_ori_b_val,
                'joint_pos': joint_pos_val,
                'project_gravity': project_gravity_val,
                'target_obj_pos_b': target_obj_pos_b_val,
                'target_obj_ori_b': target_obj_ori_b_val,
                'last_command': last_cmd,
            }
            
        # Initial fill
        init_obs = get_single_frame_gen_obs(np.zeros(36))
        for _ in range(buffer_len):
            for key in gen_history.keys():
                gen_history[key].append(init_obs[key])

    # Modular step function for simulation control loop
    def run_step(frame_idx, gen_cmd, gen_preds):
        nonlocal gen_history
        
        # 1. Update State Variables
        R_pelvis = data.xmat[pelvis_body_id].reshape(3, 3)
        curr_base_ang_vel = data.qvel[3:6].copy()
        
        q_actual = np.array([data.qpos[adr] for adr in joint_qpos_indices])
        v_actual = np.array([data.qvel[adr] for adr in joint_qvel_indices])
        curr_joint_pos_rel = q_actual - q_default
        curr_joint_vel = v_actual
        curr_project_gravity = R_pelvis.T @ np.array([0, 0, -1.0])
        
        # Object pose relative to Torso Link
        p_obj = data.xpos[box_body_id]
        p_torso = data.xpos[torso_body_id]
        R_torso = data.xmat[torso_body_id].reshape(3, 3)
        R_obj = data.xmat[box_body_id].reshape(3, 3)
        
        if args.use_vision:
            renderer.update_scene(data, camera="depth_camera")
            rgb_image = renderer.render()
            sim_node.publish_image(rgb_image, time.time_ns())
            
            if sim_node.obj_pose is not None:
                P_tag_world, R_tag_world = get_pose_from_vision(sim_node.obj_pose)
                p_obj = P_tag_world - R_tag_world @ np.array([0, 0, 0.1755])
                R_obj = R_tag_world
                
                if frame_idx % 50 == 0:
                    p_true = data.xpos[box_body_id]
                    q_true = Rotation.from_matrix(data.xmat[box_body_id].reshape(3, 3)).as_quat()
                    q_vis = Rotation.from_matrix(R_obj).as_quat()
                    print(f"[Vision Debug] Box Pos True: {np.round(p_true, 3)}, Vis: {np.round(p_obj, 3)}, Error: {np.round(p_true-p_obj, 3)}")
                    print(f"[Vision Debug] Box Quat True: {np.round(q_true, 3)}, Vis: {np.round(q_vis, 3)}, Error: {np.round(q_true-q_vis,3)}")
            
            if sim_node.tgt_pose is not None and frame_idx % 50 == 0:
                P_tgt_world, _ = get_pose_from_vision(sim_node.tgt_pose)
                print(f"[Vision Debug] Target Pos True: [0.3 0.0 0.001], Vis: {np.round(P_tgt_world, 3)}")

        obj_pos_b = R_torso.T @ (p_obj - p_torso)
        
        R_obj_torso = R_torso.T @ R_obj
        obj_ori_b = R_obj_torso[:, :2].flatten() # first 2 columns (6 elements)

        # 2. Update Tracker History queues
        base_ang_vel_hist.pop(0)
        base_ang_vel_hist.append(curr_base_ang_vel)
        
        joint_pos_rel_hist.pop(0)
        joint_pos_rel_hist.append(curr_joint_pos_rel)
        
        joint_vel_hist.pop(0)
        joint_vel_hist.append(curr_joint_vel)
        
        project_gravity_hist.pop(0)
        project_gravity_hist.append(curr_project_gravity)
        
        # Assemble observation arrays (oldest to newest)
        base_ang_vel_history = np.concatenate(base_ang_vel_hist)
        joint_pos_history = np.concatenate(joint_pos_rel_hist)
        joint_vel_history = np.concatenate(joint_vel_hist)
        actions_history = np.concatenate(actions_hist)
        project_gravity = np.concatenate(project_gravity_hist)

        # 3. Retrieve or Predict Command (36 dims)
        if args.mode == "tracker":
            # Tracker mode uses direct values from dataset
            idx = min(frame_idx, ref_data["joint_pos_urdf"].shape[0] - 1)
            ref_joint_pos_frame = ref_data["joint_pos_urdf"][idx]
            
            # Get reference base orientation
            q_ref_base = ref_data["robot_body_quat_w"][idx, 0]
            # Rotate linear and angular velocities to local base frame
            ref_root_lin_vel_b_frame = quat_apply_inverse(q_ref_base, ref_data["robot_body_lin_vel_w"][idx, 0])
            ref_root_ang_vel_b_frame = quat_apply_inverse(q_ref_base, ref_data["robot_body_ang_vel_w"][idx, 0])
            
            ref_contact_label_frame = np.array([ref_data["contact_label"][idx]], dtype=float)
            
            generated_command = np.concatenate([
                ref_joint_pos_frame,
                ref_root_lin_vel_b_frame,
                ref_root_ang_vel_b_frame,
                ref_contact_label_frame
            ])
        else:
            # Full mode uses Generator Wrapper
            last_cmd_36 = gen_cmd if gen_cmd is not None else np.zeros(36)
            
            curr_gen_obs = get_single_frame_gen_obs(last_cmd_36)
            for key in gen_history.keys():
                gen_history[key].pop(0)
                gen_history[key].append(curr_gen_obs[key])
                
            # Call generator every 20 steps
            if frame_idx % gen_call_interval == 0:
                # Prepare GeneratorObs tensor (the wrapper downsamples internally)
                gen_obs_input = GeneratorObs(
                    obj_pos_b=torch.tensor(np.expand_dims(gen_history['obj_pos_b'], axis=0), dtype=torch.float32, device=args.device),
                    obj_ori_b=torch.tensor(np.expand_dims(gen_history['obj_ori_b'], axis=0), dtype=torch.float32, device=args.device),
                    joint_pos=torch.tensor(np.expand_dims(gen_history['joint_pos'], axis=0), dtype=torch.float32, device=args.device),
                    project_gravity=torch.tensor(np.expand_dims(gen_history['project_gravity'], axis=0), dtype=torch.float32, device=args.device),
                    target_obj_pos_b=torch.tensor(np.expand_dims(gen_history['target_obj_pos_b'], axis=0), dtype=torch.float32, device=args.device),
                    target_obj_ori_b=torch.tensor(np.expand_dims(gen_history['target_obj_ori_b'], axis=0), dtype=torch.float32, device=args.device),
                    last_command=torch.tensor(np.expand_dims(gen_history['last_command'], axis=0), dtype=torch.float32, device=args.device)
                )
                
                # Run prediction
                with torch.inference_mode():
                    gen_output = generator.predict(gen_obs_input)
                    gen_preds = gen_output.cpu().numpy()[0] # (96, 36)
                    
            # Retrieve command for current step in interval
            step_in_interval = frame_idx % gen_call_interval
            gen_cmd = gen_preds[step_in_interval]
            generated_command = gen_cmd

        # 4. Assemble Full 510-dim Tracker Observation
        tracker_obs = np.concatenate([
            generated_command,      # 36 dims
            base_ang_vel_history,   # 15 dims
            joint_pos_history,      # 145 dims
            joint_vel_history,      # 145 dims
            actions_history,        # 145 dims
            project_gravity,        # 15 dims
            obj_pos_b,              # 3 dims
            obj_ori_b               # 6 dims
        ])
        
        # 5. Policy Inference to get normalized action [-1, 1]
        action = tracker_policy(tracker_obs)
        action = np.clip(action, -5, 5)
        
        # 6. PD Joint Control & MuJoCo Stepping
        # The network outputs action in REGEX order. We map it back to URDF order for MuJoCo control.
        action_urdf = np.zeros(29)
        for i, name in enumerate(ACTION_JOINT_NAMES):
            urdf_idx = JOINT_NAMES_29.index(name)
            action_urdf[urdf_idx] = action[i]
            
        q_target = q_default + action_urdf * action_scale_array

        if frame_idx == 0:
            import json
            diag_log = []
            for i, name in enumerate(JOINT_NAMES_29):
                # find indices
                alpha_idx = i # since JOINT_NAMES_29 is now URDF order
                act_id = actuator_indices[alpha_idx]
                qpos_adr = joint_qpos_indices[alpha_idx]
                
                # find action out index
                if name in ACTION_JOINT_NAMES:
                    action_out_idx = ACTION_JOINT_NAMES.index(name)
                    raw_action = action[action_out_idx]
                else:
                    raw_action = 0.0
                    
                joint_diag = {
                    "name": name,
                    "q_actual": float(q_actual[alpha_idx]),
                    "q_default": float(q_default[alpha_idx]),
                    "q_target": float(q_target[alpha_idx]),
                    "curr_joint_pos_rel": float(curr_joint_pos_rel[alpha_idx]),
                    "raw_action": float(raw_action),
                    "action_urdf": float(action_urdf[alpha_idx]),
                    "action_scale": float(action_scale_array[alpha_idx]),
                    "kp": float(kp_array[alpha_idx]),
                    "kd": float(kd_array[alpha_idx]),
                    "actuator_id": int(act_id),
                    "qpos_adr": int(qpos_adr),
                }
                diag_log.append(joint_diag)
            
            with open("sim2sim_diagnostics.log", "w") as f:
                f.write("--- FRAME 0 DIAGNOSTICS ---\n")
                f.write(json.dumps(diag_log, indent=2))
                
                f.write("\n\n--- FRAME 0 TRACKER OBSERVATION DIAGNOSTICS ---\n")
                obs_diag = {
                    "tracker_obs_shape": tracker_obs.shape,
                    "components": {
                        "generated_command": generated_command.tolist(),
                        "base_ang_vel_history": base_ang_vel_history.tolist(),
                        "joint_pos_history": joint_pos_history.tolist(),
                        "joint_vel_history": joint_vel_history.tolist(),
                        "actions_history": actions_history.tolist(),
                        "project_gravity": project_gravity.tolist(),
                        "obj_pos_b": obj_pos_b.tolist(),
                        "obj_ori_b": obj_ori_b.tolist()
                    }
                }
                f.write(json.dumps(obs_diag, indent=2))
            
            print("--- FRAME 0 DIAGNOSTICS WRITTEN TO sim2sim_diagnostics.log ---")
            
        # Update applied action history
        actions_hist.pop(0)
        actions_hist.append(action)
        
        # Step physics decimation times
        for _ in range(decimation):
            for i, (qpos_adr, dof_adr, act_id) in enumerate(zip(joint_qpos_indices, joint_qvel_indices, actuator_indices)):
                q_val = data.qpos[qpos_adr]
                dq_val = data.qvel[dof_adr]
                torque = kp_array[i] * (q_target[i] - q_val) - kd_array[i] * dq_val
                data.ctrl[act_id] = torque
                
            mujoco.mj_step(model, data)
            
        return gen_cmd, gen_preds, p_obj

    print("\n" + "="*50)
    print(f"Starting Sim2Sim Loop (Mode: {args.mode.upper()})")
    print("="*50 + "\n")
    
    if args.headless:
        print("[Sim2Sim] Running in HEADLESS mode...")
        frame_idx = 0
        while frame_idx < num_frames:
            step_start = time.time()
            generator_command, generator_predictions, p_obj = run_step(frame_idx, generator_command, generator_predictions)
            
            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"[Sim2Sim] Frame: {frame_idx}/{num_frames} | Pelvis Z: {data.qpos[2]:.3f} | Box Pos: {p_obj}")
                
            elapsed = time.time() - step_start
            if elapsed < args.control_dt:
                time.sleep(args.control_dt - elapsed)
    else:
        print("[Sim2Sim] Launching passive viewer...")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            frame_idx = 0
            while viewer.is_running() and frame_idx < num_frames:
                step_start = time.time()
                generator_command, generator_predictions, p_obj = run_step(frame_idx, generator_command, generator_predictions)
                
                viewer.sync()
                frame_idx += 1
                if frame_idx % 50 == 0:
                    print(f"[Sim2Sim] Frame: {frame_idx}/{num_frames} | Pelvis Z: {data.qpos[2]:.3f} | Box Pos: {p_obj}")
                    
                elapsed = time.time() - step_start
                if elapsed < args.control_dt:
                    time.sleep(args.control_dt - elapsed)

    print("\nSimulation complete!")

if __name__ == "__main__":
    main()
