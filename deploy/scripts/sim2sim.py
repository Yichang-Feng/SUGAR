import time
import argparse
import numpy as np
import zmq
import msgpack
import msgpack_numpy as mnp
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

from source.sugar_rl.sugar_rl.assets.robots.unitree_config import (
    JOINT_NAMES_29,
    INIT_JOINT_POS,
    INIT_POS,
    STIFFNESS_5020,
    STIFFNESS_7520_14,
    STIFFNESS_7520_22,
    STIFFNESS_4010,
    DAMPING_5020,
    DAMPING_7520_14,
    DAMPING_7520_22,
    DAMPING_4010,
)

mnp.patch()

# Configuration
XML_PATH = "deploy/data/robot_model/model_data/g1/g1_29dof_rev_1_0_box.xml"
SIM_DT = 0.005
CTRL_DT = 0.02
ACTION_PUBLISH_RATE = 50

# ZMQ Ports
STATE_PORT = 5557
OBJECT_PORT = 5558
ACTION_PORT = 5556

def get_body_frame_transform(pos_w, quat_w, base_pos_w, base_quat_w):
    base_rot = R.from_quat(base_quat_w[[1, 2, 3, 0]]) # xyzw
    rot = R.from_quat(quat_w[[1, 2, 3, 0]])
    pos_b = base_rot.inv().apply(pos_w - base_pos_w)
    quat_b = (base_rot.inv() * rot).as_quat()[[3, 0, 1, 2]] # wxyz
    return pos_b, quat_b

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_path", type=str, default=XML_PATH)
    args = parser.parse_args()

    # 1. ZMQ Setup
    ctx = zmq.Context()
    
    state_pub = ctx.socket(zmq.PUB)
    state_pub.bind(f"tcp://*:{STATE_PORT}")
    
    obj_pub = ctx.socket(zmq.PUB)
    obj_pub.bind(f"tcp://*:{OBJECT_PORT}")
    
    action_sub = ctx.socket(zmq.SUB)
    action_sub.connect(f"tcp://localhost:{ACTION_PORT}")
    action_sub.setsockopt_string(zmq.SUBSCRIBE, "pose")
    action_sub.setsockopt(zmq.CONFLATE, 1)
    action_sub.setsockopt(zmq.RCVTIMEO, 0)
    
    # 2. MuJoCo Setup
    model = mujoco.MjModel.from_xml_path(args.xml_path)
    data = mujoco.MjData(model)
    
    # Setup IDs
    actuator_ids = []
    qpos_adrs = []
    qvel_adrs = []
    for name in JOINT_NAMES_29:
        motor_name = name  # XML motor names are the same as joint names for this model
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, motor_name)
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if aid == -1:
            raise ValueError(f"Actuator not found for motor '{motor_name}'. "
                             "Check XML motor names and JOINT_NAMES_29 mapping.")
        if jid == -1:
            raise ValueError(f"Joint not found for '{name}'. "
                             "Check XML joint names and JOINT_NAMES_29 mapping.")
        actuator_ids.append(aid)
        qpos_adrs.append(model.jnt_qposadr[jid])
        qvel_adrs.append(model.jnt_dofadr[jid])
        
    pelvis_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    pelvis_jnt_id = model.body_jntadr[pelvis_body_id]
    pelvis_qpos_adr = model.jnt_qposadr[pelvis_jnt_id]
    pelvis_qvel_adr = model.jnt_dofadr[pelvis_jnt_id]
    torso_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    
    obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if obj_body_id != -1:
        obj_jnt_id = model.body_jntadr[obj_body_id]
        obj_qpos_adr = model.jnt_qposadr[obj_jnt_id]
    
    target_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target_point")
    
    # 从配置获取初始位置
    data.qpos[pelvis_qpos_adr : pelvis_qpos_adr+3] = INIT_POS
    data.qpos[pelvis_qpos_adr+3 : pelvis_qpos_adr+7] = [1.0, 0.0, 0.0, 0.0]    # 初始四元数(w,x,y,z)
    
    # Set init state
    for name, val in INIT_JOINT_POS.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid != -1:
            data.qpos[model.jnt_qposadr[jid]] = val

    # 打印初始化相关信息以便调试
    print("[Sim2Sim] 写入初始关节位置完成，打印样例：")
    pelvis_pos = data.qpos[pelvis_qpos_adr : pelvis_qpos_adr+3]
    pelvis_quat = data.qpos[pelvis_qpos_adr+3 : pelvis_qpos_adr+7]
    print(f"  pelvis pos: {pelvis_pos}, quat: {pelvis_quat}")
    # 打印前6个关节的初始 qpos（或 None）
    sample_q = []
    for i in range(min(6, len(JOINT_NAMES_29))):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAMES_29[i])
        if jid != -1:
            sample_q.append(float(data.qpos[model.jnt_qposadr[jid]]))
        else:
            sample_q.append(None)
    print(f"  sample joint qpos (first 6 joints): {sample_q}")

    # 打印 actuator id 与 joint name 对应信息
    print("  Actuator mapping (motor_name -> actuator_id):")
    for i, name in enumerate(JOINT_NAMES_29):
        motor_name = name
        aid = actuator_ids[i] if i < len(actuator_ids) else -1
        print(f"    {motor_name:30s} -> actuator_id={aid}")

    mujoco.mj_forward(model, data)

    target_q = np.array([data.qpos[adr] for adr in qpos_adrs])
    print("[Sim2Sim] Initial joint target positions (29):")
    print(np.array2string(target_q, precision=3, separator=', ', max_line_width=200))
    print("[Sim2Sim] Initial qpos addresses (first 10 shown):")
    print(qpos_adrs[:10])
    
    # PD Gains (Tuned for G1)
    kp = np.array([
        STIFFNESS_7520_14, STIFFNESS_7520_22, STIFFNESS_7520_14, STIFFNESS_7520_22, 2.0 * STIFFNESS_5020, 2.0 * STIFFNESS_5020,  # 左腿
        STIFFNESS_7520_14, STIFFNESS_7520_22, STIFFNESS_7520_14, STIFFNESS_7520_22, 2.0 * STIFFNESS_5020, 2.0 * STIFFNESS_5020,  # 右腿
        STIFFNESS_7520_14, 2.0 * STIFFNESS_5020, 2.0 * STIFFNESS_5020,                                                           # 腰部
        STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_4010, STIFFNESS_4010,           # 左臂
        STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_4010, STIFFNESS_4010            # 右臂
    ])

    kd = np.array([
        DAMPING_7520_14, DAMPING_7520_22, DAMPING_7520_14, DAMPING_7520_22, 2.0 * DAMPING_5020, 2.0 * DAMPING_5020,  # 左腿
        DAMPING_7520_14, DAMPING_7520_22, DAMPING_7520_14, DAMPING_7520_22, 2.0 * DAMPING_5020, 2.0 * DAMPING_5020,  # 右腿
        DAMPING_7520_14, 2.0 * DAMPING_5020, 2.0 * DAMPING_5020,                                                      # 腰部
        DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_4010, DAMPING_4010,             # 左臂
        DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_4010, DAMPING_4010              # 右臂
    ])
    
    print(f"[Sim2Sim] Started MuJoCo loop. Sim DT: {SIM_DT}s")
    
    last_pub_time = 0.0
    pub_interval = 1.0 / ACTION_PUBLISH_RATE
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            # --- ZMQ Receive Action ---
            try:
                raw = action_sub.recv()
                payload = msgpack.unpackb(raw[len("pose"):], raw=False)
                # Ensure it's numpy
                target_q = np.array(payload["joint_pos"]).flatten()
            except zmq.Again:
                pass
            
            # --- Apply PD Control ---
            for i in range(29):
                q = data.qpos[qpos_adrs[i]]
                dq = data.qvel[qvel_adrs[i]]
                torque = kp[i] * (target_q[i] - q) + kd[i] * (0.0 - dq)
                torque = np.clip(torque, -100.0, 100.0)
                data.ctrl[actuator_ids[i]] = torque
                
            # --- Step Physics ---
            mujoco.mj_step(model, data)
            
            # --- Publish States ---
            current_time = data.time
            if current_time - last_pub_time >= pub_interval:
                last_pub_time = current_time
                
                # Proprio
                base_pos = data.qpos[pelvis_qpos_adr : pelvis_qpos_adr+3]
                base_quat = data.qpos[pelvis_qpos_adr+3 : pelvis_qpos_adr+7] # w, x, y, z
                base_ang_vel = data.qvel[pelvis_qvel_adr+3 : pelvis_qvel_adr+6]
                
                body_q = np.array([data.qpos[adr] for adr in qpos_adrs])
                body_dq = np.array([data.qvel[adr] for adr in qvel_adrs])
                
                proprio_msg = {
                    "base_quat": base_quat,
                    "base_ang_vel": base_ang_vel,
                    "body_q": body_q,
                    "body_dq": body_dq
                }
                
                state_pub.send(b"g1_debug" + msgpack.packb(proprio_msg))
                
                # Object
                if obj_body_id != -1:
                    torso_pos = data.xpos[torso_body_id]
                    torso_quat = data.xquat[torso_body_id]
                    
                    obj_pos_w = data.qpos[obj_qpos_adr : obj_qpos_adr+3]
                    obj_quat_w = data.qpos[obj_qpos_adr+3 : obj_qpos_adr+7]
                    
                    obj_pos_b, obj_quat_b = get_body_frame_transform(obj_pos_w, obj_quat_w, torso_pos, torso_quat)
                    
                    target_point_w = data.site_xpos[target_site_id]
                    target_point_b, _ = get_body_frame_transform(target_point_w, np.array([1, 0, 0, 0]), torso_pos, torso_quat)
                    
                    obj_msg = {
                        "obj_pos_b": obj_pos_b,
                        "obj_ori_b": obj_quat_b,
                        "target_point": target_point_b
                    }
                    obj_pub.send(b"object_state" + msgpack.packb(obj_msg))
            
            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
