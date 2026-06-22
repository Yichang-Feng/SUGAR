"""
Unitree G1 Robot Configuration Constants
纯 Python 配置文件，不依赖 isaaclab，可直接在推理脚本中使用
"""

# === 电机参数 ===
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

# === 刚度常数 ===
STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2  # 14.25062309787429
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2  # 40.17923847137318
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2  # 99.09842777666113
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2  # 16.77832748089279

# === 阻尼常数 ===
DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ  # 0.907222843292423
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ  # 2.5578897650279457
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ  # 6.3088018534966395
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ  # 1.06814150219

# === 关节名称 (29 DOF) ===
JOINT_NAMES_29 = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# === 初始关节位置 ===
INIT_JOINT_POS = {
    "left_hip_pitch_joint": -0.312,
    "left_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "right_hip_pitch_joint": -0.312,
    "right_knee_joint": 0.669,
    "right_ankle_pitch_joint": -0.363,
    "left_elbow_joint": 0.6,
    "right_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}

# === 初始位置 ===
INIT_POS = (0.0, 0.0, 0.76)

# === 动作缩放系数 ===
# 计算逻辑: action_scale = 0.25 * effort_limit_sim / stiffness
# 基于 UNITREE_G1_29DOF_MIMIC_CFG 中的配置计算
UNITREE_G1_29DOF_MIMIC_ACTION_SCALE = {
    # 左腿
    "left_hip_pitch_joint": 0.25 * 88.0 / STIFFNESS_7520_14,           # 0.5475
    "left_hip_roll_joint": 0.25 * 139.0 / STIFFNESS_7520_22,          # 0.3507
    "left_hip_yaw_joint": 0.25 * 88.0 / STIFFNESS_7520_14,            # 0.5475
    "left_knee_joint": 0.25 * 139.0 / STIFFNESS_7520_22,              # 0.3507
    "left_ankle_pitch_joint": 0.25 * 50.0 / (2.0 * STIFFNESS_5020),   # 0.4386
    "left_ankle_roll_joint": 0.25 * 50.0 / (2.0 * STIFFNESS_5020),    # 0.4386
    
    # 右腿
    "right_hip_pitch_joint": 0.25 * 88.0 / STIFFNESS_7520_14,         # 0.5475
    "right_hip_roll_joint": 0.25 * 139.0 / STIFFNESS_7520_22,         # 0.3507
    "right_hip_yaw_joint": 0.25 * 88.0 / STIFFNESS_7520_14,           # 0.5475
    "right_knee_joint": 0.25 * 139.0 / STIFFNESS_7520_22,             # 0.3507
    "right_ankle_pitch_joint": 0.25 * 50.0 / (2.0 * STIFFNESS_5020),  # 0.4386
    "right_ankle_roll_joint": 0.25 * 50.0 / (2.0 * STIFFNESS_5020),   # 0.4386
    
    # 腰部
    "waist_yaw_joint": 0.25 * 88.0 / STIFFNESS_7520_14,               # 0.5475
    "waist_roll_joint": 0.25 * 50.0 / (2.0 * STIFFNESS_5020),         # 0.4386
    "waist_pitch_joint": 0.25 * 50.0 / (2.0 * STIFFNESS_5020),        # 0.4386
    
    # 左臂
    "left_shoulder_pitch_joint": 0.25 * 25.0 / STIFFNESS_5020,        # 0.4386
    "left_shoulder_roll_joint": 0.25 * 25.0 / STIFFNESS_5020,         # 0.4386
    "left_shoulder_yaw_joint": 0.25 * 25.0 / STIFFNESS_5020,          # 0.4386
    "left_elbow_joint": 0.25 * 25.0 / STIFFNESS_5020,                 # 0.4386
    "left_wrist_roll_joint": 0.25 * 25.0 / STIFFNESS_5020,            # 0.4386
    "left_wrist_pitch_joint": 0.25 * 5.0 / STIFFNESS_4010,            # 0.0745
    "left_wrist_yaw_joint": 0.25 * 5.0 / STIFFNESS_4010,              # 0.0745
    
    # 右臂
    "right_shoulder_pitch_joint": 0.25 * 25.0 / STIFFNESS_5020,       # 0.4386
    "right_shoulder_roll_joint": 0.25 * 25.0 / STIFFNESS_5020,        # 0.4386
    "right_shoulder_yaw_joint": 0.25 * 25.0 / STIFFNESS_5020,         # 0.4386
    "right_elbow_joint": 0.25 * 25.0 / STIFFNESS_5020,                # 0.4386
    "right_wrist_roll_joint": 0.25 * 25.0 / STIFFNESS_5020,           # 0.4386
    "right_wrist_pitch_joint": 0.25 * 5.0 / STIFFNESS_4010,           # 0.0745
    "right_wrist_yaw_joint": 0.25 * 5.0 / STIFFNESS_4010,             # 0.0745
}
