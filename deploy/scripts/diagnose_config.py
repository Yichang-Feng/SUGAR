#!/usr/bin/env python3
"""
诊断脚本：检查机器人配置参数是否正确
"""

import numpy as np
from source.sugar_rl.sugar_rl.assets.robots.unitree_config import (
    JOINT_NAMES_29,
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

def diagnose():
    print("\n" + "="*100)
    print("UNITREE G1 ROBOT CONFIGURATION DIAGNOSTIC")
    print("="*100)
    
    # 1. 检查关节数量
    print(f"\n[1] 关节数量检查")
    print(f"    关节总数: {len(JOINT_NAMES_29)} (应为29)")
    print(f"    初始位置条目: {len(INIT_JOINT_POS)}")
    print(f"    Action Scale条目: {len(UNITREE_G1_29DOF_MIMIC_ACTION_SCALE)}")
    
    if len(JOINT_NAMES_29) != 29:
        print("    ⚠️  WARNING: 关节数量不对!")
    if len(INIT_JOINT_POS) < 12:
        print("    ⚠️  WARNING: 初始位置条目太少!")
    if len(UNITREE_G1_29DOF_MIMIC_ACTION_SCALE) < 20:
        print("    ⚠️  WARNING: Action Scale条目太少!")
    
    # 2. 检查初始位置
    print(f"\n[2] 初始位置检查")
    print(f"    底座位置 (INIT_POS): {INIT_POS}")
    if INIT_POS != (0.0, 0.0, 0.76):
        print("    ⚠️  WARNING: 初始位置与预期不符!")
    
    # 3. 构建默认关节位置数组
    print(f"\n[3] 默认关节位置")
    default_joint_pos = np.zeros(29)
    for i, name in enumerate(JOINT_NAMES_29):
        if name in INIT_JOINT_POS:
            default_joint_pos[i] = INIT_JOINT_POS[name]
    
    print(f"    非零位置数: {np.count_nonzero(default_joint_pos)}/29")
    print(f"    关键关节位置:")
    for i, name in enumerate(JOINT_NAMES_29):
        if name in INIT_JOINT_POS:
            print(f"      [{i:2d}] {name:30s} = {default_joint_pos[i]:8.4f}")
    
    # 4. 检查Action Scale
    print(f"\n[4] Action Scale 检查")
    action_scale_array = np.array([
        UNITREE_G1_29DOF_MIMIC_ACTION_SCALE.get(name, 0.0) for name in JOINT_NAMES_29
    ], dtype=np.float32)
    
    print(f"    最小值: {action_scale_array.min():.6f}")
    print(f"    最大值: {action_scale_array.max():.6f}")
    print(f"    平均值: {action_scale_array.mean():.6f}")
    print(f"    零值个数: {np.count_nonzero(action_scale_array == 0)}")
    
    if np.any(action_scale_array == 0):
        print("    ⚠️  WARNING: 存在零值 action_scale!")
        for i, name in enumerate(JOINT_NAMES_29):
            if action_scale_array[i] == 0:
                print(f"      [{i:2d}] {name}")
    
    # 5. 检查刚度和阻尼常数
    print(f"\n[5] 刚度常数 (Stiffness)")
    stiffness_values = {
        "STIFFNESS_5020": STIFFNESS_5020,
        "STIFFNESS_7520_14": STIFFNESS_7520_14,
        "STIFFNESS_7520_22": STIFFNESS_7520_22,
        "STIFFNESS_4010": STIFFNESS_4010,
    }
    for name, val in stiffness_values.items():
        print(f"    {name:20s} = {val:10.6f}")
    
    print(f"\n[6] 阻尼常数 (Damping)")
    damping_values = {
        "DAMPING_5020": DAMPING_5020,
        "DAMPING_7520_14": DAMPING_7520_14,
        "DAMPING_7520_22": DAMPING_7520_22,
        "DAMPING_4010": DAMPING_4010,
    }
    for name, val in damping_values.items():
        print(f"    {name:20s} = {val:10.6f}")
    
    # 6. 预期的 action_scale 值
    print(f"\n[7] 预期 Action Scale 值 (根据刚度和阻尼)")
    expected_scales = {
        "左腿 (hip_pitch)": 0.25 * 88.0 / STIFFNESS_7520_14,
        "左腿 (hip_roll)": 0.25 * 139.0 / STIFFNESS_7520_22,
        "腰部 (yaw)": 0.25 * 88.0 / STIFFNESS_7520_14,
        "腿踝 (ankle_pitch)": 0.25 * 50.0 / (2.0 * STIFFNESS_5020),
        "左臂 (shoulder)": 0.25 * 25.0 / STIFFNESS_5020,
        "左腕 (wrist_pitch)": 0.25 * 5.0 / STIFFNESS_4010,
    }
    for name, val in expected_scales.items():
        print(f"    {name:25s} = {val:10.6f}")
    
    # 7. 综合诊断
    print(f"\n[8] 综合诊断结果")
    status = "✓ OK"
    if len(JOINT_NAMES_29) != 29:
        status = "✗ ERROR"
    if np.any(action_scale_array == 0):
        status = "✗ ERROR"
    if default_joint_pos[0] == 0 and default_joint_pos[1] == 0:
        status = "✗ ERROR (缺少初始位置)"
    
    print(f"    状态: {status}")
    
    print("\n" + "="*100 + "\n")

if __name__ == "__main__":
    diagnose()
