# 机器人倒下问题排查指南

## 症状
机器人快速扭曲并倒下

## 可能原因及排查步骤

### 第一步：诊断配置参数

```bash
cd /home/feng/SUGAR
python deploy/scripts/diagnose_config.py
```

这个脚本会检查：
- ✓ 关节数量是否正确 (29)
- ✓ 初始位置是否设置
- ✓ Action Scale 是否有零值
- ✓ 刚度和阻尼常数是否合理

**常见问题：**
- Action Scale 中有零值 → 某些关节没有获取scale
- 初始位置全零 → 机器人会倒下
- 刚度/阻尼为负数 → 配置错误

---

### 第二步：测试纯控制器（零动作）

```bash
# 终端1: 运行模拟器
python deploy/scripts/sim2sim.py

# 终端2: 运行推理（零动作模式）
python deploy/scripts/run_inference.py --debug_mode zero_action
```

**预期结果：**
- 机器人保持初始站立姿态，不应该倒下
- 即使有轻微抖动也正常（PD控制器响应）

**如果还是倒下：**
- ⚠️ 问题在 sim2sim.py 的 kp/kd 增益或初始位置
- 检查 `sim2sim.py` 中的 PD Gains
- 检查 `unitree_config.py` 中的 INIT_POS

---

### 第三步：检查观测构建

```bash
# 打印观测和动作的详细信息
python deploy/scripts/run_inference.py --debug_mode debug_obs
```

**输出示例（每50帧）：**
```
[Frame 50] 高层观测:
  high_obs shape: (1, 84)
  high_level_cmd: [0.123 -0.456 ...]... (前5个值)
  raw_low_action: [0.001 -0.002 ...]... (前5个值)
  action_scale: [0.5475 0.3507 ...]... (前5个值)
  target_joint_pos: [-0.312 0.669 ...]... (前5个值)
```

**检查清单：**
- ✓ high_obs 维度是否合理 (应该 > 50)
- ✓ high_level_cmd 是否全零或有极端值
- ✓ raw_low_action 是否过大 (> 1.0 或 < -1.0)
- ✓ action_scale 是否都为正数
- ✓ target_joint_pos 是否接近初始位置

---

### 第四步：检查 Action Scale 映射

```bash
# 打印所有关节的 action_scale 值
python deploy/scripts/run_inference.py --debug_mode debug_scales
```

**输出示例：**
```
ACTION SCALE DEBUG INFO
[00] left_hip_pitch_joint            scale=0.547500 default_pos=-0.312000
[01] left_hip_roll_joint             scale=0.350700 default_pos=0.000000
...
```

**检查项：**
- ✓ 所有scale都非零
- ✓ scale值在 0.01 ~ 1.0 范围内
- ✓ default_pos 是否正确（缺腿关节应有值）

---

## 关键参数说明

### 1. Action Scale (`unitree_config.py`)
```python
# 计算公式: action_scale = 0.25 * effort_limit / stiffness
# 例: left_hip_pitch = 0.25 * 88.0 / 40.179 ≈ 0.5475
```

这决定了神经网络输出的归一化幅度。如果为零：
- 该关节不会响应任何动作
- 机器人会快速倒下

### 2. 刚度和阻尼 (kp/kd in sim2sim.py)
```python
# 这些来自 unitree_config.py 的常数
kp = [STIFFNESS_7520_14, STIFFNESS_7520_22, ...]  # 位置环增益
kd = [DAMPING_7520_14, DAMPING_7520_22, ...]      # 速度环增益
```

如果太小 → 机器人软弱无力，容易倒下
如果太大 → 机器人震荡，也会倒下

### 3. 初始位置 (INIT_JOINT_POS in unitree_config.py)
```python
{
    "left_hip_pitch_joint": -0.312,   # 必须有值
    "left_knee_joint": 0.669,          # 必须有值
    "right_hip_pitch_joint": -0.312,   # 必须有值
    "right_knee_joint": 0.669,         # 必须有值
    # ... 其他关节
}
```

缺少腿部关节初始位置 → 机器人倒下

---

## 快速修复清单

- [ ] 运行 `diagnose_config.py`，检查是否有红色 ⚠️ 警告
- [ ] 测试 `--debug_mode zero_action`，确认基础控制器工作
- [ ] 检查 `unitree_config.py` 中的初始位置是否完整
- [ ] 检查 `sim2sim.py` 中的 kp/kd 值是否从 unitree_config 正确导入
- [ ] 运行 `--debug_mode debug_obs` 观察高层策略输出是否异常

---

## 重要提示

1. **先修复物理模拟（sim2sim），再测试策略推理**
   - 如果零动作都倒下 → 问题在 sim2sim.py
   - 如果零动作正常 → 问题在高层策略或观测构建

2. **逐步复杂化测试**
   - 第一步：零动作 + 默认位置
   - 第二步：低阶策略输出
   - 第三步：完整分层策略

3. **检查 git diff**
   ```bash
   git diff source/sugar_rl/sugar_rl/assets/robots/unitree_config.py
   # 看看是否意外修改了配置
   ```
