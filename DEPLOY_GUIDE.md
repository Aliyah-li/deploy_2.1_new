# M20 SDK Deploy 项目运行与对接指南

> 本文档详细说明 M20 轮足机器人强化学习部署项目的架构、运行流程、
> 与训练项目（suspension_mixed）的对接方式，以及如何适配到自己的机器人项目。

---

## 目录

1. [项目架构总览](#1-项目架构总览)
2. [快速开始](#2-快速开始)
3. [交互流程详解](#3-交互流程详解)
4. [与 suspension_mixed（轮足训练项目）的对接](#4-与-suspension_mixed轮足训练项目的对接)
5. [速度和命令实时性说明](#5-速度和命令实时性说明)
6. [如何适配到自己的项目](#6-如何适配到自己的项目)
7. [常见问题与调试技巧](#7-常见问题与调试技巧)
8. [地形修改指南](#8-地形修改指南)
9. [附录：关键文件清单](#9-附录关键文件清单)

---

## 1. 项目架构总览

### 1.1 顶层架构

```
┌──────────────────────┐      /JOINTS_CMD       ┌──────────────────────┐
│    rl_deploy (C++)   │ ──────────────────────▶ │  MuJoCo Simulation   │
│  策略推理 + 状态机     │                          │  (Python ROS2 Node)  │
│                      │ ◀────────────────────── │                      │
│                      │    /IMU_DATA + /JOINTS_DATA                    │
└──────────────────────┘                          └──────────────────────┘
         ▲                                                    │
         │ 键盘输入                                           │ 物理引擎
         ▼                                                    ▼
    KeyboardInterface                                    MuJoCo (mj_step)
```

### 1.2 ROS2 通信拓扑

| Topic | 类型 | 发布者 | 订阅者 | 频率 |
|---|---|---|---|---|
| `/JOINTS_CMD` | `JointsDataCmd` | `rl_deploy` | MuJoCo仿真 | 200-1000 Hz |
| `/JOINTS_DATA` | `JointsData` | MuJoCo仿真 | `rl_deploy` | 200 Hz |
| `/IMU_DATA` | `ImuData` | MuJoCo仿真 | `rl_deploy` | 200 Hz |
| `/BATTERY_DATA` | `BatteryData` | MuJoCo仿真 | `rl_deploy` | 低频 |

### 1.3 核心文件分布

```
Deploy_active_suspension_mixed/
├── build.sh                                    # 编译脚本
├── run_m20_simulation.sh                       # 一键运行（双终端）
├── run_m20_terminal.sh                         # 独立终端运行
├── run_on_robot.sh                             # 真机部署脚本
├── config.txt                                  # 策略配置说明
├── deploy.md                                   # 部署备忘
├── monitor_ui.py                               # 终端监控UI
│
├── src/M20_sdk_deploy/
│   ├── main.cpp                                # 入口：创建状态机
│   │
│   ├── state_machine/
│   │   ├── state_machine_base.h                # 状态机基类 + 定时器
│   │   ├── state_base.h                        # 状态基类
│   │   └── quadruped_wheel/
│   │       ├── qw_state_machine.hpp            # 状态机实现（创建所有状态）
│   │       ├── idle_state.hpp                  # 待机状态
│   │       ├── standup_state.hpp               # 站立状态
│   │       ├── rl_control_state.hpp            # RL控制状态（核心）
│   │       └── joint_damping_state.hpp         # 关节阻尼状态
│   │
│   ├── run_policy/
│   │   ├── policy_runner_base.hpp              # 策略运行器基类
│   │   └── m20_policy_runner.hpp               # M20策略推理（核心）
│   │
│   ├── interface/
│   │   ├── robot/
│   │   │   ├── robot_interface.h               # 机器人硬件接口基类
│   │   │   ├── hardware/
│   │   │   │   ├── dds_interface.hpp           # ROS2 DDS 通信层
│   │   │   │   └── m20_interface.hpp           # M20 硬件接口
│   │   │   └── simulation/
│   │   │       └── mujoco_simulation_ros2.py   # MuJoCo 仿真器
│   │   └── user_command/
│   │       ├── user_command_interface.h        # 用户命令接口基类
│   │       ├── keyboard_interface.hpp          # 全向键盘控制
│   │       ├── keyboard_interface_sim.hpp      # 仿真键盘控制
│   │       └── fixed_direction_keyboard_interface.hpp  # 固定方向键盘控制
│   │
│   ├── policy/
│   │   ├── low_speed/policy.onnx               # 低速策略
│   │   └── high_speed/                         # 高速策略集
│   │       ├── x_rash/policy.onnx
│   │       ├── y_jump/policy.onnx
│   │       └── diagonal/policy.onnx
│   │
│   ├── config/
│   │   ├── teleop_config.json                  # 遥操作参数配置
│   │   └── mixed_terrains_history_vae_config.json  # 网络结构参考
│   │
│   └── third_party/
│       ├── onnxruntime/                        # ONNX Runtime 引擎
│       └── eigen/                              # Eigen 数学库
```

---

## 2. 快速开始

### 2.1 环境要求

- Ubuntu 22.04 + ROS2 Humble
- Python 包：`numpy`, `mujoco`, `scipy`
- C++ 编译：支持 C++17 的编译器

### 2.2 编译

```bash
cd /home/john/Downloads/Deploy_active_suspension_mixed
source /opt/ros/humble/setup.bash
./build.sh x86
```

编译产物：`install/m20_sdk_deploy/lib/m20_sdk_deploy/rl_deploy`

> **Note**: 如果环境中缺少 ROS2，`build.sh` 会自动进入独立编译模式，生成 `build/standalone/rl_deploy`。

### 2.3 运行

**方式一：一键运行（推荐）**

```bash
./run_m20_simulation.sh
```

自动打开两个 gnome-terminal：
- **终端1** — `rl_deploy`（策略推理）
- **终端2** — MuJoCo 仿真（带3D可视化窗口）

**方式二：手动双终端**

```bash
# 终端1
source install/setup.bash
export ROS_DOMAIN_ID=1
ros2 run m20_sdk_deploy rl_deploy

# 终端2（另一个终端）
source install/setup.bash
export ROS_DOMAIN_ID=1
python3 src/M20_sdk_deploy/interface/robot/simulation/mujoco_simulation_ros2.py
```

**方式三：无图形界面**

```bash
M20_USE_VIEWER=0 ./run_m20_simulation.sh
```

### 2.4 键盘控制

#### 全向模式（MODE=1，默认）

| 按键 | 功能 | 备注 |
|---|---|---|
| `z` | 站立 | 从 WaitingForStand 切换到 StandingUp |
| `c` | 进入RL控制 | 从 StandingUp 切换到 RLControlMode |
| `w` / `s` | 前进 / 后退 | 按住持续加速 |
| `a` / `d` | 左移 / 右移 | 可与 w/s 同时按 |
| `q` / `e` | 逆时针 / 顺时针旋转 | |
| `r` | 回到阻尼模式 | 机器人的急停 |
| `t` | 切换高低速 | low ↔ high |

支持**多键同时按**：`w + a` 同时按下产生向左前方45°的运动指令。

#### 固定方向模式（MODE=2）

| 按键 | 功能 |
|---|---|
| `z` | 站立 |
| `c` | 进入RL控制 |
| `1` / `2` / `3` | 选择方向（x / y / 对角） |
| `Q` 按住 | 在选定方向上持续前进（松手即停） |
| `s` | 混合模式高速（仅在 Mixed 模式下有效） |
| `t` | 循环切换 low / high / mixed 速度模式 |
| `r` | 阻尼模式 |

> 在 `qw_state_machine.hpp:43` 中通过 `int MODE` 切换两种模式。

---

## 3. 交互流程详解

### 3.1 状态机流转

```
                     ┌──────────┐
       启动          │  Idle    │
       ────────▶    │ (待机)   │
                     └────┬─────┘
                          │ 按 z
                          ▼
                     ┌──────────┐
                     │ StandUp  │  关节PD控制站立
                     │ (站立)   │  ── 发布 /JOINTS_CMD
                     └────┬─────┘
                          │ 按 c
                          ▼
                     ┌──────────────┐
                     │ RLControl    │  加载4个ONNX策略
                     │ (RL控制)     │  ── 启动策略推理线程
                     │              │  ── 订阅 /IMU_DATA + /JOINTS_DATA
                     │              │  ── 组装观测 → ONNX推理 → 发关节命令
                     └──────┬───────┘
                            │
              ┌─────────────┼─────────────┐
              │ 按 w/a/s    │ 按 t        │ 按 r
              ▼             ▼             ▼
         cmd 实时变化    高低速切换    回到阻尼
         策略实时响应    换策略文件     JointDamping
```

### 3.2 数据流时序

```
时间轴 (1 kHz 控制循环)
│
├─ [0.000s]  MuJoCo mj_step()
│            接收 /JOINTS_CMD  → PD控制 → 物理更新
│
├─ [0.001s]  再次 mj_step()
│
├─ [0.002s]  再次 mj_step()
│
├─ [0.003s]  再次 mj_step()
│
├─ [0.004s]  再次 mj_step()
│
├─ [0.005s]  发布 /IMU_DATA + /JOINTS_DATA (200 Hz)
│            │
│            rl_deploy 接收回调：
│            ├─ 刷新关节状态、IMU数据
│            ├─ Run() 更新观测
│            └─ 每 4 步推理一次（decimation=4 → 50 Hz）
│               └─ getRobotAction() → ONNX推理 → 发 /JOINTS_CMD
│
└─ 循环...
```

### 3.3 状态机主循环（C++侧）

```cpp
// state_machine_base.h:109-139  —  每次定时器中断执行一次
void Run() {
    while (rclcpp::ok()) {
        if (set_timer.time_interrupt()) {        // 1ms 定时器
            ri_ptr_->RefreshRobotData();          // 刷新传感器数据
            rclcpp::spin_some(ri_ptr_->get_node());// 处理 ROS 回调
            
            current_controller_->Run();           // 当前状态的 Run()
            
            // 判断是否要切换状态
            next_state_name_ = current_controller_->GetNextStateName();
            if (next_state_name_ != current_state_name_) {
                current_controller_->OnExit();
                current_controller_ = GetStateControllerPtr(next_state_name_);
                current_controller_->OnEnter();
                // idle → standup → rl_control
            }
        }
    }
}
```

---

## 4. 与 suspension_mixed（轮足训练项目）的对接

这是典型的 **Sim-to-Sim** 流程：训练项目产出 ONNX 策略文件，部署项目加载并推理。

### 4.1 整体流程

```
┌─────────────────────────────────┐     ┌─────────────────────────────────┐
│  训练阶段 (suspension_mixed)     │     │  部署阶段 (Deploy_active_suspension_mixed) │
│                                 │     │                                 │
│  IsaacGym/IsaacSim 训练         │     │  ROS2 + MuJoCo 仿真部署          │
│  ├─ 定义观测空间                 │     │  ├─ 加载 ONNX 策略               │
│  ├─ 定义动作空间                 │     │  ├─ 构建观测向量                 │
│  ├─ 训练 RL 策略                │     │  ├─ ONNX Runtime 推理           │
│  └─ 导出为 ONNX                 │     │  └─ 解析动作输出 → 关节控制      │
│       policy.onnx               │     │                                 │
└──────────────┬──────────────────┘     └──────────────┬──────────────────┘
               │                                       │
               └────────── ONNX 模型文件 ──────────────┘
```

### 4.2 观测空间说明

#### 历史VAE策略（当前使用的架构）

**`obs` 输入** — 58 维：

| 字段 | 维度 | 缩放 |
|---|---|---|
| `base_ang_vel` (body角速度) | 3 | × 0.25 |
| `projected_gravity` (投影重力) | 3 | — |
| `current_yaw` (当前偏航角) | 1 | — |
| `command` (速度指令) | 3 | forward / side / yaw |
| `joint_pos_rel` (相对关节位置) | 16 | 减去 `dof_default` |
| `joint_vel × 0.05` (关节速度) | 16 | × 0.05 |
| `last_action` (上一步动作) | 16 | — |

**`obs_history` 输入** — 610 维：

- 10 帧历史，每帧 61 维
- 每帧包含：`base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + yaw(1) + command(3) + joint_pos(16) + joint_vel(16) + last_action(16)`

> **important**: `base_lin_vel` 在当前部署接口中不可用，被填充为零。见 `m20_policy_runner.hpp:258-260`。

#### Legacy 策略（57 维，无 history）

| 字段 | 维度 | 缩放 |
|---|---|---|
| `base_ang_vel` | 3 | × 0.25 |
| `projected_gravity` | 3 | — |
| `command` | 3 | — |
| `joint_pos_rel` | 16 | 减去 `dof_default` |
| `joint_vel × 0.05` | 16 | × 0.05 |
| `last_action` | 16 | — |

### 4.3 动作空间

**输出** — 16 维向量：

```
action[0..2]     → 左前腿 hipx, hipy, knee  位置目标
action[3]        → 左前轮 wheel              速度目标
action[4..6]     → 右前腿 hipx, hipy, knee  位置目标
action[7]        → 右前轮 wheel              速度目标
action[8..10]    → 左后腿 hipx, hipy, knee  位置目标
action[11]       → 左后轮 wheel              速度目标
action[12..14]   → 右后腿 hipx, hipy, knee  位置目标
action[15]       → 右后轮 wheel              速度目标
```

**动作后处理**：

```cpp
// m20_policy_runner.hpp:363-381
// 1. 按 policy→robot 关节顺序重映射
// 2. 乘以动作缩放因子 action_scale_robot
// 3. 加上默认关节位置 dof_default_eigen_robot
// 4. 每腿前3个关节设为位置控制，第4个(轮子)设为速度控制
// 5. 当所有命令为零时，轮子速度强制置零
```

### 4.4 关节顺序映射

```cpp
// 机器人关节顺序（16个）
robot_order = {
    "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint", "fl_wheel_joint",  // 左前
    "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint", "fr_wheel_joint",  // 右前
    "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint", "hl_wheel_joint",  // 左后
    "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint", "hr_wheel_joint"   // 右后
};

// 策略关节顺序 — 注意轮子在最后
policy_order = {
    "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint",
    "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint",
    "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint",
    "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint",
    "fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint"
};
```

> 你的训练项目输出策略的关节顺序必须和 `policy_order` 一致，否则需要修改这里。

### 4.5 策略文件加载

在 `rl_control_state.hpp:111-114` 中定义：

```cpp
auto low_path       = policy_base / "low_speed"  / "policy.onnx";
auto high_x_path    = policy_base / "high_speed" / "x_rash"   / "policy.onnx";
auto high_y_path    = policy_base / "high_speed" / "y_jump"   / "policy.onnx";
auto high_diag_path = policy_base / "high_speed" / "diagonal" / "policy.onnx";
```

策略选择逻辑（`rl_control_state.hpp:163-183`）：

```cpp
if (speed_mode >= 0.5f) {        // 高速模式
    if (有前进 && 有侧移)  → diagonal 策略
    else if (有侧移)      → y_jump 策略
    else                 → x_rash 策略（默认）
} else {                         // 低速模式
    → low_speed 策略
}
```

### 4.6 网络结构参考

`config/mixed_terrains_history_vae_config.json` 中记录了训练时的网络结构：

| 组件 | 配置 |
|---|---|
| Actor MLP | `[512, 256, 128]` |
| 激活函数 | `ELU` |
| VAE latent dim | 8 |
| VAE 编码器 | `[128, 64]` |
| VAE 解码器 | `[128, 64]` |
| Critic（训练时） | `[512, 256, 128]` |

---

## 5. 速度和命令实时性说明

### 5.1 命令更新是实时的吗？

**是的，完全实时。** 机制如下：

1. **独立键盘线程** — `keyboard_loop()` 在独立线程中运行，每 1ms 读取一次键盘输入
2. **瞬时更新 cmd** — 每次循环直接修改 `usr_cmd_→forward_vel_scale` 等字段
3. **策略线程读取** — `PolicyRunner()` 在另一个独立线程中，每 4ms 读取一次最新命令
4. **多键同时支持** — 维护一个 `held_keys_` 集合，w+a 同时按产生斜向运动

所以当你按下 `w` 时，下一个策略推理周期就能读到变化。

### 5.2 固定方向模式的加速曲线

`FixedDirectionKeyboardInterface` 有更精细的命令控制：

```
按 Q → [加速阶段] → [匀速阶段] → [自动停止]
        加速率 1.5-3.5 m/s²    持续 300-1250ms
```

- 加速阶段：当前速度逐渐增加到最大值
- 匀速阶段：保持最大速度
- 到达预定时长后自动停止
- 松开 Q 键立即停止

参数在 `teleop_config.json` 中配置：

```json
{
    "low_speed": {
        "max_velocity": 1.25,
        "acceleration": 1.25,
        "duration_x": 0.9,
        "duration_y": 0.8,
        "duration_diagonal": 1.25
    },
    "high_speed": {
        "max_velocity": 3.5,
        "acceleration": 4.5,
        "duration_x": 0.5,
        "duration_y": 0.3,
        "duration_diagonal": 0.65
    }
}
```

### 5.3 通信时延

| 阶段 | 延迟 | 说明 |
|---|---|---|
| 键盘输入→cmd | <1ms | 同一进程，直接写内存 |
| cmd→观测构建 | ~2ms | 策略线程每4ms读取一次 |
| ONNX推理 | ~0.1-1ms | 取决于模型和平台 |
| ROS2消息传输 | ~0.1-1ms | 本地回环，极低延迟 |
| MuJoCo执行 | 1ms | 控制循环周期 |

**端到端延迟：约 2-5ms**。

---

## 6. 如何适配到自己的项目

### 6.1 场景：更换机器人型号（同框架——轮足机器人）

| 修改项 | 文件 | 说明 |
|---|---|---|
| **关节数量** | `m20_policy_runner.hpp:31` | `motor_num = 16` → 你的电机数 |
| **关节名称** | `m20_policy_runner.hpp:56-69` | `robot_order` / `policy_order` |
| **默认姿态** | `m20_policy_runner.hpp:106-113` | `dof_default_eigen_policy` / `dof_default_eigen_robot` |
| **动作缩放** | `m20_policy_runner.hpp:72-75` | `action_scale_robot` 每个关节的输出缩放 |
| **PD 系数** | `m20_policy_runner.hpp:127-128` | `kp_`, `kd_` (如 80, 2) |
| **观测空间** | `m20_policy_runner.hpp:262-303` | `BuildLegacyObservation` / `BuildHistoryVaePolicyObservation` |
| **策略降采样** | `m20_policy_runner.hpp:115` | `SetDecimation(N)` — 每 N 个控制步推理一次 |
| **MJCF 模型** | `model/M20/mjcf/M20.xml` | 替换为你的机器人模型文件 |
| **MuJoCo 初始姿态** | `mujoco_simulation_ros2.py:47-52` | `JOINT_INIT` 字典 |
| **MuJoCo 标定** | `mujoco_simulation_ros2.py:43-45` | `JOINT_DIR` 方向数组、`POS_OFFSET_DEG` 偏移数组 |
| **ROS2 节点名** | 两处 | C++节点名和 Python 节点名 |

### 6.2 场景：四足机器人（非轮足）

除了 6.1 所有修改外：

1. **修改腿结构和动作处理**

```cpp
// m20_policy_runner.hpp:369-380
// 所有关节都是位置控制，去掉轮子速度特殊处理
for (int i = 0; i < action_dim; ++i) {
    robot_action.goal_joint_pos(i) = tmp_action_eigen(i);
    robot_action.goal_joint_vel(i) = 0.0f;
}
```

2. **修改状态机** — `qw_state_machine.hpp` 去掉/修改轮足相关初始化

3. **修改策略选择** — `rl_control_state.hpp:163-183` 简化策略切换逻辑

4. **修改 MJCF** — 去掉 wheel 关节，改为纯腿模型

### 6.3 场景：使用自己的训练策略

**关键对接清单**（必须匹配的参数）：

```cpp
// 1. ONNX 模型输入名称（m20_policy_runner.hpp:84-86）
// 单输入策略:
const char* input_names_[1] = {"obs"};
// 双输入策略 (History-VAE):
const char* history_input_names_[2] = {"obs", "obs_history"};

// 2. 输出名称（m20_policy_runner.hpp:87）
const char* output_names_[1] = {"actions"};

// 3. 观测维度（m20_policy_runner.hpp:32-36）
static constexpr int legacy_observation_dim = 57;
static constexpr int history_vae_observation_dim = 58;
static constexpr int history_vae_frame_dim = 61;
static constexpr int history_vae_history_length = 10;
static constexpr int history_vae_history_dim = 610;

// 4. ONNX 文件路径（rl_control_state.hpp:111-114）
// 默认加载 4 个文件，按需修改
```

**步骤**：

1. 从训练项目导出 `policy.onnx`
2. 复制到 `policy/` 对应子目录
3. 确认 ONNX 的输入/输出维度和名称与代码一致
4. 编译 & 运行

> ONNX Runtime 会自动检测输入数量来判断是否为 History-VAE 策略：`uses_history_vae_ = (input_count >= 2)`。只需一个策略文件时，确保 `obs` 维度是 57 或 58。

### 6.4 场景：更换通信方式（脱离 ROS2）

```bash
./build.sh x86   # 无 ROS2 时自动进入独立编译
```

生成 `build/standalone/rl_deploy`，特点：
- 跳过 DDS 关节数据标定
- 不能与 MuJoCo ROS2 仿真通信
- 需要自己实现 `RobotInterface` 子类

**实现自定义 RobotInterface**：

```cpp
class MyRobotInterface : public RobotInterface {
public:
    MyRobotInterface() : RobotInterface("my_robot", 12) {}
    
    VecXf GetJointPosition() override { /* 从你的硬件读取 */ }
    Vec3f GetImuRpy() override { /* 从你的IMU读取 */ }
    // ... 实现所有纯虚函数
};
```

### 6.5 场景：自定义控制接口

继承 `UserCommandInterface` 基类：

```cpp
class MyControllerInterface : public UserCommandInterface {
public:
    UserCommand* GetUserCommand() override { return usr_cmd_; }
    void Start() override { /* 启动控制线程 */ }
    void Stop() override { /* 停止控制线程 */ }
};
```

然后在 `qw_state_machine.hpp:46-63` 中替换：

```cpp
uc_ptr_ = std::make_shared<MyControllerInterface>(robot_name_);
```

### 6.6 场景：调整 MuJoCo 仿真参数

```python
# mujoco_simulation_ros2.py:39
DT = 0.001                # 仿真步长（秒），默认 1ms
RENDER_INTERVAL = 50      # 渲染间隔（步数），即每 50ms 更新一次画面

# 控制参数
JOINT_DIR                 # 关节方向（±1）—— 与 C++ 侧的 joint_dir 对应
POS_OFFSET_DEG            # 位置偏移（度）—— 与 C++ 侧的 pos_offset 对应
```

---

## 7. 常见问题与调试技巧

### 7.1 起立时卡住 / 自碰撞

这不是 bug。MuJoCo 中多关节机器人起立时可能因自碰撞陷入稳定状态，解决方式：
- 关掉 MuJoCo 窗口重新运行
- 多次尝试 `z` → `c` 流程

### 7.2 策略文件找不到

```bash
# 编译后手工检查
ls -la install/m20_sdk_deploy/lib/m20_sdk_deploy/policy/
```

所有 `.onnx` 文件必须存在于该目录，CMake 配置中已包含自动安装规则。

### 7.3 查看 Topic 数据

```bash
# 确认通信正常
ros2 topic echo /JOINTS_DATA
ros2 topic echo /IMU_DATA
ros2 topic hz /JOINTS_DATA
ros2 topic hz /IMU_DATA
```

### 7.4 base_lin_vel 不可用

当前部署中 `base_lin_vel` 被固定为零（`m20_policy_runner.hpp:258-260`）。

如果你的训练策略依赖线速度：
1. 添加状态估计器（如 Kalman 滤波器融合 IMU + 关节编码器）
2. 修改 `EstimateBaseLinearVelocity()` 方法
3. 确保对 training 和 deployment 的一致性

### 7.5 编译问题

```bash
# 清理重建
rm -rf build install log
./build.sh x86
```

### 7.6 键盘无响应

- 确认终端处于焦点（MuJoCo 窗口会抢焦点，右键选 "always on top"）
- 确认状态进入了 RLControlMode（终端会有 `[POLICY] Active: ...` 输出）
- 使用 `ros2 topic echo /JOINTS_CMD` 确认命令正在发布

### 7.7 监控界面

项目提供了终端监控 UI：

```bash
# 直接运行 rl_deploy 并通过管道进入监控
ros2 run m20_sdk_deploy rl_deploy 2>&1 | python3 monitor_ui.py
```

显示内容：机器人状态、速度模式、方向、当前速度、剩余运动时间、IMU数据等。

---

## 8. 地形修改指南

本项目地形涉及两个层面：**部署仿真（MuJoCo）** 和 **训练配置（IsaacLab）**。

### 8.1 MuJoCo 仿真地形修改（部署侧，最直接）

地形定义在 `src/M20_sdk_deploy/model/M20/mjcf/M20.xml` 的 `<worldbody>` 中。目前有三段地形：

#### 8.1.1 平坦地面（始终存在）

```xml
<!-- M20.xml 第56行 -->
<geom name='floor' pos='0 0 0' size='100 100 .125' type='plane' material="MatPlane" condim='3'
      friction="1 0.01 0.01" group="0"/>
```

**修改方式**：调整 `size` 改变平面大小，或修改 `material` 改变外观。

#### 8.1.2 不平整地形（uneven，第72-80行）

由倾斜的 box 组成，难度逐渐递增，从机器人附近一直延伸到约 5m：

```xml
<geom name="terrain_uneven_00" type="box" pos="-0.60 0 0.018" size="0.36 1.60 0.018"
      euler="0 0.04 0" material="terrain_uneven" friction="1.2 0.01 0.01" condim="3"/>
```

| 参数 | 含义 |
|------|------|
| `pos="x y z"` | 位置（x=前进方向，z=高度） |
| `size="x y z"` | 半尺寸（z=高度/2） |
| `euler="roll pitch yaw"` | 倾斜角度（弧度） |
| `friction` | 摩擦系数（三个值：滑动摩擦、滚动摩擦、扭转摩擦） |

#### 8.1.3 粗糙地形（rough，第83-100行）

散布的小方块，模拟碎石/障碍物：

```xml
<geom name="terrain_rough_00" type="box" pos="5.25 -1.15 0.030" size="0.16 0.13 0.030"
      material="terrain_rough" friction="1.2 0.01 0.01" condim="3"/>
```

#### 8.1.4 随机网格方块地形（boxes，第63-69行，已被注释掉）

```xml
<!-- 取消注释即可启用 -->
<geom name="terrain_box_00" type="box" pos="1.40 -0.80 0.035" size="0.18 0.18 0.035"
      material="terrain_boxes" friction="1.2 0.01 0.01" condim="3"/>
```

#### 8.1.5 常用修改操作

| 目的 | 操作 |
|------|------|
| 去掉所有地形，只用平地 | 注释掉或删除所有 `terrain_uneven_*` 和 `terrain_rough_*` |
| 增加障碍物高度 | 增大 `size` 的 z 值和 `pos` 的 z 值 |
| 增加障碍物密度 | 添加更多 `<geom>` 元素，调整 `pos` 的 x/y 间距 |
| 改变倾斜角度 | 修改 `euler` 中的 pitch 值（如 `0.04` → `0.15`） |
| 添加台阶 | 添加新的 box geom，逐步增加 z |
| 改变摩擦 | 修改 `friction` 的第一个值（越大越不滑） |
| 改变外观 | 修改或创建新的 `<material>`，改变 `rgba` 颜色 |

#### 8.1.6 示例：添加高台阶

```xml
<geom name="step_01" type="box" pos="2.0 0 0.05" size="0.3 0.8 0.05"
      material="terrain_boxes" friction="1.2 0.01 0.01" condim="3"/>
<geom name="step_02" type="box" pos="2.6 0 0.10" size="0.3 0.8 0.10"
      material="terrain_boxes" friction="1.2 0.01 0.01" condim="3"/>
```

#### 8.1.7 MuJoCo 地形材料定义

在 `<asset>` 中（M20.xml 第47-49行）：

```xml
<material name="terrain_boxes" rgba="0.45 0.38 0.26 1"/>
<material name="terrain_rough" rgba="0.30 0.33 0.30 1"/>
<material name="terrain_uneven" rgba="0.38 0.44 0.34 1"/>
```

修改 `rgba` 值即可改变地形颜色外观。

---

### 8.2 IsaacLab 训练地形配置修改（训练侧）

如果你在训练新策略，地形由 `src/M20_sdk_deploy/config/terrain/` 下的配置文件控制。

#### 8.2.1 核心配置

`mixed_terrains_depth_camera_vae_env_cfg.py` 中的 `MIXED_TERRAINS_DEPTH_CAMERA_VAE_RING_CFG`（第51-89行）：

```python
MIXED_TERRAINS_DEPTH_CAMERA_VAE_RING_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),           # 每块地形单元大小 (m)
    border_width=1.0,           # 边界宽度 (m)
    num_rows=8,                 # 行数（沿前进方向）
    num_cols=8,                 # 列数
    horizontal_scale=0.1,       # 水平缩放
    vertical_scale=0.005,       # 垂直缩放（噪声幅度）
    slope_threshold=0.75,       # 坡度阈值
    sub_terrains={
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0,
            grid_width=0.45,
            grid_height_range=(0.025, 0.16),  # 方块高度范围 (m)
            platform_width=2.0,
            holes=False,
        ),
        "uneven": HfUnevenTerrainCfg(
            proportion=1.0,
            border_width=0.25,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
        ),
        "waves": terrain_gen.HfWaveTerrainCfg(
            proportion=1.0,
            amplitude_range=(0.02, 0.12),  # 波浪幅度范围 (m)
            num_waves=2,
            border_width=0.25,
        ),
    },
)
```

#### 8.2.2 三种子地形按行循环排列

`mixed_terrains_depth_camera_vae_env_cfg.py:38-44`：

```python
tile_pattern = sub_row % 3
if tile_pattern == 0:   → boxes（方块障碍）
elif tile_pattern == 1: → uneven（不平整）
else:                   → waves（波浪地形）
```

#### 8.2.3 训练地形关键修改参数速查

| 参数 | 位置 | 作用 |
|------|------|------|
| `size` | `TerrainGeneratorCfg` | 每格大小，(8,8)→更大地形单元 |
| `num_rows/cols` | `TerrainGeneratorCfg` | 地形网格数，更多→更长的训练路线 |
| `grid_height_range` | `boxes` 子地形 | 方块高度范围，`(0.025, 0.16)`→更高的障碍 |
| `amplitude_range` | `waves` 子地形 | 波浪幅度，`(0.02, 0.12)`→更陡的波浪 |
| `vertical_scale` | `TerrainGeneratorCfg` / `uneven` | 不平整地形的噪声幅度 |
| `difficulty_range` | 生成器内部 | 难度递增范围（从 `lower` 到 `upper`） |

#### 8.2.4 修改地形类型循环

在 `mixed_terrains_depth_camera_vae_env_cfg.py` 的 `_generate_curriculum_terrains` 方法中修改 `tile_pattern` 逻辑：

```python
# 示例：只用一种地形
tile_pattern = 0  # 全部为 boxes

# 示例：添加第四种地形
tile_pattern = sub_row % 4
if tile_pattern == 0:   sub_cfg = boxes_cfg
elif tile_pattern == 1: sub_cfg = uneven_cfg
elif tile_pattern == 2: sub_cfg = waves_cfg
else:                   sub_cfg = random_rough_cfg
```

---

### 8.3 快速操作总结

| 你想做什么 | 改哪个文件 | 怎么改 |
|------------|-----------|--------|
| 仿真中跑平地 | `model/M20/mjcf/M20.xml` | 注释掉 uneven/rough geoms |
| 仿真中加障碍 | `model/M20/mjcf/M20.xml` | 添加 `<geom type="box">` |
| 仿真中改地面外观 | `model/M20/mjcf/M20.xml` | 修改 `<material>` 的 `rgba` |
| 仿真中改摩擦 | `model/M20/mjcf/M20.xml` | 修改 geom 的 `friction` |
| 训练时改地形难度 | `config/terrain/*.py` | 调整 `grid_height_range` 等 |
| 训练时改地形类型 | `config/terrain/*.py` | 修改 `tile_pattern` 逻辑 |
| 训练时改地形布局 | `config/terrain/*.py` | 修改 `num_rows/cols` 或 `size` |
| 训练时只跑一种地形 | `config/terrain/*.py` | 将 `tile_pattern` 固定为一个值 |

---

## 9. 附录：关键文件清单

### 9.1 需要修改的核心文件（按优先级排序）

| 优先级 | 文件 | 典型修改场景 |
|---|---|---|
| ★★★★★ | `m20_policy_runner.hpp` | 换策略、换机器人、改观测空间 |
| ★★★★★ | `rl_control_state.hpp` | 换策略文件路径、改策略选择逻辑 |
| ★★★★☆ | `mujoco_simulation_ros2.py` | 换 MJCF 模型、调仿真参数 |
| ★★★★☆ | `qw_state_machine.hpp` | 换控制模式、换控制接口 |
| ★★★☆☆ | `m20_interface.hpp` | 换机器人硬件、调标定参数 |
| ★★★☆☆ | `dds_interface.hpp` | 改通信架构、调 Topic 名称 |
| ★★★☆☆ | `teleop_config.json` | 调加速度、最大速度、运动时长 |
| ★★☆☆☆ | `keyboard_interface.hpp` | 改键盘映射 |
| ★★☆☆☆ | `fixed_direction_keyboard_interface.hpp` | 改固定方向控制逻辑 |
| ★★☆☆☆ | `CMakeLists.txt` | 加依赖、改编译配置 |
| ★☆☆☆☆ | `state_machine_base.h` | 改控制循环频率 |

### 9.2 典型的适配工作流

```
1. 替换 MJCF 模型
   └─ M20.xml → my_robot.xml
   
2. 修改关节参数
   ├─ motor_num: 16 → N
   ├─ robot_order / policy_order
   ├─ dof_default / action_scale / kp_kd
   └─ JOINT_DIR / POS_OFFSET_DEG (Python侧)
   
3. 替换 ONNX 策略文件
   └─ policy/low_speed/policy.onnx
   
4. 确认维度匹配
   ├─ obs 维度 (57 或 58)
   ├─ obs_history 维度 (如有)
   └─ action 维度 (16)
   
5. 修改状态机（如需要）
   └─ 增加/删除状态

6. 编译并测试
   └─ ./build.sh x86 && ./run_m20_simulation.sh
```

---

*最后更新：2026-06-07*
---
改了什么
1.mujoco simulation ros2.py加了极轻量的 qpos 记录
    
    - 通过环境变量RECORDQPOS PATH 和RECORD QPOS INTERVAL控制在主循环里每N步做一次self.   recorded qpos.append(self.dataqpos.copy())，开销一次 numpyarray copy每10000条自动flush一次(防崩溃丢数据)
    - 模拟正常退出(ctrl+c)时目动保存到.npz
    - 不设环境变量时完全不影响原有逻辑

2. check wheel fk error offline.py新建的离线 headless 脚本

    - 不创建 viewer，不跑mj step(物理积分)，只跑mj forward(运动学)
    - 加载记录的qpos，逐帧回放，计算手动FKVSsite xpos 的误差
    - 输出每个轮子的 max/meanzerror、pos error 和超标次数统计

3. run m20 simulation.sh


    将环境变量透传到 gnome-terminal 里的MuJoCo 进程

使用方式

#1. 带记录运行模拟

    RECORD_QPOS_PATH=/tmp/qpos_recorded.npz RECORD_QPOS_INTERVAL=50 ./run_m20_simulation.sh

#2.模拟结束后(ctrl+c退出)，离线分析
    python3check wheel fk error offline .py \
    --xml src/M20 sdk deploy/model/M20/mjcf/M20.xml \
    --qpos-file /tmp/qpos recorded.npz \
    --eps 1e-6

    记录间隔50步=每0.05s一个快照，跑1分钟产生~1200条，文件大约几百:KB，完全不影响mujoco流畅性。