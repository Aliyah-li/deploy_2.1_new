# Depth Camera Sim2Sim 部署指南

> 本文档说明如何使用带深度相机的 MuJoCo 仿真环境，与训练项目（active_suspension_mixed）中的 depth camera 策略对接。

---

## 目录

1. [架构概览](#1-架构概览)
2. [与标准 sim2sim 的区别](#2-与标准-sim2sim-的区别)
3. [快速开始](#3-快速开始)
4. [深度相机规格](#4-深度相机规格)
5. [ROS2 通信拓扑](#5-ros2-通信拓扑)
6. [关键文件](#6-关键文件)
7. [如何对接自己的 depth-camera 策略](#7-如何对接自己的-depth-camera-策略)
8. [调试技巧](#8-调试技巧)

---

## 1. 架构概览

```
┌──────────────────────┐   /JOINTS_CMD       ┌──────────────────────────────┐
│    rl_deploy (C++)   │ ──────────────────▶ │  MuJoCo Depth Simulation     │
│   策略推理 + 状态机    │                      │  (Python ROS2 Node)          │
│                      │ ◀────────────────── │                              │
│                      │   /IMU_DATA          │  ┌─ MuJoCo 物理引擎          │
│                      │   /JOINTS_DATA       │  ├─ IMU/Joint 传感器         │
└──────────────────────┘                      │  └─ Depth Camera 渲染器      │
                                              │       │                     │
                                              │       ▼                     │
                                              │  /DEPTH_IMAGE (50 Hz)        │
                                              │  64×48 float32 depth map     │
                                              └──────────────────────────────┘
```

**两层对比：**

| | 标准 sim2sim | Depth Camera sim2sim |
|---|---|---|
| 启动脚本 | `run_m20_simulation.sh` | `depth_c_simulation.sh` |
| 仿真脚本 | `mujoco_simulation_ros2.py` | `mujoco_simulation_depth_ros2.py` |
| 发布 Topic | IMU + Joints | IMU + Joints + Depth |
| 深度图 | 无 | 64×48, 50Hz, 0-10m |
| MJCF 模型 | M20.xml | M20.xml (新增 `<camera>`) |

---

## 2. 与标准 sim2sim 的区别

### 2.1 新增的 depth camera

在 `M20.xml` 的 `base_link` 内部增加了深度相机定义：

```xml
<camera name="depth_cam"
        pos="0.28 0 0.12"
        quat="0.8192 0 0.5736 0"
        fovy="60"
        resolution="64 48"/>
```

| 参数 | 值 | 说明 |
|------|------|------|
| `pos` | `(0.28, 0, 0.12)` | 安装在 base_link 前上方 |
| `quat` | `(0.8192, 0, 0.5736, 0)` | 前视 + 向下倾斜 ~20° |
| `fovy` | 60° | 垂直视场角 |
| `resolution` | 64×48 | 匹配 Isaac Lab 训练分辨率 |

### 2.2 深度图发布

仿真每 20 个物理步（50 Hz）渲染一次深度图，通过 `/DEPTH_IMAGE` topic 发布。

**消息格式** (`std_msgs/Float32MultiArray`):
```
data = [height, width, max_depth, pixel_0, pixel_1, pixel_2, ...]
         float    float   float      float    float    float

Total length: 3 + 64*48 = 3075 floats
```

**像素值含义**：从相机平面到障碍物的距离（米），范围 [0, max_depth]。天空/无遮挡为 max_depth。

### 2.3 性能开销

深度渲染每 20 步执行一次（50 Hz），使用 offscreen GPU 渲染。在测试中单次渲染耗时约 1-3ms，不影响物理仿真主循环的实时性。不影响 MuJoCo viewer 的流畅显示。

---

## 3. 快速开始

### 3.1 编译

```bash
cd /home/zhao/code/Deploy_active_suspension_mixed
source /opt/ros/humble/setup.bash  # 或你的 ROS2 环境
./build.sh x86
```

### 3.2 运行

```bash
# 基本运行
./depth_c_simulation.sh

# 带 qpos 记录（用于离线 FK 校验）
RECORD_QPOS_PATH=/tmp/qpos.npz ./depth_c_simulation.sh

# 无 GUI 模式
M20_USE_VIEWER=0 ./depth_c_simulation.sh
```

自动打开两个终端：
- **RL Deploy** — C++ 策略推理 + 监控 UI
- **MuJoCo Depth Sim** — 物理仿真 + 深度相机

### 3.3 键盘控制（同标准 sim2sim）

| 按键 | 功能 |
|------|------|
| `z` | 站立 |
| `c` | 进入 RL 控制 |
| `w/s` | 前进/后退 |
| `a/d` | 左移/右移 |
| `q/e` | 旋转 |
| `t` | 高/低速切换 |
| `r` | 急停（阻尼模式） |

---

## 4. 深度相机规格

完整规格（匹配 `active_suspension_mixed/tasks/suspension/depth_camera/depth_camera_env_cfg.py`）：

| 参数 | 值 |
|------|------|
| 分辨率 | 64 × 48 像素 (W × H) |
| 视场角 (FOV) | 60° |
| 最大距离 | 10.0 m |
| 近裁面 | 0.05 m |
| 数据类型 | float32, distance-to-image-plane |
| 发布频率 | 50 Hz |
| 安装位置 | base_link 坐标系 (0.28, 0, 0.12) |
| 朝向 | 前视 + 向下 ~20° |

---

## 5. ROS2 通信拓扑

| Topic | 类型 | 发布者 | 订阅者 | 频率 |
|-------|------|--------|--------|------|
| `/JOINTS_CMD` | `JointsDataCmd` | rl_deploy | MuJoCo Depth Sim | 200 Hz |
| `/JOINTS_DATA` | `JointsData` | MuJoCo Depth Sim | rl_deploy | 200 Hz |
| `/IMU_DATA` | `ImuData` | MuJoCo Depth Sim | rl_deploy | 200 Hz |
| `/DEPTH_IMAGE` | `Float32MultiArray` | MuJoCo Depth Sim | (待对接) | 50 Hz |
| `/BATTERY_DATA` | `BatteryData` | MuJoCo Depth Sim | rl_deploy | 低频 |

---

## 6. 关键文件

| 文件 | 说明 |
|------|------|
| `depth_c_simulation.sh` | 一键启动脚本 |
| `src/M20_sdk_deploy/interface/robot/simulation/mujoco_simulation_depth_ros2.py` | 带深度相机的仿真节点 |
| `src/M20_sdk_deploy/model/M20/mjcf/M20.xml` | MJCF 模型（含 depth_cam） |
| `DEPTH_CAMERA_SIM2SIM_GUIDE.md` | 本指南 |

标准 sim2sim 文件（保持不变）：

| 文件 | 说明 |
|------|------|
| `run_m20_simulation.sh` | 标准启动脚本（无深度） |
| `src/M20_sdk_deploy/interface/robot/simulation/mujoco_simulation_ros2.py` | 标准仿真节点 |
| `src/M20_sdk_deploy/run_policy/m20_policy_runner.hpp` | 策略推理（共用） |
| `src/M20_sdk_deploy/state_machine/quadruped_wheel/rl_control_state.hpp` | RL 状态机（共用） |

---

## 7. 如何对接自己的 depth-camera 策略

### 7.1 训练侧（Isaac Lab）

确认你的训练配置中深度相机参数与此处一致：

```python
# 训练配置中应匹配:
TiledCameraCfg(
    height=48,
    width=64,
    max_distance=10.0,
    data_type="distance_to_image_plane",
    pos=(0.28, 0.0, 0.12),
    # ...
)
```

### 7.2 策略部署侧（C++）

需要在 C++ 侧增加深度图接收和观测构建。当前 `m20_policy_runner.hpp` 仅支持 IMU + joint 观测。修改步骤：

1. **订阅 `/DEPTH_IMAGE`**：在 `dds_interface.hpp` 中增加 depth subscriber
2. **扩展观测构建**：在 `m20_policy_runner.hpp` 中增加 `BuildDepthObservation()` 方法
3. **更新 ONNX 输入**：如果策略需要额外输入（如 `depth_image`），需要在 `Onnx_infer()` 中添加对应 tensor

示例代码框架：
```cpp
// 1. 订阅 depth
depth_sub_ = node_->create_subscription<std_msgs::msg::Float32MultiArray>(
    "/DEPTH_IMAGE", 50,
    [this](const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
        // msg->data[0] = height, msg->data[1] = width, msg->data[2] = max_depth
        // msg->data[3:] = depth values (row-major)
        latest_depth_ = Eigen::Map<const Eigen::VectorXf>(
            msg->data.data() + 3, height * width
        );
    });

// 2. 构建 depth 观测
VecXf BuildDepthObservation() {
    VecXf obs(depth_height * depth_width);
    obs = latest_depth_ / max_depth_;  // normalize to [0, 1]
    return obs;
}
```

### 7.3 验证深度数据

```bash
# 查看深度图 topic 是否存在
ros2 topic list | grep DEPTH

# 查看发布频率
ros2 topic hz /DEPTH_IMAGE

# 查看一帧数据的头部（高度、宽度、最大深度）
ros2 topic echo /DEPTH_IMAGE --field data --once | head -5
```

---

## 8. 调试技巧

### 8.1 检查相机是否正确加载

```bash
python3 -c "
import mujoco
m = mujoco.MjModel.from_xml_path('src/M20_sdk_deploy/model/M20/mjcf/M20.xml')
print(f'Cameras: {m.ncam}')
print(f'depth_cam id: {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, \"depth_cam\")}')
"
```

### 8.2 离线渲染深度图

```python
import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path('src/M20_sdk_deploy/model/M20/mjcf/M20.xml')
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

renderer = mujoco.Renderer(m, height=48, width=64)
renderer.enable_depth_rendering()
renderer.update_scene(d, camera="depth_cam")
depth = renderer.render()  # shape (48, 64), float32

print(f"Depth range: {depth.min():.3f} - {depth.max():.3f} m")
```

### 8.3 性能分析

深度渲染有一定开销。如果仿真变慢：
- 降低深度发布频率：修改 `DEPTH_RATE` (默认 50 Hz)
- 减小渲染分辨率
- 在无头模式下运行 (`M20_USE_VIEWER=0`)

### 8.4 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `/DEPTH_IMAGE` 无数据 | depth_cam 未在 MJCF 中定义 | 检查 M20.xml 中是否有 `<camera name="depth_cam">` |
| 深度值全是 max_depth | 相机朝上看了天空 | 调整相机 quat 使其朝下 |
| 仿真启动慢 | GPU 初始化 | 正常现象，等待几秒 |
| rl_deploy 崩溃 | 策略文件缺失 | 确保 `src/M20_sdk_deploy/policy/` 下有 .onnx 文件 |

---

*最后更新：2026-06-08*
