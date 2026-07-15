#!/bin/bash
set -e

# ============================================================
# strong2_simulation.sh — Strong2 (变高度) Sim2Sim Launch
# ============================================================
#
# Launches:
#   1. Strong2 RL Deploy (C++ inference — variable-height command, 58-dim obs)
#   2. MuJoCo Simulation (Python)
#
# Canonical observation order comes from the exported training env.yaml.
# Height is appended last at index 57, after the previous-action block.
#
# Strong2 vs Strong PPO Base (old):
#   - Height is handled by the policy; action reference is fixed URDF -0.6/1.0
#   - Initial height = 0.38 m (matches StandUp stand_height_)
#   - PD target = action * scale + fixed URDF default
#   - Uses rl_deploy_strong2 (not rl_deploy_fft)
#
# Usage:
#   ./fft_simulation/strong2_simulation.sh [x86|aarch64]
# ============================================================

cd "$(dirname "$0")/.."

PLATFORM=${1:-x86}
ONNXRUNTIME_LIB="src/M20_sdk_deploy/third_party/onnxruntime/${PLATFORM}/lib"
CONDA_SH=""
ROS_ENV_SETUP=""

if [ -f /home/ubuntu/miniconda3/etc/profile.d/conda.sh ]; then
  CONDA_SH=/home/ubuntu/miniconda3/etc/profile.d/conda.sh
elif [ -f /home/john/Anaconda/etc/profile.d/conda.sh ]; then
  CONDA_SH=/home/john/Anaconda/etc/profile.d/conda.sh
fi

export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-1}
export LD_LIBRARY_PATH="${ONNXRUNTIME_LIB}:${LD_LIBRARY_PATH:-}"

if [ -f /opt/ros/humble/setup.bash ]; then
  source /opt/ros/humble/setup.bash
  ROS_ENV_SETUP="source /opt/ros/humble/setup.bash"
elif [ -n "${CONDA_SH}" ]; then
  source "${CONDA_SH}"
  conda activate ros2_jazzy || true
  ROS_ENV_SETUP="source ${CONDA_SH}; conda activate ros2_jazzy"
fi

if command -v ros2 >/dev/null 2>&1; then
  source install/setup.bash

  # ── Terminal 1: Strong2 RL Deploy ──
  gnome-terminal --tab --title="RL Deploy (Strong2 变高度)" -- bash -lc "
    cd '$PWD';
    export ROS_DOMAIN_ID=$ROS_DOMAIN_ID;
    export LD_LIBRARY_PATH='$ONNXRUNTIME_LIB':\${LD_LIBRARY_PATH:-};
    export STRONG2_POLICY_SUBDIR=strong_ppo_base;
    $ROS_ENV_SETUP;
    source install/setup.bash;
    ros2 run m20_sdk_deploy rl_deploy_strong2 2>&1 | tee /tmp/strong2_debug.log | python3 monitor_ui.py;
    echo '--- RL Deploy finished. Debug log: /tmp/strong2_debug.log ---';
    echo '--- Press Enter to close ---';
    read
  "

  # ── Terminal 2: MuJoCo Sim ──
  gnome-terminal --tab --title="MuJoCo Sim" -- bash -lc "
    cd '$PWD';
    export ROS_DOMAIN_ID=$ROS_DOMAIN_ID;
    export RECORD_QPOS_PATH='${RECORD_QPOS_PATH:-}';
    export RECORD_QPOS_INTERVAL='${RECORD_QPOS_INTERVAL:-50}';
    export START_X='${START_X:-0}';
    export START_Y='${START_Y:-0}';
    export START_Z='${START_Z:-0.2}';
    $ROS_ENV_SETUP;
    source install/setup.bash;
    python3 src/M20_sdk_deploy/interface/robot/simulation/mujoco_simulation_ros2.py;
    echo '--- MuJoCo finished. Press Enter to close ---';
    read
  "

  echo "Launched Strong2 (变高度) + MuJoCo Simulation."
  echo ""
  echo "┌──────────────────────────────────────────────────────┐"
  echo "│ 键盘敲在 'RL Deploy (Strong2 变高度)' 终端里！       │"
  echo "│ [STATUS] 行会直接显示在这个终端                       │"
  echo "└──────────────────────────────────────────────────────┘"
  echo ""
  echo "Controls:"
  echo "  Height:   W (up) / S (down)   0.32 – 0.425 m"
  echo "  Yaw:      hold A (left) / D (right)"
  echo "  Move:     hold Q (current direction)"
  echo "  Dir:      1(x) / 2(y) / 3(diagonal)"
  echo "  Mode:     Z (stand) → C (RL control) → R (damping)"
  echo ""
  echo "★ 变高度: policy receives h_cmd; action reference stays at URDF default"
  echo "  Start height: 0.38 m (matches StandUp stand_height_)"
  echo "  Nominal: 0.40 m"
  echo ""
  echo "Obs 58-dim (training env.yaml order):"
  echo "  [0:3]   ang_vel        3"
  echo "  [3:6]   proj_gravity   3"
  echo "  [6:9]   vel_cmd        3"
  echo "  [9:25]  joint_pos_rel  16"
  echo "  [25:41] joint_vel      16"
  echo "  [41:57] last_action    16"
  echo "  [57]    height_cmd     1"
  echo "  3+3+3+16+16+16+1 = 58 ✓"
  echo ""
  echo "Policy: strong_ppo_base/policy.onnx"
  echo "Debug log: /tmp/strong2_debug.log"
else
  echo "[WARN] ROS2 not found; cannot launch simulation."
fi
