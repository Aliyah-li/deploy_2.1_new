#!/bin/bash
set -e

# ============================================================
# variable_height_simulation/fft_simulation.sh — variable-height FFT Sim2Sim
# ============================================================
#
# Launches:
#   1. FFT2 RL Deploy (C++ inference — single ONNX, variable-height command)
#   2. MuJoCo Simulation (Python)
#
# FFT2 policy features (vs FFT):
#   - Variable height is handled by the policy from the height observation
#   - Initial height = 0.38 m (matches StandUp stand_height_)
#   - PD target = action * scale + fixed URDF default (matches training)
#   - joint_pos_rel = dof_pos - fixed URDF default
#   - Single ONNX: policy/variable_height_fft/policy.onnx
#   - 58-dim obs, 4-dim cmd (vx, vy, heading, height)
#   - 50 Hz control rate
#
# Usage:
#   ./variable_height_simulation/fft_simulation.sh [x86|aarch64]
#   RECORD_QPOS_PATH=/tmp/qpos.npz ./variable_height_simulation/fft_simulation.sh
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

  # ── Terminal 1: FFT2 RL Deploy (keyboard works HERE) ──
  gnome-terminal --tab --title="RL Deploy (FFT2 变高度)" -- bash -lc "
    cd '$PWD';
    export ROS_DOMAIN_ID=$ROS_DOMAIN_ID;
    export LD_LIBRARY_PATH='$ONNXRUNTIME_LIB':\${LD_LIBRARY_PATH:-};
    export POLICY_SUBDIR=variable_height_fft;
    $ROS_ENV_SETUP;
    source install/setup.bash;
    ros2 run m20_sdk_deploy rl_deploy_fft2 2>&1 | tee /tmp/fft2_debug.log | python3 monitor_ui.py;
    echo '--- RL Deploy finished. Debug log: /tmp/fft2_debug.log ---';
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

  echo "Launched FFT2 (变高度) + MuJoCo Simulation."
  echo ""
  echo "┌─────────────────────────────────────────────────────┐"
  echo "│ 键盘敲在 'RL Deploy (FFT2 变高度)' 终端里！         │"
  echo "│ [STATUS] 行会直接显示在这个终端                      │"
  echo "└─────────────────────────────────────────────────────┘"
  echo ""
  echo "Controls:"
  echo "  Height:   W (up) / S (down)   0.32 – 0.425 m"
  echo "  Yaw:      hold A (left) / D (right)"
  echo "  Move:     hold Q (current direction)"
  echo "  Dir:      1(x) / 2(y) / 3(diagonal)"
  echo "  Mode:     Z (stand) → C (RL control) → R (damping)"
  echo ""
  echo "★ 变高度: policy receives h_cmd; action reference stays at URDF default"
  echo "  Start height: 0.38 m (matches StandUp)"
  echo "  Nominal: 0.40 m"
  echo "  Range: 0.32 – 0.425 m"
  echo ""
  echo "Policy: src/M20_sdk_deploy/policy/variable_height_fft/policy.onnx"
  echo "Debug log: /tmp/fft2_debug.log"
  echo ""
  echo "To record qpos:"
  echo "  RECORD_QPOS_PATH=/tmp/qpos.npz ./variable_height_simulation/fft_simulation.sh"
else
  echo "[WARN] ROS2 not found; cannot launch simulation."
  echo "[WARN] Please install ROS2 Humble or source your ROS2 setup."
fi
