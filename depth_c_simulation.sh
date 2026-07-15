#!/bin/bash
set -e

# ============================================================
# depth_c_simulation.sh — Depth Camera Sim2Sim Launch Script
# ============================================================
#
# Launches:
#   1. RL Deploy (C++ policy inference)
#   2. MuJoCo Depth Simulation (Python, with depth camera)
#
# Additional topics vs standard sim2sim:
#   /DEPTH_IMAGE  (std_msgs/Float32MultiArray)  50 Hz
#     - data layout: [height, width, max_depth, pixel_0, pixel_1, ...]
#     - resolution: 64 x 48, max 10m
#
# Usage:
#   ./depth_c_simulation.sh [x86|aarch64]
#   RECORD_QPOS_PATH=/tmp/qpos.npz ./depth_c_simulation.sh  # with qpos recording
# ============================================================

cd "$(dirname "$0")"

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

  gnome-terminal --tab --title="RL Deploy" -- bash -lc "
    cd '$PWD';
    export ROS_DOMAIN_ID=$ROS_DOMAIN_ID;
    export LD_LIBRARY_PATH='$ONNXRUNTIME_LIB':\${LD_LIBRARY_PATH:-};
    $ROS_ENV_SETUP;
    source install/setup.bash;
    ros2 run m20_sdk_deploy rl_deploy 2>&1 | python3 monitor_ui.py;
    echo '--- RL Deploy finished. Press Enter to close ---';
    read
  "

  gnome-terminal --tab --title="MuJoCo Depth Sim" -- bash -lc "
    cd '$PWD';
    export ROS_DOMAIN_ID=$ROS_DOMAIN_ID;
    export RECORD_QPOS_PATH='${RECORD_QPOS_PATH:-}';
    export RECORD_QPOS_INTERVAL='${RECORD_QPOS_INTERVAL:-50}';
    export START_X='${START_X:-0}';
    export START_Y='${START_Y:-0}';
    export START_Z='${START_Z:-0.2}';
    $ROS_ENV_SETUP;
    source install/setup.bash;
    python3 src/M20_sdk_deploy/interface/robot/simulation/mujoco_simulation_depth_ros2.py;
    echo '--- MuJoCo Depth Sim finished. Press Enter to close ---';
    read
  "

  echo "Launched RL Deploy + MuJoCo Depth Simulation."
  echo ""
  echo "Topics:"
  echo "  /JOINTS_CMD   (rl_deploy → sim)"
  echo "  /JOINTS_DATA  (sim → rl_deploy)"
  echo "  /IMU_DATA     (sim → rl_deploy)"
  echo "  /DEPTH_IMAGE  (sim → external consumers, 50 Hz, 64x48)"
  echo ""
  echo "To view depth data:"
  echo "  ros2 topic echo /DEPTH_IMAGE --field data | head -20"
  echo ""
  echo "To record qpos for offline FK checking:"
  echo "  RECORD_QPOS_PATH=/tmp/qpos.npz ./depth_c_simulation.sh"
else
  echo "[WARN] ROS2 not found; cannot launch simulation."
  echo "[WARN] Please install ROS2 Humble or source your ROS2 setup."
fi
