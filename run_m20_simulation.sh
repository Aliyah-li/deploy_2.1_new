#!/bin/bash
set -e

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

  echo "Launched ROS2 RL deploy and MuJoCo simulation terminals."
  echo ""
  echo "To record qpos for offline FK checking, set env vars before running:"
  echo "  RECORD_QPOS_PATH=/tmp/qpos_recorded.npz RECORD_QPOS_INTERVAL=50 ./run_m20_simulation.sh"
  echo "Then analyze offline with:"
  echo "  python3 check_wheel_fk_error_offline.py --xml <xml> --qpos-file /tmp/qpos_recorded.npz"
else
  echo "[WARN] ROS2/ros2 not found; running standalone deploy binary only."
  echo "[WARN] MuJoCo ROS bridge requires rclpy/drdds and was not launched."
  ./install/m20_sdk_deploy/lib/m20_sdk_deploy/rl_deploy
fi
