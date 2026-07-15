#!/bin/bash
set -e

PLATFORM=${1:-x86}
STANDALONE_BUILD=0
CONDA_SH=""
ROS_SETUP=""

sync_policy_subdir() {
  local subdir="$1"
  local src_dir="src/M20_sdk_deploy/policy/${subdir}"
  local dst_dir="install/m20_sdk_deploy/lib/m20_sdk_deploy/policy/${subdir}"

  if [ -d "${src_dir}" ] && ls "${src_dir}"/*.onnx >/dev/null 2>&1; then
    mkdir -p "${dst_dir}"
    cp "${src_dir}"/*.onnx "${dst_dir}/"
    echo "  Policy files installed from ${src_dir}/"
  fi
}

# Activate conda env so colcon builds against the same Python used at runtime,
# when that environment is available on the current machine.
if [ -f /home/john/Anaconda/etc/profile.d/conda.sh ]; then
  CONDA_SH=/home/john/Anaconda/etc/profile.d/conda.sh
elif [ -f /home/ubuntu/miniconda3/etc/profile.d/conda.sh ]; then
  CONDA_SH=/home/ubuntu/miniconda3/etc/profile.d/conda.sh
fi

if [ -f /opt/ros/humble/setup.bash ]; then
  source /opt/ros/humble/setup.bash
  ROS_SETUP=/opt/ros/humble/setup.bash
elif [ -n "${CONDA_SH}" ]; then
  source "${CONDA_SH}"
  conda activate ros2_jazzy || true
fi

if command -v colcon >/dev/null 2>&1 && command -v ros2 >/dev/null 2>&1; then
  echo "Building rl_deploy + rl_deploy_fft + rl_deploy_fft2 + rl_deploy_strong2 + rl_deploy_depth ..."
  colcon build --packages-up-to m20_sdk_deploy --cmake-args -DBUILD_PLATFORM=$PLATFORM

  # always force-copy config so edits take effect without full CMake rebuild
  cp src/M20_sdk_deploy/config/teleop_config.json \
     install/m20_sdk_deploy/lib/m20_sdk_deploy/teleop_config.json
  cp src/M20_sdk_deploy/config/mixed_terrains_history_vae_config.json \
     install/m20_sdk_deploy/lib/m20_sdk_deploy/mixed_terrains_history_vae_config.json

  sync_policy_subdir fixed_height_fft
  sync_policy_subdir fixed_height_flat
  sync_policy_subdir variable_height_fft
  sync_policy_subdir variable_height_flat
else
  STANDALONE_BUILD=1
  echo "[WARN] ROS2/colcon not found; building standalone compile-check binary."
  mkdir -p build/standalone install/m20_sdk_deploy/lib/m20_sdk_deploy

  ONNXRUNTIME_LIB="src/M20_sdk_deploy/third_party/onnxruntime/${PLATFORM}/lib"
  EVDEV_LIB="/lib/x86_64-linux-gnu/libevdev.so.2"
  if [ ! -f "${EVDEV_LIB}" ]; then
    EVDEV_LIB="-levdev"
  fi
  g++ -std=c++17 -O2 -fPIC -w -DSTANDALONE_BUILD \
    -Isrc/M20_sdk_deploy/standalone_stubs \
    -Isrc/M20_sdk_deploy/include/types \
    -Isrc/M20_sdk_deploy/include/utils \
    -Isrc/M20_sdk_deploy/interface/robot \
    -Isrc/M20_sdk_deploy/interface/user_command \
    -Isrc/M20_sdk_deploy/state_machine \
    -Isrc/M20_sdk_deploy/run_policy \
    -Isrc/M20_sdk_deploy/third_party/eigen \
    -Isrc/M20_sdk_deploy/third_party/gamepad/include \
    -Isrc/M20_sdk_deploy/third_party/onnxruntime/${PLATFORM}/include \
    src/M20_sdk_deploy/main.cpp \
    src/M20_sdk_deploy/state_machine/parameters/m20_control_parameters.cpp \
    -L"${ONNXRUNTIME_LIB}" -Wl,-rpath,"\$ORIGIN/../lib/onnxruntime" \
    -lonnxruntime -lpthread -lm -lrt -ldl -lstdc++ "${EVDEV_LIB}" \
    -o build/standalone/rl_deploy

  # ── FFT standalone binary ──
  g++ -std=c++17 -O2 -fPIC -w -DSTANDALONE_BUILD \
    -Isrc/M20_sdk_deploy/standalone_stubs \
    -Isrc/M20_sdk_deploy/include/types \
    -Isrc/M20_sdk_deploy/include/utils \
    -Isrc/M20_sdk_deploy/interface/robot \
    -Isrc/M20_sdk_deploy/interface/user_command \
    -Isrc/M20_sdk_deploy/state_machine \
    -Isrc/M20_sdk_deploy/run_policy \
    -Isrc/M20_sdk_deploy/third_party/eigen \
    -Isrc/M20_sdk_deploy/third_party/gamepad/include \
    -Isrc/M20_sdk_deploy/third_party/onnxruntime/${PLATFORM}/include \
    src/M20_sdk_deploy/main_fft.cpp \
    src/M20_sdk_deploy/state_machine/parameters/m20_control_parameters.cpp \
    -L"${ONNXRUNTIME_LIB}" -Wl,-rpath,"\$ORIGIN/../lib/onnxruntime" \
    -lonnxruntime -lpthread -lm -lrt -ldl -lstdc++ "${EVDEV_LIB}" \
    -o build/standalone/rl_deploy_fft

  # ── FFT2 standalone binary (变高度) ──
  g++ -std=c++17 -O2 -fPIC -w -DSTANDALONE_BUILD \
    -Isrc/M20_sdk_deploy/standalone_stubs \
    -Isrc/M20_sdk_deploy/include/types \
    -Isrc/M20_sdk_deploy/include/utils \
    -Isrc/M20_sdk_deploy/interface/robot \
    -Isrc/M20_sdk_deploy/interface/user_command \
    -Isrc/M20_sdk_deploy/state_machine \
    -Isrc/M20_sdk_deploy/run_policy \
    -Isrc/M20_sdk_deploy/third_party/eigen \
    -Isrc/M20_sdk_deploy/third_party/gamepad/include \
    -Isrc/M20_sdk_deploy/third_party/onnxruntime/${PLATFORM}/include \
    src/M20_sdk_deploy/main_fft2.cpp \
    src/M20_sdk_deploy/state_machine/parameters/m20_control_parameters.cpp \
    -L"${ONNXRUNTIME_LIB}" -Wl,-rpath,"\$ORIGIN/../lib/onnxruntime" \
    -lonnxruntime -lpthread -lm -lrt -ldl -lstdc++ "${EVDEV_LIB}" \
    -o build/standalone/rl_deploy_fft2

  # ── Strong2 standalone binary (变高度, strong_ppo_base) ──
  g++ -std=c++17 -O2 -fPIC -w -DSTANDALONE_BUILD \
    -Isrc/M20_sdk_deploy/standalone_stubs \
    -Isrc/M20_sdk_deploy/include/types \
    -Isrc/M20_sdk_deploy/include/utils \
    -Isrc/M20_sdk_deploy/interface/robot \
    -Isrc/M20_sdk_deploy/interface/user_command \
    -Isrc/M20_sdk_deploy/state_machine \
    -Isrc/M20_sdk_deploy/run_policy \
    -Isrc/M20_sdk_deploy/third_party/eigen \
    -Isrc/M20_sdk_deploy/third_party/gamepad/include \
    -Isrc/M20_sdk_deploy/third_party/onnxruntime/${PLATFORM}/include \
    src/M20_sdk_deploy/main_strong2.cpp \
    src/M20_sdk_deploy/state_machine/parameters/m20_control_parameters.cpp \
    -L"${ONNXRUNTIME_LIB}" -Wl,-rpath,"\$ORIGIN/../lib/onnxruntime" \
    -lonnxruntime -lpthread -lm -lrt -ldl -lstdc++ "${EVDEV_LIB}" \
    -o build/standalone/rl_deploy_strong2

  cp build/standalone/rl_deploy install/m20_sdk_deploy/lib/m20_sdk_deploy/rl_deploy
  cp build/standalone/rl_deploy_fft install/m20_sdk_deploy/lib/m20_sdk_deploy/rl_deploy_fft
  cp build/standalone/rl_deploy_fft2 install/m20_sdk_deploy/lib/m20_sdk_deploy/rl_deploy_fft2
  cp build/standalone/rl_deploy_strong2 install/m20_sdk_deploy/lib/m20_sdk_deploy/rl_deploy_strong2
  cp src/M20_sdk_deploy/config/teleop_config.json \
     install/m20_sdk_deploy/lib/m20_sdk_deploy/teleop_config.json
  cp src/M20_sdk_deploy/config/mixed_terrains_history_vae_config.json \
     install/m20_sdk_deploy/lib/m20_sdk_deploy/mixed_terrains_history_vae_config.json

  sync_policy_subdir fixed_height_fft
  sync_policy_subdir fixed_height_flat
  sync_policy_subdir variable_height_fft
  sync_policy_subdir variable_height_flat
fi

echo ""
echo "Build done. Config synced."
if [ "${STANDALONE_BUILD}" = "1" ]; then
  echo "Standalone binaries:"
  echo "  build/standalone/rl_deploy         (standard)"
  echo "  build/standalone/rl_deploy_fft     (FFT sim2sim)"
  echo "  build/standalone/rl_deploy_fft2    (FFT2 变高度)"
  echo "  build/standalone/rl_deploy_strong2 (Strong2 变高度)"
else
  echo "Run with:"
  echo "  source install/setup.bash && ros2 run m20_sdk_deploy rl_deploy         (standard)"
  echo "  source install/setup.bash && ros2 run m20_sdk_deploy rl_deploy_depth   (depth camera)"
  echo "  source install/setup.bash && ros2 run m20_sdk_deploy rl_deploy_fft     (FFT sim2sim)"
  echo "  source install/setup.bash && ros2 run m20_sdk_deploy rl_deploy_fft2    (FFT2 变高度)"
  echo "  source install/setup.bash && ros2 run m20_sdk_deploy rl_deploy_strong2 (Strong2 变高度)"
  echo ""
  echo "  Launch scripts:"
  echo "    ./run_m20_simulation.sh                  (standard sim2sim)"
  echo "    ./depth_c_simulation.sh                  (depth camera sim2sim)"
  echo "    ./fixed_height_simulation/fft_simulation.sh      (fixed-height FFT)"
  echo "    ./fixed_height_simulation/flat_simulation.sh     (fixed-height flat PPO)"
  echo "    ./variable_height_simulation/fft_simulation.sh   (variable-height FFT)"
  echo "    ./variable_height_simulation/flat_simulation.sh  (variable-height flat PPO)"
fi
