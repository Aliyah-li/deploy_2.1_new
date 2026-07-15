#!/bin/bash
set -e

echo "=== Install ROS2 Humble + dependencies ==="
echo "Using Tsinghua mirror for ROS2 apt repo..."

# 1. Add ROS2 repo (Tsinghua mirror for China)
if [ ! -f /etc/apt/sources.list.d/ros2.list ]; then
  curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
fi
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] https://mirrors.ustc.edu.cn/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2.list

# 2. Update and install
apt-get update
apt-get install -y ros-humble-desktop python3-colcon-common-extensions python3-pip

# 3. Install Python packages (Tsinghua pip mirror)
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple "numpy<2.0" mujoco

echo ""
echo "=== Done! ==="
echo "Next: source /opt/ros/humble/setup.bash && colcon build --packages-up-to m20_sdk_deploy --cmake-args -DBUILD_PLATFORM=x86"
