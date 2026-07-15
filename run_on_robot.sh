#!/bin/bash

# 设置密码（明文！危险！）
PASSWORD="'"  # ← 替换为实际密码

cd ~/sdk_deploy || { echo "❌ Workspace not found!"; exit 1; }

# 使用 echo + sudo -S 自动输入密码
echo "$PASSWORD" | sudo -S -i << 'EOF'
#!/bin/bash

source /opt/ros/foxy/setup.bash
source /opt/robot/scripts/setup_ros2.sh

echo "✅ Sourced ROS2 environment."

sleep 3

ros2 service call /SDK_MODE drdds/srv/StdSrvInt32 "{command: 200}"

sleep 5

# 注意：这里必须用绝对路径（root 的 ~ 是 /root）
cd /home/user/sdk_deploy/
source /home/user/sdk_deploy/install/setup.bash

ros2 run m20_sdk_deploy rl_deploy

EOF

echo "✅ Done."
