
# Sim2Sim entrypoints and policy paths

Do not change `./run_m20_simulation.sh`; keep it for the older mixed/low-speed/high-speed deployment flow.

New fixed-height and variable-height entrypoints are separated by folder:

```text
fixed_height_simulation/
├── fft_simulation.sh   -> policy/fixed_height_fft/policy.onnx
└── flat_simulation.sh  -> policy/fixed_height_flat/policy.onnx

variable_height_simulation/
├── fft_simulation.sh   -> policy/variable_height_fft/policy.onnx
└── flat_simulation.sh  -> policy/variable_height_flat/policy.onnx
```

Use the fixed-height FFT entrypoint for tasks such as:

```text
active_perceptive_residual_fft_self_supervised_multiscale
```

Use the fixed-height flat entrypoint for:

```text
flat_ppo_base
```

Use the variable-height FFT entrypoint for height-conditioned FFT/FFT2 policies. Use the variable-height flat entrypoint for height-conditioned flat/strong2 policies.

# M20 部署
https://github.com/DeepRoboticsLab/sdk_deploy

# Go2 部署
https://github.com/CAI23sbP/go2_parkour_deploy




source /opt/ros/humble/setup.bash
colcon build --packages-up-to m20_sdk_deploy --cmake-args -DBUILD_PLATFORM=x86
colcon build --packages-up-to rl_deploy --cmake-args -DBUILD_PLATFORM=x86
# 终端1 
export ROS_DOMAIN_ID=1
source install/setup.bash
ros2 run m20_sdk_deploy rl_deploy

# 终端2 
export ROS_DOMAIN_ID=1
source install/setup.bash
python3 src/M20_sdk_deploy/interface/robot/simulation/mujoco_simulation_ros2.py

# 真机部署
ssh user@10.21.31.103
scp -r ./src user@10.21.31.103:~/sdk_deploy
scp -r ./run_on_robot.sh user@10.21.31.103:~/


source /opt/ros/foxy/setup.bash #source ROS2 env
colcon build --packages-select m20_sdk_deploy --cmake-args -DBUILD_PLATFORM=arm
colcon build --packages-select rl_deploy --cmake-args -DBUILD_PLATFORM=arm
rm -rf ./build ./install ./log
# 真机操作
sudo su # Root
source /opt/ros/foxy/setup.bash #source ROS2 env
source /opt/robot/scripts/setup_ros2.sh
ros2 service call /SDK_MODE drdds/srv/StdSrvInt32 command:\ 200 # /200 is /JOINTS_DATA topic frequency, recommended below 500 Hz. This value can only be factors of 1000.

# Run
source install/setup.bash
ros2 run m20_sdk_deploy rl_deploy
ros2 run rl_deploy rl_deploy
# exit sdk mode：
ros2 service call /SDK_MODE drdds/srv/StdSrvInt32 command:\ 0


