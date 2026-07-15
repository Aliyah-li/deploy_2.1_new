"""
 * @file mujoco_simulation.py
 * @brief simulation in mujoco
 * @author Bo (Percy) Peng
 * @version 1.0
 * @date 2025-11-05
 *
 * @copyright Copyright (c) 2025  DeepRobotics
"""

import os
import sys
import time
import select
from pathlib import Path

from scipy.spatial.transform import Rotation
import numpy as np
import mujoco
import mujoco.viewer

import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from drdds.msg import ImuData, JointsData, JointsDataCmd, MetaType, ImuDataValue, JointsDataValue, JointData, JointDataCmd
from std_msgs.msg import Float32MultiArray



MODEL_NAME = "M20"
# Get the directory of the current Python file
CURRENT_DIR = Path(__file__).resolve().parent

# Define the XML path relative to the Python file
XML_PATH = CURRENT_DIR / ".." / ".." / ".." / "model" / "M20" / "mjcf" / "M20.xml"

# Convert to absolute path as string
XML_PATH = str(XML_PATH.resolve())
USE_VIEWER = True
DT = 0.001
RENDER_INTERVAL = 50

# Calibaration parameters (for sim-to-real consistency)
JOINT_DIR = np.array([1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1], dtype=np.float32)
POS_OFFSET_DEG = np.array([-25, 229, 160, 0, 25, -131, -200, 0, -25, -229, -160, 0, 25, 131, 200, 0], dtype=np.float32)
POS_OFFSET_RAD = POS_OFFSET_DEG / 180.0 * np.pi

JOINT_INIT = {
    "M20": np.array([-0.438, -1.16, 2.76, 0,
                     0.438, -1.16, 2.76, 0,
                     -0.438, 1.16, -2.76, 0,
                     0.438, 1.16, -2.76, 0], dtype=np.float32),
}


class MuJoCoSimulationNode(Node):
    def __init__(self,
                 model_key: str = MODEL_NAME,
                 xml_path: str = XML_PATH):

        super().__init__('mujoco_simulation')

        # 加载 MJCF
        if not os.path.isfile(xml_path):
            raise FileNotFoundError(f"Cannot find MJCF: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = DT
        self.data = mujoco.MjData(self.model)

        # 机器人自由度列表
        self.actuator_ids = [a for a in range(self.model.nu)]  # 0..15
        self.dof_num = len(self.actuator_ids)
        assert self.dof_num == 16, "Expected 16 DOF for M20"

        # 初始化站立姿态
        self._set_initial_pose(model_key)

        # 缓存
        self.kp_cmd = np.zeros((self.dof_num, 1), np.float32)
        self.kd_cmd = np.zeros_like(self.kp_cmd)
        self.pos_cmd = np.zeros_like(self.kp_cmd)
        self.vel_cmd = np.zeros_like(self.kp_cmd)
        self.tau_ff = np.zeros_like(self.kp_cmd)
        self.input_tq = np.zeros_like(self.kp_cmd)

        # IMU
        self.last_base_linvel = np.zeros((3, 1), np.float64)
        self.timestamp = 0.0

        # ---- qpos recording (lightweight, env-var controlled) ----
        self.record_qpos_path = os.environ.get('RECORD_QPOS_PATH', '')
        self.record_qpos_interval = int(os.environ.get('RECORD_QPOS_INTERVAL', '50'))
        self.recorded_qpos: list = []
        self._record_save_counter = 0
        if self.record_qpos_path:
            self.get_logger().info(
                f"[RECORD] Will record qpos every {self.record_qpos_interval} steps "
                f"to {self.record_qpos_path}"
            )

        self.get_logger().info(f"[INFO] MuJoCo model loaded, dof = {self.dof_num}")

        # ROS Publishers
        self.imu_pub = self.create_publisher(ImuData, '/IMU_DATA', 200)
        self.base_lin_vel_pub = self.create_publisher(Float32MultiArray, '/BASE_LIN_VEL', 200)
        self.base_position_pub = self.create_publisher(Float32MultiArray, '/BASE_POSITION', 200)
        self.joints_pub = self.create_publisher(JointsData, '/JOINTS_DATA', 200)

        # ROS Subscriber
        self.cmd_sub = self.create_subscription(
            JointsDataCmd,
            '/JOINTS_CMD',
            self._cmd_callback,
            50
        )

        # 可视化
        self.viewer = None
        if USE_VIEWER:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            # Set camera to follow the robot
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = 0  # Track the base body (usually body 0)

    def _set_initial_pose(self, key: str):
        """关节位置设置为与 PyBullet 脚本一致的初始角度"""
        self.start_x = float(os.environ.get('START_X', '0'))
        self.start_y = float(os.environ.get('START_Y', '0'))
        self.start_z = float(os.environ.get('START_Z', '0.2'))
        self.start_joint_init = JOINT_INIT[key].copy()
        qpos0 = self.data.qpos.copy()
        qpos0[7:7 + self.dof_num] = self.start_joint_init
        qpos0[:3] = np.array([self.start_x, self.start_y, self.start_z])
        qpos0[3:7] = np.array([1, 0, 0, 0])
        self.data.qpos[:] = qpos0
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.get_logger().info(
            f"[INIT] Robot start position: x={self.start_x}, y={self.start_y}, z={self.start_z}"
        )

    def _reset_to_start(self):
        """Teleport robot back to start position (keyboard triggered)."""
        qpos0 = self.data.qpos.copy()
        qpos0[:3] = np.array([self.start_x, self.start_y, self.start_z])
        qpos0[3:7] = np.array([1, 0, 0, 0])
        qpos0[7:7 + self.dof_num] = self.start_joint_init
        self.data.qpos[:] = qpos0
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.get_logger().info(
            f"[RESET] Teleported to x={self.start_x}, y={self.start_y}, z={self.start_z}"
        )

    def _check_keyboard(self):
        """Non-blocking check for '0' key to trigger reset."""
        if select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch == '0':
                self._reset_to_start()

    def _cmd_callback(self, msg: JointsDataCmd):
        """Convert received (published) positions/velocities to internal (raw)"""
        if len(msg.data.joints_data) != 16:
            self.get_logger().warn("Received JointsDataCmd with incorrect number of joints")
            return

        pub_pos = np.zeros(self.dof_num, dtype=np.float32)
        pub_vel = np.zeros(self.dof_num, dtype=np.float32)
        for i in range(self.dof_num):
            joint_cmd = msg.data.joints_data[i]
            self.kp_cmd[i] = joint_cmd.kp
            self.kd_cmd[i] = joint_cmd.kd
            pub_pos[i] = joint_cmd.position
            pub_vel[i] = joint_cmd.velocity
            self.tau_ff[i] = joint_cmd.torque  # tau_ff no processing

        # Convert: raw = published * dir + offset_rad
        self.pos_cmd.flat = pub_pos * JOINT_DIR + POS_OFFSET_RAD
        self.vel_cmd.flat = pub_vel * JOINT_DIR

    def start(self):
        self.get_logger().info("[KEYBOARD] Press '0' to reset robot to start position")

        # 主模拟循环
        step = 0
        last_time = time.time()
        while rclpy.ok():
            if time.time() - last_time >= DT:
                last_time = time.time()
                step += 1
                # 控制律
                self._apply_joint_torque()
                # 模拟一步
                mujoco.mj_step(self.model, self.data)

                # ---- record qpos (near-zero overhead) ----
                if self.record_qpos_path and step % self.record_qpos_interval == 0:
                    self.recorded_qpos.append(self.data.qpos.copy())
                    # Periodic flush for crash safety (every ~10000 snapshots)
                    if len(self.recorded_qpos) % 10000 == 0:
                        self._save_recorded_qpos()

                self.timestamp = step * DT

                # 采样 & 发送观测 (every 5 steps for 200 Hz)
                if step % 5 == 0:
                    self._publish_robot_state(step)

                # 可视化
                if self.viewer and step % RENDER_INTERVAL == 0:
                    self.viewer.sync()

            # Handle ROS callbacks + keyboard check
            rclpy.spin_once(self, timeout_sec=0.0)
            self._check_keyboard()

        # ---- save recorded qpos on clean exit ----
        self._save_recorded_qpos()

    def _save_recorded_qpos(self):
        """Save accumulated qpos snapshots to .npz."""
        if not self.record_qpos_path or not self.recorded_qpos:
            return
        arr = np.array(self.recorded_qpos, dtype=np.float64)
        np.savez_compressed(self.record_qpos_path, qpos=arr)
        self.get_logger().info(
            f"[RECORD] Saved {len(self.recorded_qpos)} qpos snapshots "
            f"(shape={arr.shape}) to {self.record_qpos_path}"
        )

    def _apply_joint_torque(self):
        # 当前关节状态
        q = self.data.qpos[7:7 + self.dof_num].reshape(-1, 1)
        dq = self.data.qvel[6:6 + self.dof_num].reshape(-1, 1)
        self.input_tq = (
                self.kp_cmd * (self.pos_cmd - q) +
                self.kd_cmd * (self.vel_cmd - dq) +
                self.tau_ff
        )

        # 写入 control 缓冲区
        self.data.ctrl[:] = self.input_tq.flatten()

    # --------------------------------------------------------
    def quaternion_to_euler(self, q):
        """
        Convert a quaternion to Euler angles (roll, pitch, yaw).
        """
        w, x, y, z = q

        # roll (X-axis rotation)
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)

        # pitch (Y-axis rotation)
        t2 = 2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)  # 防止数值漂移导致 |t2|>1
        pitch = np.arcsin(t2)

        # yaw (Z-axis rotation)
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)

        return np.array([roll, pitch, yaw], dtype=np.float32)

    # --------------------------------------------------------

    def _publish_robot_state(self, step: int):
        # ----- IMU -----
        q_world = self.data.sensordata[:4]  # quaternion (w, x, y, z) in MuJoCo convention
        rpy_rad = self.quaternion_to_euler(q_world)  # returns [roll, pitch, yaw] in radians

        # Convert to degrees
        rpy_deg = [angle * (180.0 / 3.141592653589793) for angle in rpy_rad]

        body_acc = self.data.sensordata[4:7]
        angvel_b = self.data.sensordata[7:10]  # body frame

        # MuJoCo free-joint translational qvel is world-frame. IsaacLab's
        # base_lin_vel observation is body-frame, so rotate it before publish.
        base_linvel_w = self.data.qvel[:3]
        base_rotation = Rotation.from_quat([q_world[1], q_world[2], q_world[3], q_world[0]])
        base_linvel_b = base_rotation.inv().apply(base_linvel_w)

        imu_msg = ImuData()
        imu_msg.header = MetaType()
        imu_msg.header.frame_id = 0
        stamp = Time()
        sec = int(self.timestamp)
        nanosec = int((self.timestamp - sec) * 1e9)
        stamp.sec = sec
        stamp.nanosec = nanosec
        imu_msg.header.stamp = stamp
        imu_msg.data = ImuDataValue()
        imu_msg.data.roll = float(rpy_deg[0])
        imu_msg.data.pitch = float(rpy_deg[1])
        imu_msg.data.yaw = float(rpy_deg[2])
        imu_msg.data.omega_x = float(angvel_b[0])
        imu_msg.data.omega_y = float(angvel_b[1])
        imu_msg.data.omega_z = float(angvel_b[2])
        imu_msg.data.acc_x = float(body_acc[0])
        imu_msg.data.acc_y = float(body_acc[1])
        imu_msg.data.acc_z = float(body_acc[2])
        self.imu_pub.publish(imu_msg)

        base_lin_vel_msg = Float32MultiArray()
        base_lin_vel_msg.data = [float(v) for v in base_linvel_b]
        self.base_lin_vel_pub.publish(base_lin_vel_msg)

        base_position_msg = Float32MultiArray()
        base_position_msg.data = [float(v) for v in self.data.qpos[:3]]
        self.base_position_pub.publish(base_position_msg)

        # ----- 关节 -----
        q = self.data.qpos[7:7 + self.dof_num]
        dq = self.data.qvel[6:6 + self.dof_num]
        tau = self.input_tq.flatten()

        # Convert raw to published: published = (raw - offset_rad) * dir
        pub_pos = (q - POS_OFFSET_RAD) * JOINT_DIR
        pub_vel = dq * JOINT_DIR
        pub_tau = tau * JOINT_DIR  # Torque also needs direction flip
        
        joints_msg = JointsData()
        joints_msg.header = MetaType()
        joints_msg.header.frame_id = 0
        stamp = Time()
        sec = int(self.timestamp)
        nanosec = int((self.timestamp - sec) * 1e9)
        stamp.sec = sec
        stamp.nanosec = nanosec
        joints_msg.header.stamp = stamp
        joints_msg.data = JointsDataValue()
        joints_msg.data.joints_data = [JointData() for _ in range(self.dof_num)]
        for i in range(self.dof_num):
            joint = joints_msg.data.joints_data[i]
            joint.name = [32, 32, 32, 32]  # Dummy name (four spaces)
            joint.data_id = 0  # Dummy
            joint.status_word = 1  # Normal
            joint.position = float(pub_pos[i])
            joint.torque = float(pub_tau[i])
            joint.velocity = float(pub_vel[i])
            joint.motion_temp = 40.0  # Dummy normal temp
            joint.driver_temp = 45.0  # Dummy normal temp
        self.joints_pub.publish(joints_msg)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    rclpy.init()
    sim_node = MuJoCoSimulationNode()
    sim_node.start()
    sim_node.destroy_node()
    rclpy.shutdown()
