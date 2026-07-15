#pragma once

#include "robot_interface.h"
#include "dds_interface.hpp"
#include <std_msgs/msg/float32_multi_array.hpp>
#include <mutex>
#include <vector>

class M20Interface : public DdsInterface {
protected:
    void ResetPositionOffset() {
        memset(data_updated_, 0, dof_num_ * sizeof(bool));
        this->SetJointCommand(MatXf::Zero(dof_num_, 5));

        VecXf last_joint_pos = this->GetJointPosition();
        VecXf current_joint_pos = this->GetJointPosition();
        int cnt = 0;
        while (!IsDataUpdatedFinished()) {
            ++cnt;
            usleep(1000);

            rclcpp::spin_some(this->get_node());
            current_joint_pos = this->GetJointPosition();
            for (int i = 0; i < dof_num_; ++i) {
                if (!data_updated_[i] && current_joint_pos(i) != last_joint_pos(i) &&
                    !std::isnan(current_joint_pos(i))) {
                    data_updated_[i] = true;
                    std::cout << "joint " << i << " data updated at " << cnt << " cnt!" << std::endl;
                }
            }
            last_joint_pos = current_joint_pos;

            if (cnt > 10000) {
                for (int i = 0; i < dof_num_; ++i) {
                    std::cout << i << " :" << data_updated_[i] << std::endl;
                }
                std::cout << "joint data update is not finished\n";
            }
        }
        for (int i = 1; i < dof_num_; i += 4) {
            if (current_joint_pos(i) < -140. / 180. * M_PI) {
                pos_offset_[i] = pos_offset_[i] + 360.;
                std::cout << "joint " << i << " offset is changed to " << pos_offset_[i] << "\n";
            } else if (current_joint_pos(i) > 140. / 180. * M_PI) {
                pos_offset_[i] = pos_offset_[i] - 360.;
                std::cout << "joint " << i << " offset is changed to " << pos_offset_[i] << "\n";
            }
        }
        for (int i = 2; i < dof_num_; i += 4) {
            if (current_joint_pos(i) < -164. / 180. * M_PI) {
                pos_offset_[i] = pos_offset_[i] + 360.;
                std::cout << "joint " << i << " offset is changed to " << pos_offset_[i] << "\n";
            } else if (current_joint_pos(i) > 164. / 180. * M_PI) {
                pos_offset_[i] = pos_offset_[i] - 360.;
                std::cout << "joint " << i << " offset is changed to " << pos_offset_[i] << "\n";
            }
        }
        for (int i = 0; i < dof_num_; ++i) {
            joint_config_[i].dir = joint_dir_[i];
            joint_config_[i].offset = Deg2Rad(pos_offset_[i]);
        }
    }

public:
    M20Interface(const std::string &robot_name) : DdsInterface(robot_name, 16) {
        battery_data_.resize(2 * BATTERY_DATA_SIZE);

        float init_pos_offset[16] = {-25, -131, 160, 0.,
                                     25, -131, 160, 0,
                                     -25, 131, -160, 0,
                                     25, 131, -160, 0};
        float joint_dir[16] = {1, 1, -1, 1,
                               1, -1, 1, -1,
                               -1, 1, -1, 1.,
                               -1, -1, 1, -1};
        for (int i = 0; i < dof_num_; ++i) {
            pos_offset_[i] = init_pos_offset[i];
            joint_dir_[i] = joint_dir[i];
            data_updated_[i] = false;
            joint_config_[i].dir = joint_dir_[i];
            joint_config_[i].offset = Deg2Rad(pos_offset_[i]);
        }

        // Subscribe to depth camera topic
        depth_sub_ = node_->create_subscription<std_msgs::msg::Float32MultiArray>(
            "/DEPTH_IMAGE", 10,
            [this](const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(depth_mutex_);
                if (msg->data.size() >= 3) {
                    depth_height_ = static_cast<int>(msg->data[0]);
                    depth_width_  = static_cast<int>(msg->data[1]);
                    depth_max_    = msg->data[2];
                    depth_image_.assign(msg->data.begin() + 3, msg->data.end());
                }
            });

        // Optional sim2sim signal. It is absent on hardware unless an
        // estimator explicitly publishes it, in which case zero is retained.
        base_lin_vel_sub_ = node_->create_subscription<std_msgs::msg::Float32MultiArray>(
            "/BASE_LIN_VEL", 10,
            [this](const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
                if (msg->data.size() < 3) return;
                std::lock_guard<std::mutex> lock(base_lin_vel_mutex_);
                base_lin_vel_ << msg->data[0], msg->data[1], msg->data[2];
            });
        base_position_sub_ = node_->create_subscription<std_msgs::msg::Float32MultiArray>(
            "/BASE_POSITION", 10,
            [this](const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
                if (msg->data.size() < 3) return;
                std::lock_guard<std::mutex> lock(base_position_mutex_);
                base_position_ << msg->data[0], msg->data[1], msg->data[2];
            });
    }

    // ---- Depth camera access ----
    bool GetDepthImage(std::vector<float> &out, int &h, int &w, float &max_dist) {
        std::lock_guard<std::mutex> lock(depth_mutex_);
        if (depth_image_.empty()) return false;
        out = depth_image_;
        h = depth_height_;
        w = depth_width_;
        max_dist = depth_max_;
        return true;
    }

    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr depth_sub_;
    std::mutex depth_mutex_;
    std::vector<float> depth_image_;
    int depth_height_ = 48;
    int depth_width_  = 64;
    float depth_max_  = 10.0f;

    Vec3f GetBaseLinearVelocity() override {
        std::lock_guard<std::mutex> lock(base_lin_vel_mutex_);
        return base_lin_vel_;
    }

    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr base_lin_vel_sub_;
    std::mutex base_lin_vel_mutex_;
    Vec3f base_lin_vel_ = Vec3f::Zero();

    Vec3f GetBasePosition() override {
        std::lock_guard<std::mutex> lock(base_position_mutex_);
        return base_position_;
    }

    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr base_position_sub_;
    std::mutex base_position_mutex_;
    Vec3f base_position_ = Vec3f::Zero();

    ~M20Interface() {
    }

    virtual void Start() {
        time_stamp_ = GetTimestampMs();
#ifdef STANDALONE_BUILD
        std::cout << "[WARN] STANDALONE_BUILD: skipping DDS joint-data calibration." << std::endl;
#else
        ResetPositionOffset();
#endif
    }

    virtual void Stop() {
    }

    virtual void SetJointCommand(Eigen::Matrix<float, Eigen::Dynamic, 5> input) {
        auto msg = drdds::msg::JointsDataCmd();
        for (int i = 0; i < dof_num_; ++i) {
            msg.data.joints_data[i].position =
                    (input(i, 1) - joint_config_[i].offset) * joint_config_[i].dir;
            msg.data.joints_data[i].velocity = input(i, 3) * joint_dir_[i];
            msg.data.joints_data[i].torque = input(i, 4) * joint_dir_[i];
            msg.data.joints_data[i].kp = input(i, 0);
            if (i % 4 == 3) {
                msg.data.joints_data[i].kp = 0;
            }
            msg.data.joints_data[i].kd = input(i, 2);
            msg.data.joints_data[i].control_word = kIndexMotorControl;
        }
        joint_cmd_pub_->publish(msg);

    }

};
