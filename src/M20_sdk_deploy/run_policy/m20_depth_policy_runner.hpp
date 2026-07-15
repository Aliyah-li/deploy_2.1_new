/**
 * @file m20_depth_policy_runner.hpp
 * @brief M20 policy runner with depth camera input for CNN-based policies
 *
 * ONNX inputs:
 *   obs      (1, 57)          — 1D proprioceptive observation
 *   obs_2d_0 (1, 1, 48, 64)  — depth image (C=1, H=48, W=64)
 * ONNX output:
 *   actions  (1, 16)          — joint targets
 */

#pragma once
#define PI 3.14159265358979323846

#include "policy_runner_base.hpp"
#include <ctime>
#include <cmath>
#include <utility>
#include <unordered_map>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>

// Forward declaration (full include via rl_control_state)
class M20Interface;


class M20DepthPolicyRunner : public PolicyRunnerBase {
private:
    VecXf kp_, kd_;
    VecXf dof_default_eigen_policy, dof_default_eigen_robot;
    Vec3f max_cmd_vel_, gravity_direction = Vec3f(0., 0., -1.);
    VecXf dof_pos_default_;
    timespec system_time;

    const int motor_num = 16;
    static constexpr int observation_dim = 57;
    static constexpr int depth_height = 48;
    static constexpr int depth_width  = 64;
    static constexpr int depth_pixels = depth_height * depth_width;  // 3072
    const int action_dim = 16;
    float agent_timestep = 0.02;

    VecXf joint_pos_rl = VecXf(action_dim);
    VecXf joint_vel_rl = VecXf(action_dim);

    const std::string policy_path_;

    float omega_scale_ = 0.25;
    float dof_vel_scale_ = 0.05;
    VecXf imu_w_eigen, base_acc_eigen, motor_p_eigen, motor_v_eigen,
          current_action_eigen, last_action_eigen, current_observation_, projected_gravity,
          tmp_action_eigen;

    RobotAction robot_action;
    std::vector<std::string> robot_order = {
        "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint", "fl_wheel_joint",
        "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint", "fr_wheel_joint",
        "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint", "hl_wheel_joint",
        "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint", "hr_wheel_joint"};

    std::vector<std::string> policy_order = {
        "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint",
        "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint",
        "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint",
        "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint",
        "fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint",
    };

    std::vector<float> action_scale_robot = {0.125, 0.25, 0.25, 5,
                                             0.125, 0.25, 0.25, 5,
                                             0.125, 0.25, 0.25, 5,
                                             0.125, 0.25, 0.25, 5};

    Ort::SessionOptions session_options_;
    Ort::Session session_{nullptr};

    Ort::Env env_;
    std::vector<int> robot2policy_idx, policy2robot_idx;

    const char* input_names_[2]  = {"obs", "obs_2d_0"};
    const char* output_names_[1] = {"actions"};
    VecXf command;
    Ort::MemoryInfo memory_info{nullptr};
    std::array<int64_t, 2> input_observationShape = {1, observation_dim};
    std::array<int64_t, 4> input_depthShape = {1, 1, depth_height, depth_width};

    // Depth camera access
    M20Interface* ri_ptr_ = nullptr;

    // Pre-allocated depth tensor buffer
    std::vector<float> depth_buffer_;

public:
    M20DepthPolicyRunner(const std::string &policy_name, const std::string &policy_path,
                         M20Interface* robot_interface = nullptr)
        : PolicyRunnerBase(policy_name), policy_path_(policy_path),
          env_(ORT_LOGGING_LEVEL_WARNING, "M20DepthPolicyRunner"),
          session_options_{},
          session_{nullptr},
          memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
          ri_ptr_(robot_interface) {

        dof_default_eigen_policy.setZero(action_dim);
        dof_default_eigen_robot.setZero(action_dim);
        dof_default_eigen_policy << 0.0, -0.6,  1.0,
                                    0.0, -0.6,  1.0,
                                    0.0,  0.6, -1.0,
                                    0.0,  0.6, -1.0,
                                    0.0, 0.0, 0.0, 0.0;
        dof_default_eigen_robot << 0.0, -0.6,  1.0, 0.0,
                                   0.0, -0.6,  1.0, 0.0,
                                   0.0,  0.6, -1.0, 0.0,
                                   0.0,  0.6, -1.0, 0.0;
        SetDecimation(20);  // 50 Hz with 1ms timer (matches training)
        session_options_.SetIntraOpNumThreads(4);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        if (access(policy_path_.c_str(), F_OK) != 0) {
            std::cerr << "Model file not found: " << policy_path_ << std::endl;
            throw std::runtime_error("Model file missing");
        }

        session_ = Ort::Session(env_, policy_path_.c_str(), session_options_);
        ConfigureOnnxInputs();
        kp_ = Vec4f(80, 80, 80, 0.).replicate(4, 1);
        kd_ = Vec4f(2, 2, 2, 0.6).replicate(4, 1);

        robot2policy_idx = generate_permutation(robot_order, policy_order);
        policy2robot_idx = generate_permutation(policy_order, robot_order);

        robot_action.kp = kp_;
        robot_action.kd = kd_;
        robot_action.tau_ff = VecXf::Zero(motor_num);
        robot_action.goal_joint_pos = VecXf::Zero(motor_num);
        robot_action.goal_joint_vel = VecXf::Zero(motor_num);

        current_observation_.setZero(observation_dim);
        last_action_eigen.setZero(action_dim);
        tmp_action_eigen.setZero(action_dim);
        current_action_eigen.setZero(action_dim);

        depth_buffer_.resize(depth_pixels, 0.0f);

        memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    }

    ~M20DepthPolicyRunner() override = default;

    std::vector<int> generate_permutation(
        const std::vector<std::string>& from,
        const std::vector<std::string>& to,
        int default_index = 0)
    {
        std::unordered_map<std::string, int> idx_map;
        for (int i = 0; i < from.size(); ++i) {
            idx_map[from[i]] = i;
        }

        std::vector<int> perm;
        for (const auto& name : to) {
            auto it = idx_map.find(name);
            if (it != idx_map.end()) {
                perm.push_back(it->second);
            } else {
                perm.push_back(default_index);
            }
        }
        return perm;
    }

    void DisplayPolicyInfo() override {}

    void OnEnter(const RobotBasicState &rbs) override {
        run_cnt_ = 0;
        cmd_vel_input_.setZero();
        last_action_eigen.setZero(action_dim);
        tmp_action_eigen.setZero(action_dim);
        motor_p_eigen.setZero(action_dim);
        motor_v_eigen.setZero(action_dim);
    }

    void ConfigureOnnxInputs() {
        // Verify 2 inputs: obs (57) + obs_2d_0 (1,1,48,64)
        size_t input_count = session_.GetInputCount();
        if (input_count < 2) {
            std::cerr << "[DEPTH POLICY] WARNING: ONNX has only " << input_count
                      << " inputs, expected 2 (obs + obs_2d_0). Depth may not work.\n";
        }

        // Read actual shapes from model
        auto obs_shape = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (obs_shape.size() >= 2 && obs_shape[1] > 0) {
            input_observationShape = {obs_shape[0], obs_shape[1]};
        }

        auto depth_shape = session_.GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
        if (depth_shape.size() == 4) {
            input_depthShape = {depth_shape[0], depth_shape[1], depth_shape[2], depth_shape[3]};
        }

        std::cout << "[DEPTH POLICY] obs shape: [" << input_observationShape[0]
                  << ", " << input_observationShape[1] << "]\n";
        std::cout << "[DEPTH POLICY] depth shape: [" << input_depthShape[0]
                  << ", " << input_depthShape[1]
                  << ", " << input_depthShape[2]
                  << ", " << input_depthShape[3] << "]\n";
    }

    VecXf Onnx_infer(const VecXf& observation, const std::vector<float>& depth_data) {
        // Input 1: observation (1, 57)
        Ort::Value obs_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(observation.data()),
            observation.size(),
            input_observationShape.data(),
            input_observationShape.size()
        );

        // Input 2: depth image (1, 1, 48, 64)
        // Normalize by max_depth (10.0)
        std::vector<float> normalized_depth(depth_data.size());
        float max_d = 10.0f;
        for (size_t i = 0; i < depth_data.size(); ++i) {
            normalized_depth[i] = depth_data[i] / max_d;
        }

        Ort::Value depth_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            normalized_depth.data(),
            normalized_depth.size(),
            input_depthShape.data(),
            input_depthShape.size()
        );

        std::vector<Ort::Value> inputs;
        inputs.emplace_back(std::move(obs_tensor));
        inputs.emplace_back(std::move(depth_tensor));

        auto outputs = session_.Run(
            Ort::RunOptions{nullptr},
            input_names_,
            inputs.data(),
            2,
            output_names_,
            1
        );

        float* action_data = outputs[0].GetTensorMutableData<float>();
        Eigen::Map<Eigen::VectorXf> action_map(action_data, action_dim);
        return VecXf(action_map);
    }

    VecXf BuildObservation(const RobotBasicState &ro, const Vec3f& command) {
        Vec3f base_omgea = ro.base_omega * omega_scale_;
        Vec3f projected_gravity = ro.base_rot_mat.inverse() * gravity_direction;

        for (int i = 0; i < action_dim; ++i) {
            joint_pos_rl(i) = ro.joint_pos(robot2policy_idx[i]);
            joint_vel_rl(i) = ro.joint_vel(robot2policy_idx[i]) * dof_vel_scale_;
        }
        joint_pos_rl.segment(12, 4).setZero();
        joint_pos_rl -= dof_default_eigen_policy;

        VecXf observation(observation_dim);
        observation << base_omgea,
                       projected_gravity,
                       command,
                       joint_pos_rl,
                       joint_vel_rl,
                       last_action_eigen;
        return observation;
    }

    RobotAction getRobotAction(const RobotBasicState &ro, const UserCommand &uc) override {
        Vec3f command = Vec3f(uc.forward_vel_scale, uc.side_vel_scale, uc.turnning_vel_scale);
        std::cout << "\r[CMD] fwd=" << command(0) << " side=" << command(1)
                  << " yaw=" << command(2) << std::flush;

        // Build 1D observation
        current_observation_ = BuildObservation(ro, command);

        // Get latest depth image
        bool has_depth = false;
        if (ri_ptr_ != nullptr) {
            int h, w;
            float max_d;
            has_depth = ri_ptr_->GetDepthImage(depth_buffer_, h, w, max_d);
        }

        // If no depth yet (first frames), use zeros
        if (!has_depth || depth_buffer_.empty()) {
            depth_buffer_.assign(depth_pixels, 0.0f);
        }

        // ONNX inference with depth
        current_action_eigen = Onnx_infer(current_observation_, depth_buffer_);
        last_action_eigen = current_action_eigen;

        // Permute action from policy order to robot order
        for (int i = 0; i < action_dim; ++i) {
            tmp_action_eigen(i) = current_action_eigen(policy2robot_idx[i]);
            tmp_action_eigen(i) *= action_scale_robot[i];
        }
        tmp_action_eigen += dof_default_eigen_robot;

        // Set position goals for legs, velocity for wheels
        for (int i = 0; i < 4; ++i) {
            robot_action.goal_joint_pos.segment(i * 4, 3) = tmp_action_eigen.segment(i * 4, 3);
            robot_action.goal_joint_vel(i * 4 + 3) = tmp_action_eigen(i * 4 + 3);
        }

        // Zero wheel velocity when no command
        bool cmd_is_zero = (uc.forward_vel_scale == 0.0f &&
                            uc.side_vel_scale    == 0.0f &&
                            uc.turnning_vel_scale == 0.0f);
        if (cmd_is_zero) {
            for (int i = 0; i < 4; ++i)
                robot_action.goal_joint_vel(i * 4 + 3) = 0.0f;
        }

        ++run_cnt_;
        return robot_action;
    }

    void setDefaultJointPos(const VecXf& pos) {
        dof_pos_default_.setZero(motor_num);
        for (int i = 0; i < motor_num; ++i) {
            dof_pos_default_(i) = pos(i);
        }
    }
};
