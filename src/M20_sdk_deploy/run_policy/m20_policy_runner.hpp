/**
 * @file m20_policy_runner.hpp
 * @brief m20_policy_runner
 * @author Bo (Percy) Peng
 * @version 1.0
 * @date 2025-11-07
 * 
 * @copyright Copyright (c) 2025  DeepRobotics
 * 
 */

#pragma once
#define PI 3.14159265358979323846

#include "policy_runner_base.hpp"
#include <ctime>
#include <cmath>
#include <utility>
#include <unordered_map>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>

class M20PolicyRunner : public PolicyRunnerBase {
private:
    VecXf kp_, kd_;
    VecXf dof_default_eigen_policy, dof_default_eigen_robot;
    Vec3f max_cmd_vel_, gravity_direction = Vec3f(0., 0., -1.);
    VecXf dof_pos_default_;
    timespec system_time;

    // ── Height-dependent IK (matches standup_state.hpp) ──
    static constexpr float thigh_len_  = 0.25f;
    static constexpr float shank_len_  = 0.25f;
    static constexpr float hipy_limit_ = 2.443f;
    static constexpr float knee_limit_ = 2.758f;
    float current_height_ = 0.40f;

    static float GetHipYPosByHeight(float h) {
        float l1 = thigh_len_, l2 = shank_len_;
        if (std::fabs(h) >= l1 + l2) return 0.0f;
        float c = (l1*l1 + h*h - l2*l2) / (2.0f * h * l1);
        c = std::max(-1.0f, std::min(1.0f, c));
        float theta = -std::acos(c);
        return std::max(-hipy_limit_, std::min(hipy_limit_, theta));
    }
    static float GetKneePosByHeight(float h) {
        float l1 = thigh_len_, l2 = shank_len_;
        if (std::fabs(h) >= l1 + l2) return 0.0f;
        float c = (l1*l1 + l2*l2 - h*h) / (2.0f * l1 * l2);
        c = std::max(-1.0f, std::min(1.0f, c));
        float theta = static_cast<float>(M_PI) - std::acos(c);
        return std::max(-knee_limit_, std::min(knee_limit_, theta));
    }

    void UpdateDefaultPoseForHeight(float h) {
        h = std::max(0.20f, std::min(0.50f, h));
        if (std::fabs(h - current_height_) < 0.001f) return;
        current_height_ = h;
        float hipy = GetHipYPosByHeight(h);
        float knee = GetKneePosByHeight(h);
        // policy order: legs first, wheels last
        dof_default_eigen_policy << 0.0f,  hipy,  knee,
                                    0.0f,  hipy,  knee,
                                    0.0f, -hipy, -knee,
                                    0.0f, -hipy, -knee,
                                    0.0f, 0.0f, 0.0f, 0.0f;
        // robot order: interleaved per leg
        dof_default_eigen_robot  << 0.0f,  hipy,  knee, 0.0f,
                                    0.0f,  hipy,  knee, 0.0f,
                                    0.0f, -hipy, -knee, 0.0f,
                                    0.0f, -hipy, -knee, 0.0f;
    }

    const int motor_num = 16;
    static constexpr int legacy_observation_dim = 57;
    static constexpr int history_vae_observation_dim = 57;
    static constexpr int history_vae_frame_dim = 61;
    static constexpr int history_vae_history_length = 25;
    static constexpr int history_vae_history_dim = history_vae_frame_dim * history_vae_history_length;
    int observation_dim = legacy_observation_dim;
    int history_observation_dim = 0;
    const int action_dim = 16;
    float agent_timestep = 0.02;
    float current_time;
    bool is_fallen = true;

    VecXf joint_pos_rl = VecXf(action_dim);// in rl squenece
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

    const char* input_names_[1] = {"obs"}; // must keep the same as model export
    const char* history_input_names_[2] = {"obs", "obs_history"};
    const char* output_names_[1] = {"actions"};
    VecXf command;
    Ort::MemoryInfo memory_info{nullptr};
    std::array<int64_t, 2> input_observationShape = {1, legacy_observation_dim};
    std::array<int64_t, 2> input_historyShape = {1, history_vae_history_dim};
    bool uses_history_vae_ = false;
    std::deque<VecXf> history_frames_;
    
    float time_step = 0.;
    int stop_count = 1000;

public:
    M20PolicyRunner(const std::string &policy_name, const std::string &policy_path) :
            PolicyRunnerBase(policy_name), policy_path_(policy_path),env_(ORT_LOGGING_LEVEL_WARNING, "M20PolicyRunner"),
            session_options_{},
            session_{nullptr},
            memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {

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

        // 加载模型
        session_ = Ort::Session(env_, policy_path_.c_str(), session_options_);
        ConfigureOnnxInputs();
        kp_ = Vec4f(80, 80, 80, 0.).replicate(4, 1);
        kd_ = Vec4f(2, 2, 2, 0.6).replicate(4, 1);
        
        robot2policy_idx = generate_permutation(robot_order, policy_order);
        policy2robot_idx = generate_permutation(policy_order, robot_order);
        // for (int i = 0; i < action_dim; ++i){
        //     std::cout << "robot2policy_idx[" << i << "]: " << robot2policy_idx[i] << std::endl;
        //     std::cout << "policy2robot_idx[" << i << "]: " << policy2robot_idx[i] << std::endl;
        // }

        robot_action.kp = kp_;
        robot_action.kd = kd_;
        robot_action.tau_ff = VecXf::Zero(motor_num);
        robot_action.goal_joint_pos = VecXf::Zero(motor_num);
        robot_action.goal_joint_vel = VecXf::Zero(motor_num);


        current_observation_.setZero(observation_dim);
        last_action_eigen.setZero(action_dim);
        tmp_action_eigen.setZero(action_dim);
        current_action_eigen.setZero(action_dim);

        memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    }

    ~M20PolicyRunner() override = default;

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
                perm.push_back(default_index);  // 如果找不到，就填默认值
            }
        }

        return perm;
    }

    void DisplayPolicyInfo(){}

    void OnEnter(const RobotBasicState &rbs) {
        run_cnt_ = 0;
        cmd_vel_input_.setZero();
        last_action_eigen.setZero(action_dim);
        tmp_action_eigen.setZero(action_dim);
        motor_p_eigen.setZero(12);
        motor_v_eigen.setZero(motor_num);
        history_frames_.clear();
    }

    void ConfigureOnnxInputs() {
        size_t input_count = session_.GetInputCount();
        uses_history_vae_ = input_count >= 2;
        observation_dim = uses_history_vae_ ? history_vae_observation_dim : legacy_observation_dim;
        history_observation_dim = uses_history_vae_ ? history_vae_history_dim : 0;

        auto obs_shape = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (obs_shape.size() == 2 && obs_shape[1] > 0) {
            observation_dim = static_cast<int>(obs_shape[1]);
        }
        input_observationShape = {1, observation_dim};

        if (uses_history_vae_) {
            auto history_shape = session_.GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
            if (history_shape.size() == 2 && history_shape[1] > 0) {
                history_observation_dim = static_cast<int>(history_shape[1]);
            }
            input_historyShape = {1, history_observation_dim};
        }
    }

    VecXf Onnx_infer(const VecXf& current_observation, const VecXf* history_observation = nullptr){
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(current_observation.data()),
            current_observation.size(),
            input_observationShape.data(), 
            input_observationShape.size()
        );

        std::vector<Ort::Value> inputs;
        inputs.emplace_back(std::move(input_tensor));  // 避免拷贝构造
        const char* const* run_input_names = input_names_;
        size_t run_input_count = 1;

        if (uses_history_vae_) {
            if (history_observation == nullptr) {
                throw std::runtime_error("History-VAE policy requires obs_history input");
            }
            Ort::Value history_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                const_cast<float*>(history_observation->data()),
                history_observation->size(),
                input_historyShape.data(),
                input_historyShape.size()
            );
            inputs.emplace_back(std::move(history_tensor));
            run_input_names = history_input_names_;
            run_input_count = 2;
        }
        
        auto outputs = session_.Run(
            Ort::RunOptions{nullptr},
            run_input_names,
            inputs.data(),
            run_input_count,
            output_names_,
            1
        );

        float* action_data = outputs[0].GetTensorMutableData<float>();
        Eigen::Map<Eigen::VectorXf> action_map(action_data, action_dim);
        return VecXf(action_map);  // 返回一个Eigen向量的副本
    }

    Vec3f EstimateBaseLinearVelocity(const RobotBasicState &ro) const {
        // The referenced history-VAE task includes body-frame base_lin_vel in history.
        // This deploy stack does not expose base linear velocity, so keep the slot
        // deterministic rather than feeding stale or frame-inconsistent data.
        return Vec3f::Zero();
    }

    VecXf BuildLegacyObservation(const RobotBasicState &ro, const Vec3f& command) {
        Vec3f base_omgea = ro.base_omega * omega_scale_;
        Vec3f projected_gravity = ro.base_rot_mat.inverse() * gravity_direction;

        for (int i = 0; i < action_dim; ++i){
            joint_pos_rl(i) = ro.joint_pos(robot2policy_idx[i]);
            joint_vel_rl(i) = ro.joint_vel(robot2policy_idx[i]) * dof_vel_scale_;
        }
        joint_pos_rl.segment(12, 4).setZero();
        joint_pos_rl -= dof_default_eigen_policy;

        VecXf observation(legacy_observation_dim);
        observation << base_omgea,
                       projected_gravity,
                       command,
                       joint_pos_rl,
                       joint_vel_rl,
                       last_action_eigen;
        return observation;
    }

    VecXf BuildHistoryVaePolicyObservation(const RobotBasicState &ro, const Vec3f& command) {
        Vec3f base_omgea = ro.base_omega * omega_scale_;
        Vec3f projected_gravity = ro.base_rot_mat.inverse() * gravity_direction;

        for (int i = 0; i < action_dim; ++i){
            joint_pos_rl(i) = ro.joint_pos(robot2policy_idx[i]);
            joint_vel_rl(i) = ro.joint_vel(robot2policy_idx[i]) * dof_vel_scale_;
        }
        joint_pos_rl.segment(12, 4).setZero();
        joint_pos_rl -= dof_default_eigen_policy;

        VecXf observation(history_vae_observation_dim);
        observation << base_omgea,
                       projected_gravity,
                       command,
                       joint_pos_rl,
                       joint_vel_rl,
                       last_action_eigen;
        return observation;
    }

    VecXf BuildHistoryVaeFrame(const RobotBasicState &ro, const Vec3f& command) {
        Vec3f base_lin_vel = EstimateBaseLinearVelocity(ro);
        Vec3f projected_gravity = ro.base_rot_mat.inverse() * gravity_direction;
        float current_yaw = ro.base_rpy(2);

        VecXf history_joint_pos(action_dim);
        VecXf history_joint_vel(action_dim);
        for (int i = 0; i < action_dim; ++i){
            history_joint_pos(i) = ro.joint_pos(robot2policy_idx[i]);
            history_joint_vel(i) = ro.joint_vel(robot2policy_idx[i]);
        }
        history_joint_pos -= dof_default_eigen_policy;

        VecXf frame(history_vae_frame_dim);
        frame << base_lin_vel,
                 ro.base_omega,
                 projected_gravity,
                 current_yaw,
                 command,
                 history_joint_pos,
                 history_joint_vel,
                 last_action_eigen;
        return frame;
    }

    VecXf BuildHistoryVaeHistoryObservation(const RobotBasicState &ro, const Vec3f& command) {
        VecXf frame = BuildHistoryVaeFrame(ro, command);
        history_frames_.push_back(frame);
        while (history_frames_.size() > history_vae_history_length) {
            history_frames_.pop_front();
        }

        VecXf history = VecXf::Zero(history_vae_history_dim);
        int missing_frames = history_vae_history_length - static_cast<int>(history_frames_.size());
        for (int i = 0; i < history_vae_history_length; ++i) {
            const VecXf& src = (i < missing_frames) ? history_frames_.front() : history_frames_[i - missing_frames];
            history.segment(i * history_vae_frame_dim, history_vae_frame_dim) = src;
        }
        return history;
    }

    RobotAction getRobotAction(const RobotBasicState &ro, const UserCommand &uc) {

        Vec3f command = Vec3f(uc.forward_vel_scale, uc.side_vel_scale, uc.turnning_vel_scale);

        // ── Height-dependent default pose ──
        if (uc.height_command > 0.01f) {
            UpdateDefaultPoseForHeight(uc.height_command);
        }

        std::cout << "\r[CMD] fwd=" << command(0) << " side=" << command(1)
                  << " yaw=" << command(2) << " h=" << current_height_ << std::flush;

        if (uses_history_vae_) {
            current_observation_ = BuildHistoryVaePolicyObservation(ro, command);
            VecXf history_observation = BuildHistoryVaeHistoryObservation(ro, command);
            current_action_eigen = Onnx_infer(current_observation_, &history_observation);
        } else {
            current_observation_ = BuildLegacyObservation(ro, command);
            current_action_eigen = Onnx_infer(current_observation_);
        }
        last_action_eigen = current_action_eigen;

        
        for (int i = 0; i < action_dim; ++i){
            tmp_action_eigen(i) = current_action_eigen(policy2robot_idx[i]);
            tmp_action_eigen(i) *= action_scale_robot[i];
        }
        tmp_action_eigen += dof_default_eigen_robot;
        
        for (int i = 0; i < 4; ++i){
            robot_action.goal_joint_pos.segment(i*4, 3) = tmp_action_eigen.segment(i*4, 3);
            robot_action.goal_joint_vel(i*4+3) = tmp_action_eigen(i*4+3);
        }

        // zero wheel velocity when no command (before q or after s)
        bool cmd_is_zero = (uc.forward_vel_scale == 0.0f &&
                            uc.side_vel_scale    == 0.0f &&
                            uc.turnning_vel_scale == 0.0f);
        if (cmd_is_zero) {
            for (int i = 0; i < 4; ++i)
                robot_action.goal_joint_vel(i*4+3) = 0.0f;
        }

        
        ++run_cnt_;
        ++time_step;
        return robot_action;
    }

    void setDefaultJointPos(const VecXf& pos){
        dof_pos_default_.setZero(motor_num); 
        for(int i=0;i<motor_num;++i) {
            dof_pos_default_(i) = pos(i);
        }
    }

    double getCurrentTime() {
        clock_gettime(1, &system_time);
        return system_time.tv_sec + system_time.tv_nsec / 1e9;
    }
};
