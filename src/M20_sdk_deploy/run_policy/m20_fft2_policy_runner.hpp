/**
 * @file m20_fft2_policy_runner.hpp
 * @brief FFT2 policy runner — single ONNX, variable-height command, 58-dim obs
 *
 * Height is an observation consumed by the policy. Joint observations and
 * actions are residuals around the live M20 IK pose, matching 2.1 height tasks.
 *
 * Config reference: sim2sim_deploy_config.md and
 * fft_simulation/sim2sim_deploy_config.md
 * Observation (58 dims):
 *   [0:3]   base_ang_vel * 0.25
 *   [3:6]   projected_gravity
 *   [6:9]   velocity commands: vx, vy, heading/yaw command
 *   [9:25]  joint_pos_rel (robot_pos - live height IK pose)
 *   [25:41] joint_vel * 0.05
 *   [41:57] last_action
 *   [57]    base_height_command (term appended in env __post_init__)
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
#include <deque>

class M20Fft2PolicyRunner : public PolicyRunnerBase {
private:
    // ── IK geometry ──
    static constexpr float thigh_len_  = 0.25f;
    static constexpr float shank_len_  = 0.25f;
    static constexpr float hipy_limit_ = 2.443f;
    static constexpr float knee_limit_ = 2.758f;

    // ── Height range ──
    static constexpr float height_min_     = 0.32f;
    static constexpr float height_max_     = 0.425f;

    // Initial height command matches stand_height_=0.38 in control params.
    static constexpr float height_initial_  = 0.38f;
    static constexpr float height_nominal_  = 0.40f;  // only as fallback
    static constexpr float height_slew_rate_ = 0.04f; // m/s, same as training
    static constexpr float policy_dt_ = 0.02f;        // 50 Hz

    VecXf kp_, kd_;
    VecXf dof_default_eigen_policy, dof_default_eigen_robot;
    Vec3f max_cmd_vel_, gravity_direction = Vec3f(0., 0., -1.);
    timespec system_time;

    const int motor_num = 16;
    const int action_dim = 16;
    int observation_dim_ = 58;  // auto-detected from ONNX; fallback 58

    VecXf joint_pos_rl = VecXf(action_dim);
    VecXf joint_vel_rl = VecXf(action_dim);

    const std::string policy_path_;
    const bool height_residual_mode_;

    float omega_scale_ = 0.25;
    float dof_vel_scale_ = 0.05;
    VecXf current_action_eigen, last_action_eigen, current_observation_,
          projected_gravity, tmp_action_eigen;

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

    // ── TCN/self-supervised history: 61 legacy fields + height command ──
    static constexpr int history_length_ = 25;
    static constexpr int height_history_frame_dim_ = 62;

    bool uses_history_vae_ = false;
    int history_observation_dim = 0;
    std::deque<VecXf> history_frames_;

    const char* input_names_[1] = {"obs"};
    const char* history_input_names_[2] = {"obs", "obs_history"};
    const char* output_names_[1] = {"actions"};
    Ort::MemoryInfo memory_info{nullptr};
    std::vector<int64_t> input_observationShape = {1, 58};
    std::vector<int64_t> input_historyShape = {1, history_length_, height_history_frame_dim_};

    float time_step = 0.;
    float current_height_target_ = height_initial_;  // ★ start at standup height

public:
    // ═══════════════ IK helpers ═══════════════
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

    static VecXf BuildIkPosePolicy(float h) {
        float hipy = GetHipYPosByHeight(h), knee = GetKneePosByHeight(h);
        VecXf pose(16);
        pose << 0.0f, hipy, knee, 0.0f, hipy, knee,
                0.0f, -hipy, -knee, 0.0f, -hipy, -knee,
                0.0f, 0.0f, 0.0f, 0.0f;
        return pose;
    }

    static VecXf BuildIkPoseRobot(float h) {
        float hipy = GetHipYPosByHeight(h), knee = GetKneePosByHeight(h);
        VecXf pose(16);
        pose << 0.0f, hipy, knee, 0.0f,
                0.0f, hipy, knee, 0.0f,
                0.0f, -hipy, -knee, 0.0f,
                0.0f, -hipy, -knee, 0.0f;
        return pose;
    }

    // ── Constructor ──
    M20Fft2PolicyRunner(const std::string &policy_name,
                       const std::string &policy_path,
                       bool height_residual_mode = true)
            : PolicyRunnerBase(policy_name),
              policy_path_(policy_path),
              height_residual_mode_(height_residual_mode),
              env_(ORT_LOGGING_LEVEL_WARNING, "M20Fft2PolicyRunner"),
              session_options_{},
              session_{nullptr},
              memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {

        dof_default_eigen_policy.setZero(action_dim);
        dof_default_eigen_robot.setZero(action_dim);
        // Fixed asset defaults used by training's use_default_offset=True.
        dof_default_eigen_policy << 0.0f, -0.6f,  1.0f,
                                    0.0f, -0.6f,  1.0f,
                                    0.0f,  0.6f, -1.0f,
                                    0.0f,  0.6f, -1.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f;
        // robot order: interleaved per leg
        dof_default_eigen_robot  << 0.0f, -0.6f,  1.0f, 0.0f,
                                    0.0f, -0.6f,  1.0f, 0.0f,
                                    0.0f,  0.6f, -1.0f, 0.0f,
                                    0.0f,  0.6f, -1.0f, 0.0f;

        SetDecimation(20);

        if (access(policy_path_.c_str(), F_OK) != 0) {
            std::cerr << "[FFT2] Policy file not found: " << policy_path_ << std::endl;
            throw std::runtime_error("FFT2 policy file missing");
        }
        session_options_.SetIntraOpNumThreads(4);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
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

        current_observation_.setZero(observation_dim_);
        last_action_eigen.setZero(action_dim);
        tmp_action_eigen.setZero(action_dim);
        current_action_eigen.setZero(action_dim);

        memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        std::cout << "[FFT2] Policy: " << policy_path_
                  << " | Obs dim: " << observation_dim_
                  << " | Init height command: " << current_height_target_
                  << " | Action reference: "
                  << (height_residual_mode_ ? "live height IK" : "fixed URDF") << std::endl;
    }

    ~M20Fft2PolicyRunner() override = default;

    std::vector<int> generate_permutation(
        const std::vector<std::string>& from,
        const std::vector<std::string>& to,
        int default_index = 0)
    {
        std::unordered_map<std::string, int> idx_map;
        for (int i = 0; i < from.size(); ++i) idx_map[from[i]] = i;
        std::vector<int> perm;
        for (const auto& name : to) {
            auto it = idx_map.find(name);
            perm.push_back(it != idx_map.end() ? it->second : default_index);
        }
        return perm;
    }

    void DisplayPolicyInfo() override {}

    void OnEnter(const RobotBasicState &rbs) override {
        run_cnt_ = 0;
        cmd_vel_input_.setZero();
        last_action_eigen.setZero(action_dim);
        tmp_action_eigen.setZero(action_dim);
        // ★ keep current_height_target_ as-is (standup height), don't reset to nominal
        history_frames_.clear();
    }

    void ConfigureOnnxInputs() {
        size_t input_count = session_.GetInputCount();
        uses_history_vae_ = input_count >= 2;
        history_observation_dim = 0;

        auto obs_shape = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (obs_shape.size() == 2 && obs_shape[1] > 0) {
            observation_dim_ = static_cast<int>(obs_shape[1]);
            input_observationShape = {1, static_cast<int64_t>(obs_shape[1])};
        }

        if (uses_history_vae_) {
            auto history_shape = session_.GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
            if (history_shape.size() >= 2) {
                input_historyShape = history_shape;
                input_historyShape[0] = 1;
                history_observation_dim = 1;
                for (size_t i = 1; i < input_historyShape.size(); ++i) {
                    if (input_historyShape[i] <= 0)
                        throw std::runtime_error("FFT2 obs_history has a dynamic non-batch dimension");
                    history_observation_dim *= static_cast<int>(input_historyShape[i]);
                }
            }
        }
    }

    Vec3f EstimateBaseLinearVelocity(const RobotBasicState &ro) const {
        return ro.base_lin_vel;
    }

    // ═══════════════ History VAE ═══════════════
    VecXf BuildHistoryFrame(const RobotBasicState &ro, const Vec3f& vel_command, float height) {
        Vec3f base_lin_vel = EstimateBaseLinearVelocity(ro);
        Vec3f proj_gravity = ro.base_rot_mat.inverse() * gravity_direction;
        float current_yaw = ro.base_rpy(2);

        VecXf history_joint_pos(action_dim);
        VecXf history_joint_vel(action_dim);
        for (int i = 0; i < action_dim; ++i) {
            history_joint_pos(i) = ro.joint_pos(robot2policy_idx[i]);
            history_joint_vel(i) = ro.joint_vel(robot2policy_idx[i]);
        }
        history_joint_pos -= height_residual_mode_ ? BuildIkPosePolicy(height) : dof_default_eigen_policy;
        history_joint_pos.segment(12, 4).setZero();

        VecXf frame(height_history_frame_dim_);
        frame << base_lin_vel,
                 ro.base_omega,
                 proj_gravity,
                 current_yaw,
                 vel_command,
                 history_joint_pos,
                 history_joint_vel,
                 last_action_eigen,
                 height;
        return frame;
    }

    VecXf BuildHistoryObservation(const RobotBasicState &ro, const Vec3f& vel_command, float height) {
        VecXf frame = BuildHistoryFrame(ro, vel_command, height);
        history_frames_.push_back(frame);
        while (static_cast<int>(history_frames_.size()) > history_length_)
            history_frames_.pop_front();

        int n = static_cast<int>(history_frames_.size());
        int missing = history_length_ - n;
        const VecXf& pad = history_frames_.front();
        VecXf history = VecXf::Zero(history_length_ * height_history_frame_dim_);
        for (int f = 0; f < history_length_; ++f) {
            const VecXf& src = (f < missing) ? pad : history_frames_[f - missing];
            history.segment(f * height_history_frame_dim_, height_history_frame_dim_) = src;
        }
        if (history_observation_dim != history.size())
            throw std::runtime_error("FFT2 obs_history dimension does not match 25x62 height history");
        return history;
    }

    // ═══════════════ ONNX Inference ═══════════════
    VecXf OnnxInfer(const VecXf& obs, const VecXf* history_observation = nullptr) {
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, const_cast<float*>(obs.data()), obs.size(),
            input_observationShape.data(), input_observationShape.size());

        std::vector<Ort::Value> inputs;
        inputs.emplace_back(std::move(input_tensor));
        const char* const* run_input_names = input_names_;
        size_t run_input_count = 1;

        if (uses_history_vae_) {
            if (history_observation == nullptr)
                throw std::runtime_error("History-VAE policy requires obs_history input");
            Ort::Value history_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                const_cast<float*>(history_observation->data()),
                history_observation->size(),
                input_historyShape.data(), input_historyShape.size());
            inputs.emplace_back(std::move(history_tensor));
            run_input_names = history_input_names_;
            run_input_count = 2;
        }

        auto outputs = session_.Run(Ort::RunOptions{nullptr}, run_input_names,
                                    inputs.data(), run_input_count, output_names_, 1);
        float* action_data = outputs[0].GetTensorMutableData<float>();
        Eigen::Map<Eigen::VectorXf> action_map(action_data, action_dim);
        return VecXf(action_map);
    }

    void UpdateHeightCommandTowards(float requested_height) {
        requested_height = std::max(height_min_, std::min(height_max_, requested_height));
        const float max_step = height_slew_rate_ * policy_dt_;
        const float error = requested_height - current_height_target_;
        current_height_target_ += std::max(-max_step, std::min(max_step, error));
    }

    VecXf BuildObservation(const RobotBasicState &ro, const Vec4f& cmd) {
        Vec3f base_omega_scaled = ro.base_omega * omega_scale_;
        Vec3f proj_gravity = ro.base_rot_mat.inverse() * gravity_direction;

        for (int i = 0; i < action_dim; ++i) {
            joint_pos_rl(i) = ro.joint_pos(robot2policy_idx[i]);
            joint_vel_rl(i) = ro.joint_vel(robot2policy_idx[i]) * dof_vel_scale_;
        }
        // Height-residual training observes joints relative to the live IK pose.
        joint_pos_rl -= height_residual_mode_ ? BuildIkPosePolicy(cmd(3)) : dof_default_eigen_policy;
        joint_pos_rl.segment(12, 4).setZero();  // training zeros wheel positions

        // 58-dim: 3+3+3+16+16+16+1. Height is the final observation
        // because its term is appended in the training env __post_init__.
        VecXf obs(observation_dim_);
        obs << base_omega_scaled,      // [0:3]
               proj_gravity,           // [3:6]
               cmd.head<3>(),          // [6:9] vx, vy, heading/yaw command
               joint_pos_rl,           // [9:25]
               joint_vel_rl,           // [25:41]
               last_action_eigen,      // [41:57]
               cmd(3);                 // [57] height command
        return obs;
    }

    RobotAction getRobotAction(const RobotBasicState &ro, const UserCommand &uc) override {

        float vx      = uc.forward_vel_scale;
        float vy      = uc.side_vel_scale;
        float heading = uc.turnning_vel_scale;
        float height  = uc.height_command;

        // ★ 变高度: persist current height if no command; don't jump to nominal
        if (height < 0.01f) height = current_height_target_;
        UpdateHeightCommandTowards(height);
        height = current_height_target_;

        Vec4f cmd(vx, vy, heading, height);

        std::cout << "\r[FFT2 CMD] vx=" << vx << " vy=" << vy
                  << " yaw=" << heading << " h=" << height << std::flush;

        current_observation_ = BuildObservation(ro, cmd);

        if (uses_history_vae_) {
            Vec3f vel_cmd(vx, vy, heading);
            VecXf history_obs = BuildHistoryObservation(ro, vel_cmd, height);
            current_action_eigen = OnnxInfer(current_observation_, &history_obs);
        } else {
            current_action_eigen = OnnxInfer(current_observation_);
        }

        last_action_eigen = current_action_eigen;

        // Match height-residual training: scaled action around live IK reference.
        for (int i = 0; i < action_dim; ++i) {
            tmp_action_eigen(i) = current_action_eigen(policy2robot_idx[i]);
            tmp_action_eigen(i) *= action_scale_robot[i];
        }
        tmp_action_eigen += height_residual_mode_ ? BuildIkPoseRobot(height) : dof_default_eigen_robot;

        for (int i = 0; i < 4; ++i) {
            robot_action.goal_joint_pos.segment(i*4, 3) = tmp_action_eigen.segment(i*4, 3);
            robot_action.goal_joint_vel(i*4+3) = tmp_action_eigen(i*4+3);
        }

        if (run_cnt_ < 3) {
            std::cerr << "\n[FFT2 DEBUG frame " << run_cnt_ << "]"
                      << "\n  obs[0:10]=" << current_observation_.head(10).transpose()
                      << "\n  raw_act=" << current_action_eigen.transpose()
                      << "\n  goal_pos=" << robot_action.goal_joint_pos.transpose()
                      << "\n  height_cmd=" << current_height_target_
                      << " ik_hipy=" << BuildIkPoseRobot(height)(1)
                      << " ik_knee=" << BuildIkPoseRobot(height)(2)
                      << std::endl;
        }

        bool cmd_is_zero = (vx == 0.0f && vy == 0.0f && heading == 0.0f);
        if (cmd_is_zero) {
            for (int i = 0; i < 4; ++i)
                robot_action.goal_joint_vel(i*4+3) = 0.0f;
        }

        ++run_cnt_;
        ++time_step;
        return robot_action;
    }

    void setDefaultJointPos(const VecXf& pos) {
        for (int i = 0; i < motor_num; ++i) {
            // no-op: the training action reference is generated from height IK
        }
    }

    double getCurrentTime() {
        clock_gettime(1, &system_time);
        return system_time.tv_sec + system_time.tv_nsec / 1e9;
    }

    void SetHeightTarget(float h) { UpdateHeightCommandTowards(h); }
    float GetHeightTarget() const { return current_height_target_; }
};
