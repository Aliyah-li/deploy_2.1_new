/**
 * @file m20_fft_policy_runner.hpp
 * @brief FFT policy runner — single ONNX model, fixed training defaults, 58-dim obs
 * @author DeepRobotics
 * @version 1.0
 * @date 2025-11-07
 *
 * @copyright Copyright (c) 2025  DeepRobotics
 *
 * Differences from M20PolicyRunner:
 *  - Single ONNX model (not base + residual)
 *  - Commands: vx, vy, heading at indices 6:9; height is appended at index 57
 *  - 58-dim observation (3+3+3+16+16+16+1)
 *  - Height command is passed to the policy; action offsets use the fixed URDF default
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

class M20FftPolicyRunner : public PolicyRunnerBase {
private:
    // ── IK geometry constants ──
    static constexpr float thigh_len_  = 0.25f;
    static constexpr float shank_len_  = 0.25f;
    static constexpr float hipy_limit_ = 2.443f;
    static constexpr float knee_limit_ = 2.758f;

    // ── Height command range ──
    static constexpr float height_min_     = 0.32f;
    static constexpr float height_max_     = 0.425f;
    static constexpr float height_nominal_ = 0.40f;

    VecXf kp_, kd_;
    VecXf dof_default_eigen_policy, dof_default_eigen_robot;
    Vec3f max_cmd_vel_, gravity_direction = Vec3f(0., 0., -1.);
    VecXf dof_pos_default_;
    timespec system_time;

    const int motor_num = 16;
    int observation_dim_ = 58;  // detected from ONNX: fixed self-FFT is 57
    const int action_dim = 16;

    VecXf joint_pos_rl = VecXf(action_dim);
    VecXf joint_vel_rl = VecXf(action_dim);

    const std::string policy_path_;

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

    // ── History VAE config ──
    static constexpr int history_vae_frame_dim = 61;       // base_lin_vel(3) + ang_vel(3) + gravity(3) + yaw(1) + vel_cmd(3) + joint_pos(16) + joint_vel(16) + action(16)
    static constexpr int history_vae_history_length = 25;  // frames in buffer
    static constexpr int history_vae_history_dim = history_vae_frame_dim * history_vae_history_length;  // 1525

    bool uses_history_vae_ = false;
    int history_observation_dim = 0;
    std::deque<VecXf> history_frames_;

    const char* input_names_[1] = {"obs"};
    const char* history_input_names_[2] = {"obs", "obs_history"};
    const char* output_names_[1] = {"actions"};
    Ort::MemoryInfo memory_info{nullptr};
    std::vector<int64_t> input_observationShape = {1, 58};
    std::vector<int64_t> input_historyShape = {1, history_vae_history_dim};

    float time_step = 0.;
    float current_height_target_ = height_nominal_;

public:
    // ── IK helpers ──
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

    static VecXf BuildDefaultPoseAtHeight(float h) {
        float hipy = GetHipYPosByHeight(h);
        float knee = GetKneePosByHeight(h);
        VecXf pose(16);
        pose << 0.0f,  hipy,  knee, 0.0f,
               0.0f,  hipy,  knee, 0.0f,
               0.0f, -hipy, -knee, 0.0f,
               0.0f, -hipy, -knee, 0.0f;
        return pose;
    }

    static VecXf BuildDefaultPosePolicy(float h) {
        float hipy = GetHipYPosByHeight(h);
        float knee = GetKneePosByHeight(h);
        VecXf pose(16);
        pose << 0.0f,  hipy,  knee,
               0.0f,  hipy,  knee,
               0.0f, -hipy, -knee,
               0.0f, -hipy, -knee,
               0.0f, 0.0f, 0.0f, 0.0f;
        return pose;
    }

    // ── Constructor ──
    M20FftPolicyRunner(const std::string &policy_name,
                       const std::string &policy_path)
            : PolicyRunnerBase(policy_name),
              policy_path_(policy_path),
              env_(ORT_LOGGING_LEVEL_WARNING, "M20FftPolicyRunner"),
              session_options_{},
              session_{nullptr},
              memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {

        dof_default_eigen_policy.setZero(action_dim);
        dof_default_eigen_robot.setZero(action_dim);
        // Training JointPositionActionCfg uses use_default_offset=True: the
        // reference is the fixed asset default, not the height-command IK.
        dof_default_eigen_policy << 0.0f, -0.6f,  1.0f,
                                    0.0f, -0.6f,  1.0f,
                                    0.0f,  0.6f, -1.0f,
                                    0.0f,  0.6f, -1.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f;
        // robot order: fl(hipx,hipy,knee,wheel), fr, hl, hr
        dof_default_eigen_robot << 0.0f, -0.6f,  1.0f, 0.0f,
                                   0.0f, -0.6f,  1.0f, 0.0f,
                                   0.0f,  0.6f, -1.0f, 0.0f,
                                   0.0f,  0.6f, -1.0f, 0.0f;

        SetDecimation(20);

        if (access(policy_path_.c_str(), F_OK) != 0) {
            std::cerr << "[FFT] Policy file not found: " << policy_path_ << std::endl;
            throw std::runtime_error("FFT policy file missing");
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

        std::cout << "[FFT] Policy: " << policy_path_
                  << " | Obs dim: " << observation_dim_ << std::endl;
    }

    ~M20FftPolicyRunner() override = default;

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
        current_height_target_ = height_nominal_;
        history_frames_.clear();
    }

    // ── Detect model input count and configure dims ──
    void ConfigureOnnxInputs() {
        size_t input_count = session_.GetInputCount();
        uses_history_vae_ = input_count >= 2;
        history_observation_dim = uses_history_vae_ ? history_vae_history_dim : 0;

        // Read actual observation dim from model (may differ from hardcoded)
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
                        throw std::runtime_error("FFT obs_history has a dynamic non-batch dimension");
                    history_observation_dim *= static_cast<int>(input_historyShape[i]);
                }
            }
        }
    }

    Vec3f EstimateBaseLinearVelocity(const RobotBasicState &ro) const {
        return ro.base_lin_vel;
    }

    // ── Build one 61-dim history frame from robot state ──
    VecXf BuildHistoryVaeFrame(const RobotBasicState &ro, const Vec3f& vel_command) {
        Vec3f base_lin_vel = EstimateBaseLinearVelocity(ro);
        Vec3f projected_gravity = ro.base_rot_mat.inverse() * gravity_direction;
        float current_yaw = ro.base_rpy(2);

        VecXf history_joint_pos(action_dim);
        VecXf history_joint_vel(action_dim);
        for (int i = 0; i < action_dim; ++i) {
            history_joint_pos(i) = ro.joint_pos(robot2policy_idx[i]);
            history_joint_vel(i) = ro.joint_vel(robot2policy_idx[i]);
        }
        history_joint_pos -= dof_default_eigen_policy;

        VecXf frame(history_vae_frame_dim);
        frame << base_lin_vel,
                 ro.base_omega,
                 projected_gravity,
                 current_yaw,
                 vel_command,
                 history_joint_pos,
                 history_joint_vel,
                 last_action_eigen;
        return frame;
    }

    // ── Maintain rolling history buffer, return full 1525-dim tensor ──
    // Training layout (IsaacLab flatten_history_dim=True): TERM-MAJOR
    // [lin_vel(3)×25 | ang_vel(3)×25 | gravity(3)×25 | yaw(1)×25 | vel_cmd(3)×25 | joint_pos(16)×25 | joint_vel(16)×25 | action(16)×25]
    VecXf BuildHistoryVaeHistoryObservation(const RobotBasicState &ro, const Vec3f& vel_command) {
        VecXf frame = BuildHistoryVaeFrame(ro, vel_command);
        history_frames_.push_back(frame);
        while (static_cast<int>(history_frames_.size()) > history_vae_history_length) {
            history_frames_.pop_front();
        }

        int n = static_cast<int>(history_frames_.size());
        int missing = history_vae_history_length - n;
        const VecXf& pad = history_frames_.front();

        VecXf history = VecXf::Zero(history_vae_history_dim);

        // Self-supervised TCN export expects contiguous [batch, time, feature].
        if (input_historyShape.size() == 3) {
            for (int f = 0; f < history_vae_history_length; ++f) {
                const VecXf& src = (f < missing) ? pad : history_frames_[f - missing];
                history.segment(f * history_vae_frame_dim, history_vae_frame_dim) = src;
            }
            if (history_observation_dim != history.size())
                throw std::runtime_error("FFT obs_history dimension does not match 25x61 fixed-height history");
            return history;
        }

        auto write3 = [&](int i0, int& off) {
            for (int f = 0; f < history_vae_history_length; ++f) {
                const VecXf& src = (f < missing) ? pad : history_frames_[f - missing];
                history.segment(off + f*3, 3) = src.segment(i0, 3);
            }
            off += 75;
        };
        auto write1 = [&](int i0, int& off) {
            for (int f = 0; f < history_vae_history_length; ++f) {
                const VecXf& src = (f < missing) ? pad : history_frames_[f - missing];
                history(off + f) = src(i0);
            }
            off += 25;
        };
        auto write16 = [&](int i0, int& off) {
            for (int f = 0; f < history_vae_history_length; ++f) {
                const VecXf& src = (f < missing) ? pad : history_frames_[f - missing];
                history.segment(off + f*16, 16) = src.segment(i0, 16);
            }
            off += 400;
        };

        int off = 0;
        write3(0, off);   // base_lin_vel   3×25=75
        write3(3, off);   // base_ang_vel   3×25=75
        write3(6, off);   // gravity        3×25=75
        write1(9, off);   // yaw            1×25=25
        write3(10, off);  // vel_cmd        3×25=75
        write16(13, off); // joint_pos     16×25=400
        write16(29, off); // joint_vel     16×25=400
        write16(45, off); // last_action   16×25=400

        return history;
    }

    // ── Dual-mode inference (legacy 1-input or history-VAE 2-input) ──
    VecXf OnnxInfer(const VecXf& obs, const VecXf* history_observation = nullptr) {
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, const_cast<float*>(obs.data()), obs.size(),
            input_observationShape.data(), input_observationShape.size());

        std::vector<Ort::Value> inputs;
        inputs.emplace_back(std::move(input_tensor));
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
                input_historyShape.size());
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

    void UpdateDefaultPoseForHeight(float h) {
        h = std::max(height_min_, std::min(height_max_, h));
        // Height is a policy command. The policy itself learns the required
        // joint offsets around the fixed URDF default used during training.
        current_height_target_ = h;
    }

    VecXf BuildObservation(const RobotBasicState &ro, const Vec4f& cmd) {
        Vec3f base_omega_scaled = ro.base_omega * omega_scale_;
        Vec3f proj_gravity = ro.base_rot_mat.inverse() * gravity_direction;

        for (int i = 0; i < action_dim; ++i) {
            joint_pos_rl(i) = ro.joint_pos(robot2policy_idx[i]);
            joint_vel_rl(i) = ro.joint_vel(robot2policy_idx[i]) * dof_vel_scale_;
        }
        joint_pos_rl -= dof_default_eigen_policy;
        joint_pos_rl.segment(12, 4).setZero();  // training zeros wheel positions

        // ObservationManager preserves term insertion order. The height term
        // is added in StrongPpoBaseEnvCfg.__post_init__, so it is LAST.
        VecXf obs(observation_dim_);
        obs.head(57) << base_omega_scaled, // [0:3]
                        proj_gravity,      // [3:6]
                        cmd.head<3>(),     // [6:9]
                        joint_pos_rl,      // [9:25]
                        joint_vel_rl,      // [25:41]
                        last_action_eigen; // [41:57]
        if (observation_dim_ == 58) {
            obs(57) = cmd(3);
        } else if (observation_dim_ != 57) {
            throw std::runtime_error("FFT policy obs must be 57 (fixed) or 58 (legacy height)");
        }
        return obs;
    }

    RobotAction getRobotAction(const RobotBasicState &ro, const UserCommand &uc) override {

        float vx      = uc.forward_vel_scale;
        float vy      = uc.side_vel_scale;
        float heading = uc.turnning_vel_scale;
        float height  = uc.height_command;

        if (height < 0.01f) height = height_nominal_;
        height = std::max(height_min_, std::min(height_max_, height));
        UpdateDefaultPoseForHeight(height);

        Vec4f cmd(vx, vy, heading, height);

        std::cout << "\r[FFT CMD] vx=" << vx << " vy=" << vy
                  << " yaw=" << heading << " h=" << height << std::flush;

        current_observation_ = BuildObservation(ro, cmd);

        if (uses_history_vae_) {
            Vec3f vel_cmd(vx, vy, heading);
            VecXf history_obs = BuildHistoryVaeHistoryObservation(ro, vel_cmd);
            current_action_eigen = OnnxInfer(current_observation_, &history_obs);
        } else {
            current_action_eigen = OnnxInfer(current_observation_);
        }


        // Smooth blend-in over first 5s
        if (run_cnt_ < 250) {
            current_action_eigen *= run_cnt_ / 250.0f;
        }

        // MuJoCo wheel direction opposite to IsaacLab (confirmed by +30 test)
        // current_action_eigen[12] = -current_action_eigen[12];
        // current_action_eigen[13] = -current_action_eigen[13];
        // current_action_eigen[14] = -current_action_eigen[14];
        // current_action_eigen[15] = -current_action_eigen[15];

        last_action_eigen = current_action_eigen;

        for (int i = 0; i < action_dim; ++i) {
            tmp_action_eigen(i) = current_action_eigen(policy2robot_idx[i]);
            tmp_action_eigen(i) *= action_scale_robot[i];
        }
        tmp_action_eigen += dof_default_eigen_robot;

        for (int i = 0; i < 4; ++i) {
            robot_action.goal_joint_pos.segment(i*4, 3) = tmp_action_eigen.segment(i*4, 3);
            robot_action.goal_joint_vel(i*4+3) = tmp_action_eigen(i*4+3);
        }

        // DEBUG after all processing
        if (run_cnt_ < 3) {
            std::cerr << "\n[DEBUG frame " << run_cnt_ << "]"
                      << "\n  obs[0:6]=" << current_observation_.head(6).transpose()
                      << "\n  obs[9:21]=" << current_observation_.segment(9,12).transpose()
                      << "\n  raw_act=" << current_action_eigen.transpose()
                      << "\n  scaled_act=" << tmp_action_eigen.transpose()
                      << "\n  goal_pos=" << robot_action.goal_joint_pos.transpose()
                      << "\n  default_pol[1:3]=" << dof_default_eigen_policy.segment(1,2).transpose()
                      << std::endl;
        }

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

    void setDefaultJointPos(const VecXf& pos) {
        dof_pos_default_.setZero(motor_num);
        for (int i = 0; i < motor_num; ++i) dof_pos_default_(i) = pos(i);
    }

    double getCurrentTime() {
        clock_gettime(1, &system_time);
        return system_time.tv_sec + system_time.tv_nsec / 1e9;
    }

    void SetHeightTarget(float h) { UpdateDefaultPoseForHeight(h); }
    float GetHeightTarget() const { return current_height_target_; }
};
