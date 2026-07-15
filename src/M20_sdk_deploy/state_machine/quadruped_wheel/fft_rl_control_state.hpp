/**
 * @file fft_rl_control_state.hpp
 * @brief RL control state for FFT (Fast Fine-Tuning) policy — single ONNX
 * @author DeepRobotics
 * @version 1.0
 * @date 2025-11-07
 *
 * @copyright Copyright (c) 2025  DeepRobotics
 *
 * Differences from RLControlState:
 *  - Uses M20FftPolicyRunner (single ONNX model, fixed URDF defaults, 58-dim obs)
 *  - Velocity command is [6:9]; height is the final observation at index 57
 *  - No speed-mode / direction switching (single FFT policy handles all)
 *  - Height uses UserCommand::height_command, not reserved_scale
 */

#pragma once
#include <cstdlib>
#include "state_base.h"
#include "policy_runner_base.hpp"
#include "m20_fft_policy_runner.hpp"
#include "robot_interface.h"
#include "user_command_interface.h"
#include "json.hpp"
#include "basic_function.hpp"

namespace qw {
    class FftRLControlState : public StateBase {
    private:
        RobotBasicState rbs_;
        int state_run_cnt_;

        std::shared_ptr<PolicyRunnerBase> policy_ptr_;
        std::shared_ptr<M20FftPolicyRunner> fft_policy_;

        std::thread run_policy_thread_;
        bool start_flag_ = true;

        float policy_cost_time_ = 1;

        void init_rbs_() {
            rbs_.flt_base_acc_mat = Eigen::MatrixXf::Zero(20, 3);
        }

        void UpdateRobotObservation() {
            rbs_.base_rpy = ri_ptr_->GetImuRpy();
            rbs_.base_rot_mat = RpyToRm(rbs_.base_rpy);
            rbs_.base_lin_vel = ri_ptr_->GetBaseLinearVelocity();
            rbs_.base_omega = ri_ptr_->GetImuOmega();
            rbs_.base_acc = ri_ptr_->GetImuAcc();
            rbs_.joint_pos = ri_ptr_->GetJointPosition();
            rbs_.joint_vel = ri_ptr_->GetJointVelocity();
            rbs_.joint_tau = ri_ptr_->GetJointTorque();
        }

        void PolicyRunner() {
            int run_cnt_record = -1;
            std::cerr << "[FFT] PolicyRunner thread started" << std::endl;
            while (start_flag_) {
                try {
                if (state_run_cnt_ % policy_ptr_->decimation_ == 0 && state_run_cnt_ != run_cnt_record) {
                    timespec start_timestamp, end_timestamp;
                    clock_gettime(CLOCK_MONOTONIC, &start_timestamp);
                    auto ra = policy_ptr_->getRobotAction(rbs_, *(uc_ptr_->GetUserCommand()));

                    MatXf res = ra.ConvertToMat();
                    ri_ptr_->SetJointCommand(res);
                    run_cnt_record = state_run_cnt_;
                    clock_gettime(CLOCK_MONOTONIC, &end_timestamp);
                    policy_cost_time_ = (end_timestamp.tv_sec - start_timestamp.tv_sec) * 1e3
                                        + (end_timestamp.tv_nsec - start_timestamp.tv_nsec) / 1e6;
                }
                } catch (const std::exception& e) {
                    std::cerr << "[FFT] PolicyRunner exception: " << e.what() << std::endl;
                } catch (...) {
                    std::cerr << "[FFT] PolicyRunner unknown exception" << std::endl;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
            std::cerr << "[FFT] PolicyRunner thread exiting" << std::endl;
        }

    public:
        FftRLControlState(const RobotName &robot_name, const std::string &state_name,
                          std::shared_ptr<ControllerData> data_ptr)
                : StateBase(robot_name, state_name, data_ptr) {
            std::memset(&rbs_, 0, sizeof(rbs_));
            if (robot_name_ == RobotName::M20) {
                namespace fs = std::filesystem;

                fs::path exe_dir = fs::canonical("/proc/self/exe").parent_path();
                fs::path policy_base = exe_dir / "policy";

                // Fallback to source-relative path
                if (!fs::exists(policy_base)) {
                    fs::path src_base = fs::path(__FILE__).parent_path() / ".." / ".." / "policy";
                    if (fs::exists(src_base)) {
                        policy_base = fs::canonical(src_base);
                    } else {
                        std::cerr << "[FFT ERROR] Cannot find policy directory at:\n"
                                  << "  " << (exe_dir / "policy").string() << "\n"
                                  << "  " << fs::weakly_canonical(src_base).string() << "\n";
                        exit(1);
                    }
                }

                // Policy subdir: env var POLICY_SUBDIR overrides default fixed-height FFT policy.
                const char* env_subdir = std::getenv("POLICY_SUBDIR");
                std::string subdir = env_subdir ? env_subdir : "fixed_height_fft";
                auto fft_policy_path = policy_base / subdir / "policy.onnx";

                if (!fs::exists(fft_policy_path)) {
                    std::cerr << "[FFT ERROR] Policy not found: "
                              << fft_policy_path.string() << "\n";
                    exit(1);
                }

                fft_policy_ = std::make_shared<M20FftPolicyRunner>(
                    "fft_policy", fft_policy_path.string());
            }

            policy_ptr_ = fft_policy_;
            if (!policy_ptr_) {
                std::cerr << "[FFT ERROR] Failed to create FFT policy runner" << std::endl;
                exit(0);
            }
            policy_ptr_->DisplayPolicyInfo();
            init_rbs_();
        }

        ~FftRLControlState() {}

        virtual void OnEnter() {
            std::cerr << "[FFT] OnEnter RLControl START" << std::endl;
            state_run_cnt_ = -1;
            start_flag_ = true;
            auto* cmd = uc_ptr_->GetUserCommand();
            if (!cmd) {
                std::cerr << "[FFT] FATAL: GetUserCommand returned null" << std::endl;
                exit(1);
            }
            cmd->forward_vel_scale = 0.0f;
            cmd->side_vel_scale = 0.0f;
            cmd->turnning_vel_scale = 0.0f;
            cmd->reserved_scale = 0.40f;  // nominal height
            std::cerr << "[FFT] OnEnter starting policy thread..." << std::endl;
            try {
                run_policy_thread_ = std::thread(std::bind(&FftRLControlState::PolicyRunner, this));
            } catch (const std::exception& e) {
                std::cerr << "[FFT] Exception starting policy thread: " << e.what() << std::endl;
                exit(1);
            }
            std::cerr << "[FFT] OnEnter calling policy_ptr_->OnEnter..." << std::endl;
            policy_ptr_->OnEnter(rbs_);
            StateBase::msfb_.UpdateCurrentState(RobotMotionState::RLControlMode);
            std::cerr << "[FFT] OnEnter RLControl DONE" << std::endl;
        };

        virtual void OnExit() {
            start_flag_ = false;
            run_policy_thread_.join();
            state_run_cnt_ = -1;
        }

        virtual void Run() {
            UpdateRobotObservation();

            if (state_run_cnt_ % 50 == 0) {
                float h = fft_policy_->GetHeightTarget();
                std::cout << "[FFT] omega=" << rbs_.base_omega.transpose()
                          << " height=" << h << "\n";
            }
            state_run_cnt_++;
        }

        virtual bool LoseControlJudge() {
            if (uc_ptr_->GetUserCommand()->target_mode == uint8_t(RobotMotionState::JointDamping))
                return true;
            return PostureUnsafeCheck();
        }

        bool PostureUnsafeCheck() {
            return false;
        }

        virtual StateName GetNextStateName() {
            if (uc_ptr_->GetUserCommand()->safe_control_mode != 0)
                return StateName::kJointDamping;
            return StateName::kRLControl;
        }
    };
};
