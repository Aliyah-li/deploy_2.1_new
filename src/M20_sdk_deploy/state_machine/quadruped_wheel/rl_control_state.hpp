/**
 * @file rl_control_state.hpp
 * @brief rl policy runnning state for quadruped-wheel robot
 * @author DeepRobotics
 * @version 1.0
 * @date 2025-11-07
 * 
 * @copyright Copyright (c) 2025  DeepRobotics
 * 
 */
#pragma once
#include "state_base.h"
#include "policy_runner_base.hpp"
#include "m20_policy_runner.hpp"
#include "robot_interface.h"
#include "user_command_interface.h"
#include "hardware/m20_interface.hpp"   // before depth runner: needed for GetDepthImage()
#include "m20_depth_policy_runner.hpp"
#include "json.hpp"
#include "basic_function.hpp"

namespace qw {
    class RLControlState : public StateBase {
    private:
        RobotBasicState rbs_;
        int state_run_cnt_;

        std::shared_ptr<PolicyRunnerBase> policy_ptr_;
        std::shared_ptr<M20PolicyRunner> m20_policy_;
        std::shared_ptr<M20PolicyRunner> m20_policy_low_speed_;
        std::shared_ptr<M20PolicyRunner> m20_policy_high_speed_;
        // High-speed per-direction policies
        std::shared_ptr<M20PolicyRunner> m20_policy_high_x_;
        std::shared_ptr<M20PolicyRunner> m20_policy_high_y_;
        std::shared_ptr<M20PolicyRunner> m20_policy_high_diag_;

        // Depth camera policy runners
        std::shared_ptr<M20DepthPolicyRunner> depth_policy_low_;
        std::shared_ptr<M20DepthPolicyRunner> depth_policy_high_x_;
        std::shared_ptr<M20DepthPolicyRunner> depth_policy_high_y_;
        std::shared_ptr<M20DepthPolicyRunner> depth_policy_high_diag_;
        bool has_depth_policy_ = false;

        std::thread run_policy_thread_;
        bool start_flag_ = true;
        float last_speed_mode_ = -1.f;  // track changes to reserved_scale

        float policy_cost_time_ = 1;

        Eigen::MatrixXf acc_rot = Eigen::MatrixXf::Zero(20, 3);
        int acc_rot_count = 0;

        void init_rbs_() {
            rbs_.flt_base_acc_mat = Eigen::MatrixXf::Zero(20, 3);
        }

        void UpdateRobotObservation() {
            rbs_.base_rpy = ri_ptr_->GetImuRpy();
            rbs_.base_rot_mat = RpyToRm(rbs_.base_rpy);
            rbs_.base_omega = ri_ptr_->GetImuOmega();
            rbs_.base_acc = ri_ptr_->GetImuAcc();
            rbs_.joint_pos = ri_ptr_->GetJointPosition();
            rbs_.joint_vel = ri_ptr_->GetJointVelocity();
            rbs_.joint_tau = ri_ptr_->GetJointTorque();

            // 储存
            rbs_.flt_base_acc_mat.row(acc_rot_count) = rbs_.base_acc.transpose();
            acc_rot_count += 1;
            acc_rot_count = acc_rot_count % 20;
        }

        void PolicyRunner() {
            int run_cnt_record = -1;
            while (start_flag_) {

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
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        }

    public:
        RLControlState(const RobotName &robot_name, const std::string &state_name,
                       std::shared_ptr<ControllerData> data_ptr) : StateBase(robot_name, state_name, data_ptr) {
            std::memset(&rbs_, 0, sizeof(rbs_));
            if (robot_name_ == RobotName::M20) {
                namespace fs = std::filesystem;
                // Resolve policy paths relative to the running executable,
                // NOT __FILE__ (which is a compile-time source path that may
                // not exist at runtime after installation).
                fs::path exe_dir = fs::canonical("/proc/self/exe").parent_path();
                fs::path policy_base = exe_dir / "policy";

                // Fallback: if policies aren't next to the executable (e.g.
                // running from the build tree), try the original source-relative path.
                if (!fs::exists(policy_base)) {
                    fs::path src_base = fs::path(__FILE__).parent_path() / ".." / ".." / "policy";
                    if (fs::exists(src_base)) {
                        policy_base = fs::canonical(src_base);
                    } else {
                        std::cerr << "[ERROR] Cannot find policy directory at:\n"
                                  << "  " << (exe_dir / "policy").string() << "\n"
                                  << "  " << fs::weakly_canonical(src_base).string() << "\n";
                        exit(1);
                    }
                }

                auto low_path       = policy_base / "low_speed"  / "policy.onnx";
                auto high_x_path    = policy_base / "high_speed" / "x_rash"   / "policy.onnx";
                auto high_y_path    = policy_base / "high_speed" / "y_jump"   / "policy.onnx";
                auto high_diag_path = policy_base / "high_speed" / "diagonal" / "policy.onnx";

                // Verify all policy files exist before proceeding
                for (const auto& p : {low_path, high_x_path, high_y_path, high_diag_path}) {
                    if (!fs::exists(p)) {
                        std::cerr << "[ERROR] Policy file not found: " << p.string() << "\n";
                        exit(1);
                    }
                }
                m20_policy_low_speed_  = std::make_shared<M20PolicyRunner>("m20_low_speed",  low_path.string());
                m20_policy_high_x_    = std::make_shared<M20PolicyRunner>("m20_high_x",     high_x_path.string());
                m20_policy_high_y_    = std::make_shared<M20PolicyRunner>("m20_high_y",     high_y_path.string());
                m20_policy_high_diag_ = std::make_shared<M20PolicyRunner>("m20_high_diag",  high_diag_path.string());
                m20_policy_high_speed_ = m20_policy_high_x_;  // default high-speed
                m20_policy_ = m20_policy_low_speed_;

                // ---- Depth camera policies (optional) ----
                M20Interface* m20_ri = dynamic_cast<M20Interface*>(ri_ptr_.get());
                auto depth_low_path       = policy_base / "depth_low_speed"  / "policy.onnx";
                auto depth_high_x_path    = policy_base / "depth_high_speed" / "x_rash"   / "policy.onnx";
                auto depth_high_y_path    = policy_base / "depth_high_speed" / "y_jump"   / "policy.onnx";
                auto depth_high_diag_path = policy_base / "depth_high_speed" / "diagonal" / "policy.onnx";
                if (false && fs::exists(depth_low_path)) {
                    depth_policy_low_       = std::make_shared<M20DepthPolicyRunner>(
                        "depth_low_speed", depth_low_path.string(), m20_ri);
                    depth_policy_high_x_    = std::make_shared<M20DepthPolicyRunner>(
                        "depth_high_x", depth_high_x_path.string(), m20_ri);
                    depth_policy_high_y_    = std::make_shared<M20DepthPolicyRunner>(
                        "depth_high_y", depth_high_y_path.string(), m20_ri);
                    depth_policy_high_diag_ = std::make_shared<M20DepthPolicyRunner>(
                        "depth_high_diag", depth_high_diag_path.string(), m20_ri);
                    has_depth_policy_ = true;
                    std::cout << "[DEPTH POLICY] Depth camera policies loaded.\n";
                } else {
                    std::cout << "[DEPTH POLICY] No depth camera policies found ("
                              << depth_low_path.string() << " missing). Using standard policies only.\n";
                }
            }

            policy_ptr_ = m20_policy_;
            if (!policy_ptr_) {
                std::cerr << "error policy" << std::endl;
                exit(0);
            }
            policy_ptr_->DisplayPolicyInfo();
            init_rbs_();
        }

        ~RLControlState() {}

        virtual void OnEnter() {
            state_run_cnt_ = -1;
            start_flag_ = true;
            auto* cmd = uc_ptr_->GetUserCommand();
            cmd->forward_vel_scale = 0.0f;
            cmd->side_vel_scale = 0.0f;
            cmd->turnning_vel_scale = 0.0f;
            run_policy_thread_ = std::thread(std::bind(&RLControlState::PolicyRunner, this));
            policy_ptr_->OnEnter(rbs_);
            StateBase::msfb_.UpdateCurrentState(RobotMotionState::RLControlMode);
        };

        virtual void OnExit() {
            start_flag_ = false;
            run_policy_thread_.join();
            state_run_cnt_ = -1;
        }

        virtual void Run() {
            UpdateRobotObservation();

            // Switch policy based on speed mode and direction
            float speed_mode = uc_ptr_->GetUserCommand()->reserved_scale;
            float fwd  = uc_ptr_->GetUserCommand()->forward_vel_scale;
            float side = uc_ptr_->GetUserCommand()->side_vel_scale;

            if (has_depth_policy_) {
                // --- Depth camera policy switching ---
                std::shared_ptr<M20DepthPolicyRunner> desired_depth;
                if (speed_mode >= 0.5f) {
                    bool has_fwd  = std::abs(fwd)  > 0.01f;
                    bool has_side = std::abs(side) > 0.01f;
                    if (has_fwd && has_side)  desired_depth = depth_policy_high_diag_;
                    else if (has_side)        desired_depth = depth_policy_high_y_;
                    else                      desired_depth = depth_policy_high_x_;
                } else {
                    desired_depth = depth_policy_low_;
                }
                if (desired_depth != policy_ptr_) {
                    policy_ptr_ = desired_depth;
                    std::cout << "[POLICY] Active (depth): " << policy_ptr_->policy_name_ << "\n";
                }
            } else {
                // --- Standard policy switching ---
                std::shared_ptr<M20PolicyRunner> desired;
                if (speed_mode >= 0.5f) {
                    bool has_fwd  = std::abs(fwd)  > 0.01f;
                    bool has_side = std::abs(side) > 0.01f;
                    if (has_fwd && has_side)  desired = m20_policy_high_diag_;
                    else if (has_side)        desired = m20_policy_high_y_;
                    else                      desired = m20_policy_high_x_;
                } else {
                    desired = m20_policy_low_speed_;
                }
                if (desired != policy_ptr_) {
                    policy_ptr_ = desired;
                    std::cout << "[POLICY] Active: " << policy_ptr_->policy_name_ << "\n";
                }
            }

            if (state_run_cnt_ % 50 == 0) {
                std::cout << "[ROBOT] omega=" << rbs_.base_omega.transpose()
                          << " acc=" << rbs_.base_acc.transpose() << "\n";
            }
            state_run_cnt_++;
        }

        virtual bool LoseControlJudge() {
            if (uc_ptr_->GetUserCommand()->target_mode == uint8_t(RobotMotionState::JointDamping)) return true;
            return PostureUnsafeCheck();
        }

        bool PostureUnsafeCheck() {
            // Vec3f rpy = ri_ptr_->GetImuRpy();
            // if(rpy(0) > 30./180*M_PI || rpy(1) > 45./180*M_PI){
            //     std::cout << "posture value: " << 180./M_PI*rpy.transpose() << std::endl;
            //     return true;
            // }
            return false;
        }

        virtual StateName GetNextStateName() {
            if (uc_ptr_->GetUserCommand()->safe_control_mode != 0) return StateName::kJointDamping;
            return StateName::kRLControl;
        }
    };
};