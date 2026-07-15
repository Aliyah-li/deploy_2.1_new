/**
 * @file strong2_state_machine.hpp
 * @brief State machine for Strong2 sim2sim deployment (变高度)
 *
 * State transitions:
 *   kIdle → kStandUp → kRLControl (Strong2 policy)
 *   any → kJointDamping (on safety trigger)
 *
 * Uses Strong2RLControlState with M20Fft2PolicyRunner (height-aware IK).
 * Policy path: strong_ppo_base/policy.onnx
 */

#pragma once

#include "state_base.h"
#include "state_machine_base.h"
#include "quadruped_wheel/idle_state.hpp"
#include "quadruped_wheel/standup_state.hpp"
#include "quadruped_wheel/joint_damping_state.hpp"
#include "quadruped_wheel/strong2_rl_control_state.hpp"
#include "keyboard_interface.hpp"
#include "fixed_direction_keyboard_interface.hpp"
// Optional evdev keyboard path kept for rollback, but disabled for ARM/no-lib builds.
// #include "fixed_direction_keyboard_interface_sim.hpp"
#include "hardware/m20_interface.hpp"

namespace qw {
class Strong2StateMachine : public StateMachineBase {
private:

    std::shared_ptr<StateBase> idle_controller_;
    std::shared_ptr<StateBase> standup_controller_;
    std::shared_ptr<StateBase> rl_controller_;
    std::shared_ptr<StateBase> joint_damping_controller_;

public:
    const RobotName robot_name_;
    const RemoteCommandType remote_cmd_type_;

    Strong2StateMachine(RobotName robot_name, RemoteCommandType rct)
        : StateMachineBase(RobotType::QuadrupedWheel),
          robot_name_(robot_name),
          remote_cmd_type_(rct) {}

    ~Strong2StateMachine() {}

    void Start() {
        if (remote_cmd_type_ == RemoteCommandType::kKeyBoard) {
            uc_ptr_ = std::make_shared<FixedDirectionKeyboardInterface>(robot_name_);
        } else {
            std::cerr << "[STRONG2] error: unsupported command type" << std::endl;
            exit(0);
        }

        uc_ptr_->SetMotionStateFeedback(&StateBase::msfb_);

        if (robot_name_ == RobotName::M20) {
            ri_ptr_ = std::make_shared<M20Interface>("M20");
            cp_ptr_ = std::make_shared<ControlParameters>(robot_name_);
        }

        std::shared_ptr<ControllerData> data_ptr = std::make_shared<ControllerData>();
        data_ptr->ri_ptr = ri_ptr_;
        data_ptr->uc_ptr = uc_ptr_;
        data_ptr->cp_ptr = cp_ptr_;

        sc_ptr_ = std::make_shared<SafeController>(QuadrupedWheel, "");
        sc_ptr_->SetRobotDataSource(ri_ptr_);
        sc_ptr_->SetUserCommandDataSource(uc_ptr_);

        idle_controller_          = std::make_shared<IdleState>(robot_name_, "idle_state", data_ptr);
        standup_controller_       = std::make_shared<StandUpState>(robot_name_, "standup_state", data_ptr);
        rl_controller_            = std::make_shared<Strong2RLControlState>(robot_name_, "strong2_rl_control", data_ptr);
        joint_damping_controller_ = std::make_shared<JointDampingState>(robot_name_, "joint_damping", data_ptr);

        current_controller_ = idle_controller_;
        current_state_name_ = kIdle;
        next_state_name_ = kIdle;

        uc_ptr_->Start();
        ri_ptr_->Start();
        sc_ptr_->Start();
        current_controller_->OnEnter();

        std::cout << "[STRONG2] State machine started. Policy: strong_ppo_base (变高度)" << std::endl;
    }

    std::shared_ptr<StateBase> GetStateControllerPtr(StateName state_name) {
        switch (state_name) {
            case StateName::kInvalid:      return nullptr;
            case StateName::kIdle:         return idle_controller_;
            case StateName::kStandUp:      return standup_controller_;
            case StateName::kRLControl:    return rl_controller_;
            case StateName::kJointDamping: return joint_damping_controller_;
            default: {
                std::cerr << "[STRONG2] error: unknown state name" << std::endl;
                return joint_damping_controller_;
            }
        }
        return nullptr;
    }

    void Stop() {
        sc_ptr_->Stop();
        uc_ptr_->Stop();
        ri_ptr_->Stop();
    }
};
};
