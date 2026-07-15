/**
 * @file main_strong2.cpp
 * @brief Entry point for Strong2 sim2sim deployment (变高度)
 * @author DeepRobotics
 * @version 1.0
 * @date 2025-06-27
 *
 * Uses Strong2StateMachine — single ONNX, variable-height command and a
 * fixed URDF action reference.
 *
 * Config: sim2sim_deploy_config.md
 * Policy: strong_ppo_base/policy.onnx
 *
 * Exact 58-dim observation order from the exported training env.yaml:
 *   [0:3]   base_ang_vel   (3)
 *   [3:6]   projected_gravity (3)
 *   [6:9]   velocity command (3)
 *   [9:25]  joint_pos_rel  (16)
 *   [25:41] joint_vel      (16)
 *   [41:57] last_action    (16)
 *   [57]    base_height_command (1)
 *   3+3+3+16+16+16+1 = 58
 */

#include "quadruped_wheel/strong2_state_machine.hpp"

#ifdef USE_SIMULATION
    #define BACKWARD_HAS_DW 1
    #include "backward.hpp"
    namespace backward {
        backward::SignalHandling sh;
    }
#endif

using namespace types;
MotionStateFeedback StateBase::msfb_ = MotionStateFeedback();

int main() {
    std::cout << "[STRONG2] State Machine Starting — Strong2 Sim2Sim (变高度)" << std::endl;
    rclcpp::init(0, nullptr);

    std::shared_ptr<StateMachineBase> fsm =
        std::make_shared<qw::Strong2StateMachine>(RobotName::M20, RemoteCommandType::kKeyBoard);

    fsm->Start();
    fsm->Run();
    fsm->Stop();

    rclcpp::shutdown();
    return 0;
}
