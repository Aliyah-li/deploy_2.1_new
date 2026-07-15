/**
 * @file main_fft2.cpp
 * @brief Entry point for FFT2 sim2sim deployment (变高度)
 * @author DeepRobotics
 * @version 1.0
 * @date 2025-06-27
 *
 * @copyright Copyright (c) 2025  DeepRobotics
 *
 * Uses Fft2StateMachine — single ONNX, variable-height command and fixed URDF
 * action reference. Height is the final observation at index 57.
 *
 * Key difference from main_fft:
 *  - Initial height = 0.38 m (matches StandUp), not 0.40 m
 *  - Default policy subdirectory: fft/
 *  - Height changes the policy input only; StandUp IK is not the action offset
 */

#include "quadruped_wheel/fft2_state_machine.hpp"

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
    std::cout << "[FFT2] State Machine Starting — FFT2 Sim2Sim Deployment (变高度)" << std::endl;
    rclcpp::init(0, nullptr);

    std::shared_ptr<StateMachineBase> fsm =
        std::make_shared<qw::Fft2StateMachine>(RobotName::M20, RemoteCommandType::kKeyBoard);

    fsm->Start();
    fsm->Run();
    fsm->Stop();

    rclcpp::shutdown();
    return 0;
}
