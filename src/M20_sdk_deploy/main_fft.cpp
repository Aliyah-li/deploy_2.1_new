/**
 * @file main_fft.cpp
 * @brief Entry point for FFT (Fast Fine-Tuning) sim2sim deployment
 * @author DeepRobotics
 * @version 1.0
 * @date 2025-11-07
 *
 * @copyright Copyright (c) 2025  DeepRobotics
 *
 * Uses FftStateMachine — single FFT ONNX policy, fixed URDF action reference,
 * 58-dim observation with height appended at index 57.
 */

#include "quadruped_wheel/fft_state_machine.hpp"

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
    std::cout << "[FFT] State Machine Starting — FFT Sim2Sim Deployment" << std::endl;
    rclcpp::init(0, nullptr);

    std::shared_ptr<StateMachineBase> fsm =
        std::make_shared<qw::FftStateMachine>(RobotName::M20, RemoteCommandType::kKeyBoard);

    fsm->Start();
    fsm->Run();
    fsm->Stop();

    rclcpp::shutdown();
    return 0;
}
