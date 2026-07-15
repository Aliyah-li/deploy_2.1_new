// fixed_direction_keyboard_interface.hpp
#pragma once

#include "user_command_interface.h"
#include "custom_types.h"
#include "json.hpp"
#include <thread>
#include <atomic>
#include <unordered_map>
#include <unordered_set>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <mutex>
#include <vector>

using namespace interface;
using namespace types;

class FixedDirectionKeyboardInterface : public UserCommandInterface
{
private:
    std::atomic<bool> running_{false};
    std::thread kb_thread_;
    mutable std::mutex keys_mutex_;

    int run_cnt_ = 0;

    float ACCELERATION = 1.5f;
    float MAX_VELOCITY = 1.5f;

    float ACCELERATION_LOW  = 1.5f;
    float MAX_VELOCITY_LOW  = 1.5f;
    float ACCELERATION_HIGH = 3.5f;
    float MAX_VELOCITY_HIGH = 2.5f;

    float max_forward_ = 2.0f;
    float max_side_    = 2.0f;
    float max_yaw_     = 0.7f;

    std::unordered_set<char> held_keys_;
    std::unordered_map<char, double> last_seen_time_;

    const double key_timeout_ms_ = 500.0;

    struct DirectionConfig {
        std::string name;
        float angle; // radians: 0=x, pi/2=y, pi/4=diagonal
    };

    // current active direction
    bool has_direction_ = false;
    DirectionConfig current_dir_cfg_{"x", 0.0f};

    // Speed mode: 0=low, 1=high, 2=mixed
    int speed_mode_ = 0;
    bool current_motion_high_speed_ = false;  // speed of the currently running motion
    bool current_motion_mixed_high_ = false;  // whether current motion is the high phase of mixed

    // Mixed mode: configurable high-speed direction
    DirectionConfig mixed_high_dir_{"y", 1.5708f};

    // key -> direction config
    std::unordered_map<char, DirectionConfig> key_direction_map_ = {
        {'1', {"x",        0.0f}},
        {'2', {"y",        1.5708f}},
        {'3', {"diagonal", 0.7854f}},
    };

    float current_velocity_ = 0.0f;
    bool is_accelerating_ = false;
    double move_start_time_ = -1.0;  // ms timestamp when motion started, -1 = idle
    bool q_hold_active_ = false;
    double last_q_seen_time_ = -1.0;

    float run_duration_low_x_    = 900.0f;  // ms
    float run_duration_low_y_    = 800.0f;  // ms
    float run_duration_low_diag_ = 1000.0f; // ms
    float run_duration_high_x_    = 500.0f;  // ms
    float run_duration_high_y_    = 400.0f;  // ms
    float run_duration_high_diag_ = 600.0f;  // ms

    // Mixed mode high-speed phase durations (default = high-speed values)
    float run_duration_mixed_high_x_    = 500.0f;  // ms
    float run_duration_mixed_high_y_    = 400.0f;  // ms
    float run_duration_mixed_high_diag_ = 600.0f;  // ms

    // ── Height control ──
    float height_min_      = 0.32f;
    float height_max_      = 0.425f;
    float height_step_     = 0.005f;
    float height_nominal_  = 0.40f;

    // ── Yaw hold tracking ──
    double last_a_seen_time_ = -1.0;
    double last_d_seen_time_ = -1.0;
    static constexpr double yaw_timeout_ms_ = 200.0;
    float yaw_magnitude_ = 0.5f;   // rad/s when A/D held

    void ClipNumber(float& num, float low, float high)
    {
        if (num < low) num = low;
        if (num > high) num = high;
    }

    double GetCurrentTimeStamp()
    {
        static auto start = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(now - start).count();
    }

    static void setup_raw_mode()
    {
        termios t{};
        tcgetattr(STDIN_FILENO, &t);
        termios raw = t;
        raw.c_lflag &= ~(ECHO | ICANON);
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 0;
        tcsetattr(STDIN_FILENO, TCSANOW, &raw);
        int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
        fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
    }

    static void restore_terminal()
    {
        termios t{};
        tcgetattr(STDIN_FILENO, &t);
        t.c_lflag |= (ECHO | ICANON);
        tcsetattr(STDIN_FILENO, TCSANOW, &t);
    }

    void update_fixed_direction_movement()
    {
        std::lock_guard<std::mutex> lock(keys_mutex_);

        if (q_hold_active_) {
            double elapsed_since_q = GetCurrentTimeStamp() - last_q_seen_time_;
            float accel = current_motion_high_speed_ ? ACCELERATION_HIGH : ACCELERATION_LOW;
            if (elapsed_since_q > key_timeout_ms_) {
                // Q released — decelerate smoothly toward 0
                if (current_velocity_ > 0.0f) {
                    current_velocity_ -= accel * 0.001f;
                    if (current_velocity_ < 0.0f) current_velocity_ = 0.0f;
                }
                if (current_velocity_ <= 0.0f) {
                    current_velocity_ = 0.0f;
                    q_hold_active_ = false;
                    is_accelerating_ = false;
                    move_start_time_ = -1.0;
                    usr_cmd_->forward_vel_scale  = 0.0f;
                    usr_cmd_->side_vel_scale     = 0.0f;
                    usr_cmd_->turnning_vel_scale = 0.0f;
                    std::cout << "[MOVE] Q released, Velocity: 0.0\n" << std::flush;
                    return;
                }
            } else {
                // Q held — accelerate smoothly toward target
                float target = current_motion_high_speed_ ? MAX_VELOCITY_HIGH : MAX_VELOCITY_LOW;
                if (current_velocity_ < target) {
                    current_velocity_ += accel * 0.001f;
                    if (current_velocity_ > target) current_velocity_ = target;
                }
            }
            usr_cmd_->forward_vel_scale  = current_velocity_ * std::cos(current_dir_cfg_.angle);
            usr_cmd_->side_vel_scale     = current_velocity_ * std::sin(current_dir_cfg_.angle);
            usr_cmd_->turnning_vel_scale = 0.0f;
            return;
        }

        if (!has_direction_ || move_start_time_ < 0.0)
            return;

        float duration_ms;
        if (current_motion_high_speed_) {
            const std::string& dir = current_dir_cfg_.name;
            if (current_motion_mixed_high_) {
                // Mixed mode high phase: use mixed durations
                if (dir == "x")             duration_ms = run_duration_mixed_high_x_;
                else if (dir == "y")        duration_ms = run_duration_mixed_high_y_;
                else                        duration_ms = run_duration_mixed_high_diag_;
            } else {
                // Pure high speed mode
                if (dir == "x")             duration_ms = run_duration_high_x_;
                else if (dir == "y")        duration_ms = run_duration_high_y_;
                else                        duration_ms = run_duration_high_diag_;
            }
        } else {
            const std::string& dir = current_dir_cfg_.name;
            if (dir == "x")             duration_ms = run_duration_low_x_;
            else if (dir == "y")        duration_ms = run_duration_low_y_;
            else                        duration_ms = run_duration_low_diag_;
        }
        double elapsed = GetCurrentTimeStamp() - move_start_time_;

        if (elapsed >= duration_ms) {
            // Time's up — stop automatically
            current_velocity_ = 0.0f;
            is_accelerating_  = false;
            move_start_time_  = -1.0;
            usr_cmd_->forward_vel_scale  = 0.0f;
            usr_cmd_->side_vel_scale     = 0.0f;
            usr_cmd_->turnning_vel_scale = 0.0f;
            std::cout << "[MOVE] Direction: IDLE, Velocity: 0.0\n" << std::flush;
            return;
        }

        if (is_accelerating_ && current_velocity_ < MAX_VELOCITY) {
            float accel = current_motion_high_speed_ ? ACCELERATION_HIGH : ACCELERATION_LOW;
            float max_v = current_motion_high_speed_ ? MAX_VELOCITY_HIGH : MAX_VELOCITY_LOW;
            current_velocity_ += accel * 0.001f;
            if (current_velocity_ >= max_v) {
                current_velocity_ = max_v;
                is_accelerating_ = false;
            }
        }

        usr_cmd_->forward_vel_scale  = current_velocity_ * std::cos(current_dir_cfg_.angle);
        usr_cmd_->side_vel_scale     = current_velocity_ * std::sin(current_dir_cfg_.angle);
        usr_cmd_->turnning_vel_scale = 0.0f;
    }

    void print_status()
    {
        // Emit structured log lines at ~10 Hz for monitor_ui.py to parse
        if (run_cnt_ % 100 != 0) return;

        // Robot state
        const char* robot_state = "Unknown";
        switch (msfb_->GetCurrentState()) {
            case RobotMotionState::WaitingForStand: robot_state = "Waiting";    break;
            case RobotMotionState::StandingUp:      robot_state = "StandingUp"; break;
            case RobotMotionState::JointDamping:    robot_state = "Damping";    break;
            case RobotMotionState::RLControlMode:   robot_state = "RLControl";  break;
        }

        // Speed mode
        const char* speed_labels[] = {"LOW", "HIGH", "MIXED"};
        const char* speed_str = speed_labels[speed_mode_];

        // Motion phase + remaining time
        std::string phase_str;
        float remain_s = 0.f;
        if (move_start_time_ < 0.0) {
            phase_str = "IDLE";
        } else {
            double elapsed_s = (GetCurrentTimeStamp() - move_start_time_) / 1000.0;
            float  dur_ms = 0.f;
            const std::string& dir = current_dir_cfg_.name;
            if (current_motion_high_speed_) {
                if (current_motion_mixed_high_) {
                    dur_ms = (dir == "x") ? run_duration_mixed_high_x_ :
                             (dir == "y") ? run_duration_mixed_high_y_ : run_duration_mixed_high_diag_;
                } else {
                    dur_ms = (dir == "x") ? run_duration_high_x_ :
                             (dir == "y") ? run_duration_high_y_ : run_duration_high_diag_;
                }
            } else {
                dur_ms = (dir == "x") ? run_duration_low_x_ :
                         (dir == "y") ? run_duration_low_y_ : run_duration_low_diag_;
            }
            remain_s = dur_ms / 1000.f - static_cast<float>(elapsed_s);
            if (remain_s < 0.f) remain_s = 0.f;
            phase_str = current_motion_high_speed_ ? "HIGH" : "LOW";
        }

        // Emit one structured line that monitor_ui.py can parse
        char buf[320];
        snprintf(buf, sizeof(buf),
            "[STATUS] robot=%s speed=%s dir=%s phase=%s remain=%.2f "
            "vx=%.4f vy=%.4f yaw=%.4f spd=%.4f h=%.3f mix_hi=%s",
            robot_state, speed_str, current_dir_cfg_.name.c_str(),
            phase_str.c_str(), remain_s,
            usr_cmd_->forward_vel_scale, usr_cmd_->side_vel_scale,
            usr_cmd_->turnning_vel_scale, current_velocity_,
            usr_cmd_->height_command, mixed_high_dir_.name.c_str());
        std::cout << buf << "\n" << std::flush;
    }

    void reset_all_movements_unlocked()
    {
        q_hold_active_ = false;
        last_q_seen_time_ = -1.0;
        move_start_time_ = -1.0;
        current_velocity_ = 0.0f;
        is_accelerating_ = false;
        usr_cmd_->forward_vel_scale = 0.0f;
        usr_cmd_->side_vel_scale = 0.0f;
        usr_cmd_->turnning_vel_scale = 0.0f;
    }

    void reset_all_movements()
    {
        std::lock_guard<std::mutex> lock(keys_mutex_);
        reset_all_movements_unlocked();
    }

    void process_mode_command(char k)
    {
        if (k == 'r') {
            usr_cmd_->target_mode = uint8_t(RobotMotionState::JointDamping);
            std::cout << "[MODE] Joint Damping\n";
        }
        else if (k == 'z' && msfb_->GetCurrentState() == RobotMotionState::WaitingForStand) {
            usr_cmd_->target_mode = uint8_t(RobotMotionState::StandingUp);
            std::cout << "[MODE] Standing Up\n";
        }
        else if (k == 'c' && msfb_->GetCurrentState() == RobotMotionState::StandingUp) {
            usr_cmd_->target_mode = uint8_t(RobotMotionState::RLControlMode);
            std::cout << "[MODE] RL Control\n";
        }
    }

    void process_direction_mapping(char k)
    {
        auto it = key_direction_map_.find(k);
        if (it != key_direction_map_.end()) {
            std::lock_guard<std::mutex> lock(keys_mutex_);
            current_dir_cfg_ = it->second;
            has_direction_ = true;
            // if already moving, restart in new direction immediately
            if (current_velocity_ > 0.0f || is_accelerating_) {
                current_velocity_ = 0.0f;
                is_accelerating_ = true;
            }
            usr_cmd_->forward_vel_scale = 0.0f;
            usr_cmd_->side_vel_scale    = 0.0f;
            std::cout << "[DIRECTION] Switched to " << current_dir_cfg_.name << " direction\n" << std::flush;
        }
    }

    void keyboard_loop()
    {
        setup_raw_mode();

        std::cout << "\n╔════════════════════════════════════════════════╗\n"
                  << "║    FIXED DIRECTION KEYBOARD TELEOP READY       ║\n"
                  << "╚════════════════════════════════════════════════╝\n"
                  << "  Move:   hold Q (fixed forward, stop on release)\n"
                  << "  Yaw:    hold A (left) / D (right)\n"
                  << "  Height: W (up) / S (down)  0.32 ~ 0.425 m\n"
                  << "  Speed:  T (cycle: low / high / mixed)\n"
                  << "  Dir:    1/2/3 (x / y / diagonal)\n"
                  << "  Mode:   R (damping)  Z (stand)  C (control)\n\n";

        char ch;
        while (running_) {
            double now = GetCurrentTimeStamp();
            usr_cmd_->time_stamp = now;

            while (read(STDIN_FILENO, &ch, 1) == 1) {
                char k = std::tolower(static_cast<unsigned char>(ch));

                if (k == 'r' || k == 'z' || k == 'c') {
                    process_mode_command(k);
                    continue;
                }

                if (k == '1' || k == '2' || k == '3') {
                    process_direction_mapping(k);
                    continue;
                }

                if (k == 'q' && msfb_->GetCurrentState() == RobotMotionState::RLControlMode) {
                    std::lock_guard<std::mutex> lock(keys_mutex_);
                    current_motion_high_speed_ = (speed_mode_ == 1);
                    current_motion_mixed_high_ = false;
                    usr_cmd_->reserved_scale = current_motion_high_speed_ ? 1.0f : 0.0f;
                    is_accelerating_ = true;
                    q_hold_active_ = true;
                    last_q_seen_time_ = GetCurrentTimeStamp();
                    if (!has_direction_) {
                        current_dir_cfg_ = {"x", 0.0f};
                        has_direction_ = true;
                    }
                    std::cout << "[MOVE] Hold Q: dir=" << current_dir_cfg_.name
                              << " vx=" << usr_cmd_->forward_vel_scale
                              << " vy=" << usr_cmd_->side_vel_scale
                              << "\n" << std::flush;
                }

                if (k == 's' && speed_mode_ == 2 && msfb_->GetCurrentState() == RobotMotionState::RLControlMode) {
                    std::lock_guard<std::mutex> lock(keys_mutex_);
                    // Mixed mode S → high speed in configured direction
                    current_motion_high_speed_ = true;
                    current_motion_mixed_high_ = true;
                    current_dir_cfg_ = mixed_high_dir_;
                    has_direction_ = true;
                    usr_cmd_->reserved_scale = 1.0f;
                    current_velocity_ = 0.0f;
                    is_accelerating_ = true;
                    move_start_time_ = GetCurrentTimeStamp();
                    std::cout << "[MOVE] Direction: " << mixed_high_dir_.name
                              << ", Velocity: " << MAX_VELOCITY_HIGH << "\n" << std::flush;
                }

                // ── Height control: W / S ──
                if (k == 'w' || k == 's') {
                    float dir = (k == 'w') ? 1.0f : -1.0f;
                    usr_cmd_->height_command += dir * height_step_;
                    if (usr_cmd_->height_command < height_min_) usr_cmd_->height_command = height_min_;
                    if (usr_cmd_->height_command > height_max_) usr_cmd_->height_command = height_max_;
                    std::cout << "[HEIGHT] " << usr_cmd_->height_command << " m\n" << std::flush;
                }

                // ── Yaw control: A (left) / D (right) ──
                if (k == 'a') { last_a_seen_time_ = now; }
                if (k == 'd') { last_d_seen_time_ = now; }

                if (k == 't') {
                    speed_mode_ = (speed_mode_ + 1) % 3;
                    const char* mode_names[] = {"LOW SPEED", "HIGH SPEED", "MIXED"};
                    usr_cmd_->reserved_scale = (speed_mode_ == 1) ? 1.0f : 0.0f;
                    std::cout << "[SPEED] Mode: " << mode_names[speed_mode_] << "\n";
                }
            }

            // ── Yaw hold / decay ──
            {
                double now_ms = GetCurrentTimeStamp();
                bool a_active = (now_ms - last_a_seen_time_) < yaw_timeout_ms_;
                bool d_active = (now_ms - last_d_seen_time_) < yaw_timeout_ms_;
                if (a_active && !d_active)
                    usr_cmd_->turnning_vel_scale = -yaw_magnitude_;
                else if (d_active && !a_active)
                    usr_cmd_->turnning_vel_scale =  yaw_magnitude_;
                else if (!a_active && !d_active)
                    usr_cmd_->turnning_vel_scale = 0.0f;
                // both held → stay at previous value (no change)
            }

            if (msfb_->GetCurrentState() == RobotMotionState::RLControlMode)
                update_fixed_direction_movement();

            print_status();
            run_cnt_++;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        reset_all_movements();
        restore_terminal();
        std::cout << "\n[KEYBOARD] Stopped.\n";
    }

public:
    FixedDirectionKeyboardInterface(RobotName robot_name) : UserCommandInterface(robot_name)
    {
        // Load config from JSON (next to binary)
        namespace fs = std::filesystem;
        char exe_buf[4096] = {};
        readlink("/proc/self/exe", exe_buf, sizeof(exe_buf) - 1);
        auto config_path = fs::path(exe_buf).parent_path() / "teleop_config.json";
        std::ifstream f(config_path);
        if (f.is_open()) {
            nlohmann::json j;
            f >> j;
            if (j.contains("max_velocity")) MAX_VELOCITY = j["max_velocity"].get<float>();
            if (j.contains("acceleration"))  ACCELERATION = j["acceleration"].get<float>();

            // Section: low_speed
            if (j.contains("low_speed")) {
                auto& ls = j["low_speed"];
                if (ls.contains("max_velocity"))    MAX_VELOCITY_LOW  = ls["max_velocity"].get<float>();
                if (ls.contains("acceleration"))    ACCELERATION_LOW  = ls["acceleration"].get<float>();
                if (ls.contains("duration_x"))      run_duration_low_x_    = ls["duration_x"].get<float>()    * 1000.f;
                if (ls.contains("duration_y"))      run_duration_low_y_    = ls["duration_y"].get<float>()    * 1000.f;
                if (ls.contains("duration_diagonal")) run_duration_low_diag_ = ls["duration_diagonal"].get<float>() * 1000.f;
            }
            // Section: high_speed
            if (j.contains("high_speed")) {
                auto& hs = j["high_speed"];
                if (hs.contains("max_velocity"))    MAX_VELOCITY_HIGH  = hs["max_velocity"].get<float>();
                if (hs.contains("acceleration"))    ACCELERATION_HIGH  = hs["acceleration"].get<float>();
                if (hs.contains("duration_x"))      run_duration_high_x_    = hs["duration_x"].get<float>()    * 1000.f;
                if (hs.contains("duration_y"))      run_duration_high_y_    = hs["duration_y"].get<float>()    * 1000.f;
                if (hs.contains("duration_diagonal")) run_duration_high_diag_ = hs["duration_diagonal"].get<float>() * 1000.f;
            }
            // Section: mixed — default durations to high_speed values, then override
            run_duration_mixed_high_x_    = run_duration_high_x_;
            run_duration_mixed_high_y_    = run_duration_high_y_;
            run_duration_mixed_high_diag_ = run_duration_high_diag_;
            if (j.contains("mixed")) {
                auto& mx = j["mixed"];
                if (mx.contains("high_direction")) {
                    auto& mhd = mx["high_direction"];
                    mixed_high_dir_.name  = mhd.value("name",  "y");
                    mixed_high_dir_.angle = mhd.value("angle", 1.5708f);
                    std::cout << "[CONFIG] Mixed high direction: " << mixed_high_dir_.name
                              << " angle=" << mixed_high_dir_.angle << " rad\n";
                }
                if (mx.contains("duration_x"))        run_duration_mixed_high_x_    = mx["duration_x"].get<float>()        * 1000.f;
                if (mx.contains("duration_y"))        run_duration_mixed_high_y_    = mx["duration_y"].get<float>()        * 1000.f;
                if (mx.contains("duration_diagonal")) run_duration_mixed_high_diag_ = mx["duration_diagonal"].get<float>() * 1000.f;
            }
            // Section: key_directions
            if (j.contains("key_directions")) {
                for (auto& [key_str, val] : j["key_directions"].items()) {
                    if (key_str.size() == 1) {
                        char k = key_str[0];
                        DirectionConfig dc;
                        dc.name  = val.value("name",  "x");
                        dc.angle = val.value("angle", 0.0f);
                        key_direction_map_[k] = dc;
                        std::cout << "[CONFIG] Key " << k << " → " << dc.name
                                  << " angle=" << dc.angle << " rad\n";
                    }
                }
            }
            std::cout << "[CONFIG] max_velocity=" << MAX_VELOCITY
                      << " acceleration=" << ACCELERATION << "\n";
        } else {
            std::cerr << "[CONFIG] teleop_config.json not found at " << config_path << ", using defaults.\n";
        }

        // default direction is key '1'
        auto it = key_direction_map_.find('1');
        if (it != key_direction_map_.end()) {
            current_dir_cfg_ = it->second;
            has_direction_ = true;
        }

        std::cout << "[INIT] Default direction: " << current_dir_cfg_.name
                  << " (angle=" << current_dir_cfg_.angle << " rad)\n";
        std::memset(usr_cmd_, 0, sizeof(UserCommand));
        usr_cmd_->height_command = height_nominal_;
    }

    ~FixedDirectionKeyboardInterface() { Stop(); }

    void Start() override
    {
        if (running_) return;
        running_ = true;
        kb_thread_ = std::thread(&FixedDirectionKeyboardInterface::keyboard_loop, this);
    }

    void Stop() override
    {
        running_ = false;
        if (kb_thread_.joinable())
            kb_thread_.join();
        reset_all_movements();
    }

    UserCommand* GetUserCommand() override { return usr_cmd_; }

    void set_max_velocities(float fwd, float side, float yaw)
    {
        max_forward_ = std::abs(fwd);
        max_side_    = std::abs(side);
        max_yaw_     = std::abs(yaw);
        std::cout << "[CONFIG] Max velocities: fwd=" << max_forward_
                  << " side=" << max_side_
                  << " yaw=" << max_yaw_ << "\n";
    }
};
