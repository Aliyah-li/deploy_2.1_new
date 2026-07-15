// fixed_direction_keyboard_interface_sim.hpp
// libevdev-based version — reads directly from /dev/input/event*
// avoids terminal stdin conflict with curses / monitor_ui.py
// sudo apt install libevdev-dev
// sudo adduser $USER input
// newgrp input

#pragma once

#include "user_command_interface.h"
#include "custom_types.h"
#include "json.hpp"
#include <libevdev-1.0/libevdev/libevdev.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <termios.h>
#include <thread>
#include <atomic>
#include <unordered_map>
#include <unordered_set>
#include <cctype>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <mutex>
#include <vector>
#include <cstring>

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

    std::unordered_set<int> pressed_keys_;

    const double key_timeout_ms_ = 500.0;

    struct DirectionConfig {
        std::string name;
        float angle; // radians: 0=x, pi/2=y, pi/4=diagonal
    };

    struct EvDevKeyboard {
        int fd;
        struct libevdev* dev;
        std::string name;
    };
    std::vector<EvDevKeyboard> keyboards_;

    // current active direction
    bool has_direction_ = false;
    DirectionConfig current_dir_cfg_{"x", 0.0f};

    // Speed mode: 0=low, 1=high, 2=mixed
    int speed_mode_ = 0;
    bool current_motion_high_speed_ = false;
    bool current_motion_mixed_high_ = false;

    // Mixed mode: configurable high-speed direction
    DirectionConfig mixed_high_dir_{"y", 1.5708f};

    // keycode -> direction config
    std::unordered_map<int, DirectionConfig> keycode_direction_map_ = {
        {KEY_1, {"x",        0.0f}},
        {KEY_2, {"y",        1.5708f}},
        {KEY_3, {"diagonal", 0.7854f}},
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

    // Mixed mode high-speed phase durations
    float run_duration_mixed_high_x_    = 500.0f;
    float run_duration_mixed_high_y_    = 400.0f;
    float run_duration_mixed_high_diag_ = 600.0f;

    // ── Height control ──
    float height_min_      = 0.32f;
    float height_max_      = 0.425f;
    float height_step_     = 0.005f;
    float height_nominal_  = 0.40f;

    // ── Yaw hold tracking ──
    double last_a_seen_time_ = -1.0;
    double last_d_seen_time_ = -1.0;
    static constexpr double yaw_timeout_ms_ = 200.0;
    float yaw_magnitude_ = 0.5f;

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

    bool isKeyPressed(int keycode) const
    {
        std::lock_guard<std::mutex> lock(keys_mutex_);
        return pressed_keys_.count(keycode);
    }

    // ── libevdev keyboard detection ──
    bool init_all_keyboards()
    {
        DIR* dir = opendir("/dev/input");
        if (!dir) return false;

        struct dirent* ent;
        while ((ent = readdir(dir)) != nullptr) {
            if (strncmp(ent->d_name, "event", 5) != 0) continue;

            std::string path = "/dev/input/" + std::string(ent->d_name);
            int fd = open(path.c_str(), O_RDONLY | O_NONBLOCK);
            if (fd < 0) continue;

            struct libevdev* dev = nullptr;
            if (libevdev_new_from_fd(fd, &dev) < 0) {
                close(fd);
                continue;
            }

            if (!libevdev_has_event_type(dev, EV_KEY) ||
                !libevdev_has_event_code(dev, EV_KEY, KEY_A)) {
                libevdev_free(dev);
                close(fd);
                continue;
            }

            const char* name = libevdev_get_name(dev);
            std::string name_str = name ? name : "Unknown";

            if (name_str.find("Mouse") != std::string::npos ||
                name_str.find("Touchpad") != std::string::npos) {
                libevdev_free(dev);
                close(fd);
                continue;
            }

            keyboards_.push_back({fd, dev, name_str});
            std::cout << "[KB] Detected: " << name_str << " → " << path << std::endl;
        }
        closedir(dir);

        if (keyboards_.empty()) {
            std::cerr << "[FixedDirectionKeyboardInterface] ERROR: No keyboards found!" << std::endl;
            return false;
        }

        std::cout << "[FixedDirectionKeyboardInterface] Initialized " << keyboards_.size() << " keyboard(s)." << std::endl;
        return true;
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
                if (dir == "x")             duration_ms = run_duration_mixed_high_x_;
                else if (dir == "y")        duration_ms = run_duration_mixed_high_y_;
                else                        duration_ms = run_duration_mixed_high_diag_;
            } else {
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
        if (run_cnt_ % 100 != 0) return;

        const char* robot_state = "Unknown";
        switch (msfb_->GetCurrentState()) {
            case RobotMotionState::WaitingForStand: robot_state = "Waiting";    break;
            case RobotMotionState::StandingUp:      robot_state = "StandingUp"; break;
            case RobotMotionState::JointDamping:    robot_state = "Damping";    break;
            case RobotMotionState::RLControlMode:   robot_state = "RLControl";  break;
        }

        const char* speed_labels[] = {"LOW", "HIGH", "MIXED"};
        const char* speed_str = speed_labels[speed_mode_];

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

    void process_direction_keycode(int keycode)
    {
        auto it = keycode_direction_map_.find(keycode);
        if (it != keycode_direction_map_.end()) {
            std::lock_guard<std::mutex> lock(keys_mutex_);
            current_dir_cfg_ = it->second;
            has_direction_ = true;
            if (current_velocity_ > 0.0f || is_accelerating_) {
                current_velocity_ = 0.0f;
                is_accelerating_ = true;
            }
            usr_cmd_->forward_vel_scale = 0.0f;
            usr_cmd_->side_vel_scale    = 0.0f;
            std::cout << "[DIRECTION] Switched to " << current_dir_cfg_.name << " direction\n" << std::flush;
        }
    }

    // ── stdin fallback when libevdev not available ──
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

    void keyboard_loop()
    {
        bool use_evdev = init_all_keyboards();

        if (use_evdev) {
            std::cout << "\n╔════════════════════════════════════════════════╗\n"
                      << "║  FIXED DIRECTION KEYBOARD TELEOP (libevdev)    ║\n"
                      << "╚════════════════════════════════════════════════╝\n";
        } else {
            std::cout << "\n╔════════════════════════════════════════════════╗\n"
                      << "║  FIXED DIRECTION KEYBOARD TELEOP (stdin fallback)║\n"
                      << "╚════════════════════════════════════════════════╝\n"
                      << "  [WARN] libevdev unavailable — using stdin.\n"
                      << "  Keys may conflict with monitor_ui.py curses.\n";
            setup_raw_mode();
        }
        std::cout << "  Move:   hold Q (fixed forward, stop on release)\n"
                  << "  Yaw:    hold A (left) / D (right)\n"
                  << "  Height: W (up) / S (down)  0.32 ~ 0.425 m\n"
                  << "  Speed:  T (cycle: low / high / mixed)\n"
                  << "  Dir:    1/2/3 (x / y / diagonal)\n"
                  << "  Mode:   R (damping)  Z (stand)  C (control)\n\n" << std::flush;

        struct input_event ev;
        char ch;

        while (running_) {
            bool got_event = false;

            if (use_evdev) {
                // ── libevdev path ──
                for (auto& kb : keyboards_) {
                    while (libevdev_next_event(kb.dev, LIBEVDEV_READ_FLAG_NORMAL, &ev) == 0) {
                        got_event = true;
                        if (ev.type != EV_KEY) continue;
                        process_evdev_event(ev);
                    }
                }
            } else {
                // ── stdin fallback path ──
                while (read(STDIN_FILENO, &ch, 1) == 1) {
                    got_event = true;
                    process_stdin_char(ch);
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
            }

            if (msfb_->GetCurrentState() == RobotMotionState::RLControlMode)
                update_fixed_direction_movement();

            usr_cmd_->time_stamp = GetCurrentTimeStamp();

            print_status();
            run_cnt_++;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        if (!use_evdev) restore_terminal();
    }

    // ── Process a libevdev key event ──
    void process_evdev_event(const struct input_event& ev)
    {
        bool was_pressed = pressed_keys_.count(ev.code);

        {
            std::lock_guard<std::mutex> lock(keys_mutex_);
            if (ev.value == 1 || ev.value == 2) {
                pressed_keys_.insert(ev.code);
            } else if (ev.value == 0) {
                pressed_keys_.erase(ev.code);
            }
        }

        if (ev.value == 1) {
            if (ev.code == KEY_R) {
                usr_cmd_->target_mode = uint8_t(RobotMotionState::JointDamping);
                std::cout << "[MODE] Joint Damping\n";
            }
            else if (ev.code == KEY_Z && msfb_->GetCurrentState() == RobotMotionState::WaitingForStand) {
                usr_cmd_->target_mode = uint8_t(RobotMotionState::StandingUp);
                std::cout << "[MODE] Standing Up\n";
            }
            else if (ev.code == KEY_C && msfb_->GetCurrentState() == RobotMotionState::StandingUp) {
                usr_cmd_->target_mode = uint8_t(RobotMotionState::RLControlMode);
                std::cout << "[MODE] RL Control\n";
            }
        }

        if (ev.value == 1) {
            if (ev.code == KEY_1 || ev.code == KEY_2 || ev.code == KEY_3) {
                process_direction_keycode(ev.code);
            }
        }

        if (ev.code == KEY_Q) {
            if ((ev.value == 1 || ev.value == 2) &&
                msfb_->GetCurrentState() == RobotMotionState::RLControlMode) {
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
                          << "\n" << std::flush;
            }
            if (was_pressed && ev.value == 0) {
                std::lock_guard<std::mutex> lock(keys_mutex_);
                last_q_seen_time_ = -1.0;
            }
        }

        if (ev.code == KEY_W && (ev.value == 1 || ev.value == 2)) {
            usr_cmd_->height_command += height_step_;
            if (usr_cmd_->height_command > height_max_) usr_cmd_->height_command = height_max_;
            if (usr_cmd_->height_command < height_min_) usr_cmd_->height_command = height_min_;
            std::cout << "[HEIGHT] " << usr_cmd_->height_command << " m\n" << std::flush;
        }
        if (ev.code == KEY_S && (ev.value == 1 || ev.value == 2)) {
            usr_cmd_->height_command -= height_step_;
            if (usr_cmd_->height_command < height_min_) usr_cmd_->height_command = height_min_;
            std::cout << "[HEIGHT] " << usr_cmd_->height_command << " m\n" << std::flush;
        }

        if (ev.code == KEY_A) {
            if (ev.value == 1 || ev.value == 2) last_a_seen_time_ = GetCurrentTimeStamp();
        }
        if (ev.code == KEY_D) {
            if (ev.value == 1 || ev.value == 2) last_d_seen_time_ = GetCurrentTimeStamp();
        }

        if (ev.code == KEY_T && ev.value == 1) {
            speed_mode_ = (speed_mode_ + 1) % 3;
            const char* mode_names[] = {"LOW SPEED", "HIGH SPEED", "MIXED"};
            usr_cmd_->reserved_scale = (speed_mode_ == 1) ? 1.0f : 0.0f;
            std::cout << "[SPEED] Mode: " << mode_names[speed_mode_] << "\n";
        }

        if (ev.code == KEY_ESC && ev.value == 1) {
            std::cout << "\n[KEYBOARD] ESC pressed → stopping.\n";
            running_ = false;
        }
    }

    // ── Process a stdin char (fallback) ──
    void process_stdin_char(char ch)
    {
        char k = std::tolower(static_cast<unsigned char>(ch));
        double now = GetCurrentTimeStamp();

        if (k == 'r') {
            usr_cmd_->target_mode = uint8_t(RobotMotionState::JointDamping);
            std::cout << "[MODE] Joint Damping\n" << std::flush;
            return;
        }
        if (k == 'z' && msfb_->GetCurrentState() == RobotMotionState::WaitingForStand) {
            usr_cmd_->target_mode = uint8_t(RobotMotionState::StandingUp);
            std::cout << "[MODE] Standing Up\n" << std::flush;
            return;
        }
        if (k == 'c' && msfb_->GetCurrentState() == RobotMotionState::StandingUp) {
            usr_cmd_->target_mode = uint8_t(RobotMotionState::RLControlMode);
            std::cout << "[MODE] RL Control\n" << std::flush;
            return;
        }

        if (k == '1' || k == '2' || k == '3') {
            int kc = (k == '1') ? KEY_1 : ((k == '2') ? KEY_2 : KEY_3);
            process_direction_keycode(kc);
            return;
        }

        if (k == 'q' && msfb_->GetCurrentState() == RobotMotionState::RLControlMode) {
            std::lock_guard<std::mutex> lock(keys_mutex_);
            current_motion_high_speed_ = (speed_mode_ == 1);
            current_motion_mixed_high_ = false;
            usr_cmd_->reserved_scale = current_motion_high_speed_ ? 1.0f : 0.0f;
            is_accelerating_ = true;
            q_hold_active_ = true;
            last_q_seen_time_ = now;
            if (!has_direction_) {
                current_dir_cfg_ = {"x", 0.0f};
                has_direction_ = true;
            }
            std::cout << "[MOVE] Hold Q: dir=" << current_dir_cfg_.name << "\n" << std::flush;
            return;
        }

        if (k == 'w') {
            usr_cmd_->height_command += height_step_;
            if (usr_cmd_->height_command > height_max_) usr_cmd_->height_command = height_max_;
            std::cout << "[HEIGHT] " << usr_cmd_->height_command << " m\n" << std::flush;
            return;
        }
        if (k == 's') {
            usr_cmd_->height_command -= height_step_;
            if (usr_cmd_->height_command < height_min_) usr_cmd_->height_command = height_min_;
            std::cout << "[HEIGHT] " << usr_cmd_->height_command << " m\n" << std::flush;
            return;
        }

        if (k == 'a') last_a_seen_time_ = now;
        if (k == 'd') last_d_seen_time_ = now;

        if (k == 't') {
            speed_mode_ = (speed_mode_ + 1) % 3;
            const char* mode_names[] = {"LOW SPEED", "HIGH SPEED", "MIXED"};
            usr_cmd_->reserved_scale = (speed_mode_ == 1) ? 1.0f : 0.0f;
            std::cout << "[SPEED] Mode: " << mode_names[speed_mode_] << "\n" << std::flush;
        }
    }

public:
    FixedDirectionKeyboardInterface(RobotName robot_name) : UserCommandInterface(robot_name)
    {
        // Load config from JSON (next to binary)
        namespace fs = std::filesystem;
        char exe_buf[4096] = {};
        ssize_t len = readlink("/proc/self/exe", exe_buf, sizeof(exe_buf) - 1);
        std::string exe_path;
        if (len != -1) {
            exe_buf[len] = '\0';
            exe_path = std::string(exe_buf);
        }
        auto config_path = fs::path(exe_path).parent_path() / "teleop_config.json";
        std::ifstream f(config_path);
        if (f.is_open()) {
            nlohmann::json j;
            f >> j;
            if (j.contains("max_velocity")) MAX_VELOCITY = j["max_velocity"].get<float>();
            if (j.contains("acceleration"))  ACCELERATION = j["acceleration"].get<float>();

            if (j.contains("low_speed")) {
                auto& ls = j["low_speed"];
                if (ls.contains("max_velocity"))    MAX_VELOCITY_LOW  = ls["max_velocity"].get<float>();
                if (ls.contains("acceleration"))    ACCELERATION_LOW  = ls["acceleration"].get<float>();
                if (ls.contains("duration_x"))      run_duration_low_x_    = ls["duration_x"].get<float>()    * 1000.f;
                if (ls.contains("duration_y"))      run_duration_low_y_    = ls["duration_y"].get<float>()    * 1000.f;
                if (ls.contains("duration_diagonal")) run_duration_low_diag_ = ls["duration_diagonal"].get<float>() * 1000.f;
            }
            if (j.contains("high_speed")) {
                auto& hs = j["high_speed"];
                if (hs.contains("max_velocity"))    MAX_VELOCITY_HIGH  = hs["max_velocity"].get<float>();
                if (hs.contains("acceleration"))    ACCELERATION_HIGH  = hs["acceleration"].get<float>();
                if (hs.contains("duration_x"))      run_duration_high_x_    = hs["duration_x"].get<float>()    * 1000.f;
                if (hs.contains("duration_y"))      run_duration_high_y_    = hs["duration_y"].get<float>()    * 1000.f;
                if (hs.contains("duration_diagonal")) run_duration_high_diag_ = hs["duration_diagonal"].get<float>() * 1000.f;
            }
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
            if (j.contains("key_directions")) {
                for (auto& [key_str, val] : j["key_directions"].items()) {
                    if (key_str.size() == 1) {
                        char k = key_str[0];
                        DirectionConfig dc;
                        dc.name  = val.value("name",  "x");
                        dc.angle = val.value("angle", 0.0f);
                        // Map char to keycode
                        int kc = KEY_1;
                        if (k == '1') kc = KEY_1;
                        else if (k == '2') kc = KEY_2;
                        else if (k == '3') kc = KEY_3;
                        keycode_direction_map_[kc] = dc;
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
        auto it = keycode_direction_map_.find(KEY_1);
        if (it != keycode_direction_map_.end()) {
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

        for (auto& kb : keyboards_) {
            libevdev_free(kb.dev);
            close(kb.fd);
        }
        keyboards_.clear();

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
