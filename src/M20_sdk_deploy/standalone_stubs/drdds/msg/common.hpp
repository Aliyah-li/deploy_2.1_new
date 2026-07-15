#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace drdds::msg {

struct TimeStamp {
    int32_t sec = 0;
    uint32_t nanosec = 0;
};

struct MetaType {
    using SharedPtr = std::shared_ptr<MetaType>;
    uint64_t frame_id = 0;
    TimeStamp stamp;
};

struct JointData {
    std::array<char, 4> name{};
    uint16_t data_id = 0;
    uint16_t status_word = 0;
    float position = 0.0f;
    float torque = 0.0f;
    float velocity = 0.0f;
    float motion_temp = 0.0f;
    float driver_temp = 0.0f;
};

struct JointDataCmd {
    std::array<char, 4> name{};
    uint16_t data_id = 0;
    uint16_t control_word = 0;
    float position = 0.0f;
    float torque = 0.0f;
    float velocity = 0.0f;
    float kp = 0.0f;
    float kd = 0.0f;
};

struct JointsDataValue {
    std::array<JointData, 16> joints_data{};
};

struct JointsDataCmdValue {
    std::array<JointDataCmd, 16> joints_data{};
};

struct JointsData {
    using SharedPtr = std::shared_ptr<JointsData>;
    MetaType header;
    JointsDataValue data;
};

struct JointsDataCmd {
    using SharedPtr = std::shared_ptr<JointsDataCmd>;
    MetaType header;
    JointsDataCmdValue data;
};

struct ImuDataValue {
    float roll = 0.0f;
    float pitch = 0.0f;
    float yaw = 0.0f;
    float omega_x = 0.0f;
    float omega_y = 0.0f;
    float omega_z = 0.0f;
    float acc_x = 0.0f;
    float acc_y = 0.0f;
    float acc_z = 0.0f;
};

struct ImuData {
    using SharedPtr = std::shared_ptr<ImuData>;
    MetaType header;
    ImuDataValue data;
};

struct BatteryDataValue {
    uint16_t voltage = 0;
    int16_t current = 0;
    uint16_t remaining_capacity = 0;
    uint16_t nominal_capacity = 0;
    uint16_t cycles = 0;
    uint16_t production_date = 0;
    uint16_t balanced_low = 0;
    uint16_t balanced_high = 0;
    uint16_t protected_state = 0;
    uint8_t software_version = 0;
    uint8_t battery_level = 0;
    uint8_t mos_state = 0;
    uint8_t battery_quantity = 0;
    uint8_t battery_ntc = 0;
    std::vector<float> battery_temperature;
    std::array<char, 32> battery_serialnum{};
};

struct BatteryData {
    using SharedPtr = std::shared_ptr<BatteryData>;
    MetaType header;
    std::vector<BatteryDataValue> data = std::vector<BatteryDataValue>(2);
};

}  // namespace drdds::msg
