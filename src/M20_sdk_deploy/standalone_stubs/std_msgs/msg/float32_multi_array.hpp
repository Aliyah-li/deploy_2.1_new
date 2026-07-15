#pragma once

#include <memory>
#include <vector>

namespace std_msgs {
namespace msg {

struct Float32MultiArray {
    using SharedPtr = std::shared_ptr<Float32MultiArray>;
    std::vector<float> data;
};

}  // namespace msg
}  // namespace std_msgs
