#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <string>

namespace rclcpp {

inline void init(int, char**) {}
inline void shutdown() {}
inline bool ok() { return false; }

class Time {
public:
    Time() = default;

    template <typename StampT>
    explicit Time(const StampT& stamp)
        : seconds_(static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1.0e-9) {}

    double seconds() const { return seconds_; }

private:
    double seconds_ = 0.0;
};

template <typename MsgT>
class Publisher {
public:
    using SharedPtr = std::shared_ptr<Publisher<MsgT>>;
    void publish(const MsgT&) {}
};

template <typename MsgT>
class Subscription {
public:
    using SharedPtr = std::shared_ptr<Subscription<MsgT>>;
};

class Node : public std::enable_shared_from_this<Node> {
public:
    using SharedPtr = std::shared_ptr<Node>;

    explicit Node(const std::string& name) : name_(name) {}

    template <typename MsgT>
    typename Publisher<MsgT>::SharedPtr create_publisher(const std::string&, int) {
        return std::make_shared<Publisher<MsgT>>();
    }

    template <typename MsgT, typename CallbackT>
    typename Subscription<MsgT>::SharedPtr create_subscription(const std::string&, int, CallbackT&&) {
        return std::make_shared<Subscription<MsgT>>();
    }

private:
    std::string name_;
};

template <typename NodeT>
inline void spin_some(const std::shared_ptr<NodeT>&) {}

inline void sleep_for(const std::chrono::nanoseconds&) {}

template <typename Rep, typename Period>
inline void sleep_for(const std::chrono::duration<Rep, Period>&) {}

}  // namespace rclcpp
