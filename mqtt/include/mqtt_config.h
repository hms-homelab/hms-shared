#pragma once

#include <string>

namespace hms {

/// MQTT connection configuration (standalone, no yaml-cpp dependency)
struct MqttConfig {
    std::string broker = "localhost";
    int port = 1883;
    std::string username;
    std::string password;
    std::string client_id;          // Empty = auto-generate from PID
    std::string topic_prefix;       // Empty = no LWT status topic
    int qos = 1;
};

}  // namespace hms
