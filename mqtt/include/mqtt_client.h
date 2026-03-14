#pragma once

#include "mqtt_config.h"

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <mqtt/async_client.h>

namespace hms {

/// Thread-safe Paho MQTT async_client wrapper with auto-reconnect.
/// Pattern: fire-and-forget publish (no token->wait()), batch subscribe,
/// callback-based message dispatch with wildcard matching.
class MqttClient : public mqtt::callback {
public:
    using MessageCallback = std::function<void(const std::string& topic,
                                               const std::string& payload)>;

    explicit MqttClient(const MqttConfig& config);
    ~MqttClient() override;

    MqttClient(const MqttClient&) = delete;
    MqttClient& operator=(const MqttClient&) = delete;

    /// Connect to broker (blocking, with timeout). Returns true on success.
    bool connect();

    /// Disconnect gracefully
    void disconnect();

    /// Publish to a topic. Returns true if the message was handed to Paho.
    bool publish(const std::string& topic, const std::string& payload,
                 int qos = 1, bool retain = false);

    /// Subscribe to a single topic with a callback.
    bool subscribe(const std::string& topic, MessageCallback callback, int qos = 1);

    /// Subscribe to multiple topics with a single callback (batch).
    void subscribe(const std::vector<std::string>& topics,
                   MessageCallback callback, int qos = 1);

    /// Thread-safe connection status check
    bool isConnected() const;

    /// Topic prefix from config
    const std::string& topicPrefix() const { return config_.topic_prefix; }

private:
    // mqtt::callback overrides (called on Paho's internal thread)
    void connected(const std::string& cause) override;
    void connection_lost(const std::string& cause) override;
    void message_arrived(mqtt::const_message_ptr msg) override;

    static bool topicMatches(const std::string& pattern, const std::string& topic);

    MqttConfig config_;
    std::unique_ptr<mqtt::async_client> client_;
    mqtt::connect_options conn_opts_;

    mutable std::recursive_mutex mutex_;
    std::map<std::string, MessageCallback> subscriptions_;

    struct PendingSub {
        std::vector<std::string> topics;
        int qos;
    };
    std::vector<PendingSub> pending_subs_;
};

}  // namespace hms
