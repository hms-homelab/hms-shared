#include <catch2/catch_test_macros.hpp>
#include "mqtt_client.h"

#include <chrono>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Needs a real broker: HMS_MQTT_TEST_BROKER (host), optional
// HMS_MQTT_TEST_PORT / HMS_MQTT_TEST_USER / HMS_MQTT_TEST_PASS.
// Skipped when the broker is not configured.

namespace {

std::string env(const char* name, const char* fallback = "") {
    const char* v = std::getenv(name);
    return v ? v : fallback;
}

hms::MqttConfig brokerConfig() {
    hms::MqttConfig c;
    c.broker = env("HMS_MQTT_TEST_BROKER");
    c.port = std::stoi(env("HMS_MQTT_TEST_PORT", "1883"));
    c.username = env("HMS_MQTT_TEST_USER");
    c.password = env("HMS_MQTT_TEST_PASS");
    return c;
}

// Records every payload seen on one topic, retained ones included.
struct Observer {
    std::mutex m;
    std::vector<std::string> seen;

    void add(const std::string& p) {
        std::lock_guard lock(m);
        seen.push_back(p);
    }
    std::string last() {
        std::lock_guard lock(m);
        return seen.empty() ? "" : seen.back();
    }
    bool sawAfter(size_t from, const std::string& p) {
        std::lock_guard lock(m);
        for (size_t i = from; i < seen.size(); ++i)
            if (seen[i] == p) return true;
        return false;
    }
    size_t size() {
        std::lock_guard lock(m);
        return seen.size();
    }
};

template <typename Pred>
bool waitFor(Pred pred, std::chrono::seconds limit) {
    auto end = std::chrono::steady_clock::now() + limit;
    while (std::chrono::steady_clock::now() < end) {
        if (pred()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return pred();
}

}  // namespace

TEST_CASE("Status returns to online after the broker drops the client", "[mqtt][broker]") {
    if (env("HMS_MQTT_TEST_BROKER").empty()) SKIP("HMS_MQTT_TEST_BROKER not set");

    const std::string id = "hms_shared_test_" + std::to_string(
        std::chrono::system_clock::now().time_since_epoch().count());
    const std::string status = id + "/status";

    Observer obs;
    auto watcherCfg = brokerConfig();
    watcherCfg.client_id = id + "_watch";
    hms::MqttClient watcher(watcherCfg);
    REQUIRE(watcher.connect());
    watcher.subscribe(status, [&](const std::string&, const std::string& p) { obs.add(p); });

    {
        auto cfg = brokerConfig();
        cfg.client_id = id;
        cfg.topic_prefix = id;
        hms::MqttClient svc(cfg);
        REQUIRE(svc.connect());

        // The first connect announces itself: no caller publishes "online".
        REQUIRE(waitFor([&] { return obs.last() == "online"; }, std::chrono::seconds(5)));

        // Take the client id over: the broker drops svc without a DISCONNECT,
        // which publishes its retained LWT "offline", the same as a broker restart.
        size_t mark = obs.size();
        {
            auto kickCfg = brokerConfig();
            kickCfg.client_id = id;
            hms::MqttClient kicker(kickCfg);
            REQUIRE(kicker.connect());
        }
        REQUIRE(waitFor([&] { return obs.sawAfter(mark, "offline"); }, std::chrono::seconds(5)));

        // svc reconnects on its own and must replace the retained "offline".
        mark = obs.size();
        REQUIRE(waitFor([&] { return svc.isConnected() && obs.sawAfter(mark, "online"); },
                        std::chrono::seconds(15)));
        CHECK(obs.last() == "online");
    }

    // Leave no retained message behind on the broker.
    watcher.publish(status, "", 1, true);
}
