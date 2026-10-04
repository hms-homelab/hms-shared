#include <catch2/catch_test_macros.hpp>
#include "mqtt_client.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
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

// subscribe() held the client's lock across the Paho subscribe and its wait,
// while Paho's receive thread held Paho's lock to deliver a message into that
// same lock. A second subscribe made while the first one's retained messages
// were still arriving hung both threads for good. A service subscribing to a
// few topics at startup, one of them retained, is exactly that.
TEST_CASE("A subscribe while retained messages arrive does not deadlock", "[mqtt][broker]") {
    if (env("HMS_MQTT_TEST_BROKER").empty()) SKIP("HMS_MQTT_TEST_BROKER not set");

    const std::string id = "hms_shared_dl_" + std::to_string(
        std::chrono::system_clock::now().time_since_epoch().count());
    constexpr int kRetained = 200;

    auto seedCfg = brokerConfig();
    seedCfg.client_id = id + "_seed";
    hms::MqttClient seed(seedCfg);
    REQUIRE(seed.connect());
    for (int i = 0; i < kRetained; ++i)
        seed.publish(id + "/a/" + std::to_string(i), "x", 1, true);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    Observer obs;
    auto cfg = brokerConfig();
    cfg.client_id = id;
    auto client = std::make_unique<hms::MqttClient>(cfg);
    REQUIRE(client->connect());

    // On a thread with a deadline: a deadlock must fail the test, not hang CI.
    auto done = std::make_shared<std::atomic<bool>>(false);
    std::thread t([&, done] {
        for (int round = 0; round < 20; ++round) {
            client->subscribe(id + "/a/#", [&](const std::string&, const std::string& p) {
                obs.add(p);
            });
            client->subscribe(id + "/b/" + std::to_string(round),
                              [](const std::string&, const std::string&) {});
        }
        *done = true;
    });
    const bool finished = waitFor([&] { return done->load(); }, std::chrono::seconds(30));
    if (!finished) {
        t.detach();
        client.release();   // its threads are wedged; destroying it would hang too
        FAIL("subscribe deadlocked while retained messages were being delivered");
    }
    t.join();
    CHECK(waitFor([&] { return obs.size() >= static_cast<size_t>(kRetained); },
                  std::chrono::seconds(10)));

    for (int i = 0; i < kRetained; ++i)
        seed.publish(id + "/a/" + std::to_string(i), "", 1, true);
}
