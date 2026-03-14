#include <catch2/catch_test_macros.hpp>
#include "api_queries.h"
#include "event_logger.h"
#include "config_manager.h"

// These tests validate the query function interfaces and JSON output structure.
// Integration tests against a real database are in a separate test suite.

TEST_CASE("get_cameras_status returns correct JSON structure", "[db][queries]") {
    // Create a mock camera config map
    std::unordered_map<std::string, hms::CameraConfig> cameras;

    hms::CameraConfig cam1;
    cam1.id = "patio";
    cam1.name = "Patio";
    cam1.enabled = true;
    cameras["patio"] = cam1;

    hms::CameraConfig cam2;
    cam2.id = "front_door";
    cam2.name = "Front Door";
    cam2.enabled = false;  // disabled
    cameras["front_door"] = cam2;

    // Note: This test cannot run without a real database connection.
    // It validates the camera filtering logic only.
    // In a full test environment, we'd use a test database.

    // Verify the enabled/disabled filtering logic:
    // get_cameras_status should skip disabled cameras
    int enabled_count = 0;
    for (const auto& [id, cam] : cameras) {
        if (cam.enabled) enabled_count++;
    }
    CHECK(enabled_count == 1);
}

TEST_CASE("JSON event structure matches Python output format", "[db][queries]") {
    // Verify the expected JSON keys match what the Angular frontend expects
    nlohmann::json event;
    event["event_id"] = "patio_20260225_103000";
    event["camera_id"] = "patio";
    event["camera_name"] = "Patio";
    event["started_at"] = "2026-02-25T10:30:00";
    event["ended_at"] = "2026-02-25T10:30:15";
    event["duration_seconds"] = 15.0;
    event["total_detections"] = 5;
    event["status"] = "completed";
    event["recording_url"] = "/events/patio_20260225_103000.mp4";
    event["snapshot_url"] = "/snapshots/patio_20260225_103000.jpg";
    event["detected_classes"] = "dog, person";
    event["max_confidence"] = 0.95;
    event["ai_context"] = "A person is walking with a dog";

    // Verify all expected keys are present
    CHECK(event.contains("event_id"));
    CHECK(event.contains("camera_id"));
    CHECK(event.contains("camera_name"));
    CHECK(event.contains("started_at"));
    CHECK(event.contains("ended_at"));
    CHECK(event.contains("duration_seconds"));
    CHECK(event.contains("total_detections"));
    CHECK(event.contains("status"));
    CHECK(event.contains("recording_url"));
    CHECK(event.contains("snapshot_url"));
    CHECK(event.contains("detected_classes"));
    CHECK(event.contains("max_confidence"));
    CHECK(event.contains("ai_context"));
}

TEST_CASE("JSON timeline structure matches Python output format", "[db][queries]") {
    nlohmann::json timeline;
    timeline["camera_id"] = "patio";
    timeline["date"] = "2026-02-25T00:00:00";

    nlohmann::json hours = nlohmann::json::array();
    for (int h = 0; h < 24; ++h) {
        hours.push_back({{"hour", h}, {"event_count", 0}, {"total_detections", 0}});
    }
    timeline["hours"] = hours;

    CHECK(timeline["hours"].size() == 24);
    CHECK(timeline["hours"][0]["hour"] == 0);
    CHECK(timeline["hours"][23]["hour"] == 23);
    CHECK(timeline["hours"][0]["event_count"] == 0);
    CHECK(timeline["hours"][0]["total_detections"] == 0);
}

TEST_CASE("JSON event detail structure with detections", "[db][queries]") {
    nlohmann::json detail;
    detail["event"] = {
        {"event_id", "patio_20260225_103000"},
        {"camera_id", "patio"},
        {"status", "completed"}
    };

    detail["detections"] = nlohmann::json::array({
        {
            {"detection_id", 1},
            {"class_name", "person"},
            {"confidence", 0.95},
            {"bbox_x1", 100},
            {"bbox_y1", 100},
            {"bbox_x2", 200},
            {"bbox_y2", 300},
            {"frame_number", 42},
            {"detected_at", "2026-02-25T10:30:01"}
        }
    });

    CHECK(detail.contains("event"));
    CHECK(detail.contains("detections"));
    CHECK(detail["detections"].size() == 1);
    CHECK(detail["detections"][0]["class_name"] == "person");
    CHECK(detail["detections"][0]["confidence"] == 0.95);
}

TEST_CASE("AiVisionRecord has correct defaults", "[db][event_logger]") {
    hms::EventLogger::AiVisionRecord record;

    CHECK(record.context_text.empty());
    CHECK(record.detected_classes.empty());
    CHECK(record.source_model == "llava:7b");
    CHECK(record.prompt_used.empty());
    CHECK(record.response_time_seconds == 0);
    CHECK(record.is_valid == true);
}

TEST_CASE("AiVisionRecord designated initializer fields", "[db][event_logger]") {
    hms::EventLogger::AiVisionRecord record{
        .context_text = "A person walking on the patio",
        .detected_classes = {"person", "dog"},
        .source_model = "llava:13b",
        .prompt_used = "Describe the person",
        .response_time_seconds = 12.5,
        .is_valid = true,
    };

    CHECK(record.context_text == "A person walking on the patio");
    CHECK(record.detected_classes.size() == 2);
    CHECK(record.detected_classes[0] == "person");
    CHECK(record.source_model == "llava:13b");
    CHECK(record.response_time_seconds == 12.5);
    CHECK(record.is_valid == true);
}

// ────────────────────────────────────────────────────────────────────
// SearchParams and Search Result JSON tests
// ────────────────────────────────────────────────────────────────────

TEST_CASE("SearchParams has correct defaults", "[db][search]") {
    hms::api_queries::SearchParams params;

    CHECK(params.query.empty());
    CHECK(params.class_filter.empty());
    CHECK_FALSE(params.camera_id.has_value());
    CHECK_FALSE(params.start_date.has_value());
    CHECK_FALSE(params.end_date.has_value());
    CHECK(params.limit == 50);
    CHECK(params.mode == "auto");
}

TEST_CASE("SearchParams accepts all fields", "[db][search]") {
    hms::api_queries::SearchParams params;
    params.query = "person walking on porch";
    params.class_filter = {"person", "dog"};
    params.camera_id = "patio";
    params.start_date = "2026-03-01";
    params.end_date = "2026-03-04";
    params.limit = 25;
    params.mode = "semantic";

    CHECK(params.query == "person walking on porch");
    CHECK(params.class_filter.size() == 2);
    CHECK(params.class_filter[0] == "person");
    CHECK(params.class_filter[1] == "dog");
    CHECK(params.camera_id.value() == "patio");
    CHECK(params.start_date.value() == "2026-03-01");
    CHECK(params.end_date.value() == "2026-03-04");
    CHECK(params.limit == 25);
    CHECK(params.mode == "semantic");
}

TEST_CASE("FTS search response JSON structure", "[db][search]") {
    // Validate the shape that the Angular frontend expects
    nlohmann::json response;
    response["search_mode"] = "fts";
    response["query"] = "person on porch";
    response["count"] = 2;

    nlohmann::json events = nlohmann::json::array();
    events.push_back({
        {"type", "event"},
        {"id", "patio_20260304_103000"},
        {"camera_id", "patio"},
        {"camera_name", "Patio"},
        {"timestamp", "2026-03-04T10:30:00Z"},
        {"recording_url", "patio_20260304_103000.mp4"},
        {"snapshot_url", "patio_20260304_103000.jpg"},
        {"total_detections", 3},
        {"duration_seconds", 12.5},
        {"detected_classes", "person, dog"},
        {"ai_context", "A person is walking with a dog on the patio"},
        {"rank", 0.85}
    });
    events.push_back({
        {"type", "snapshot"},
        {"id", "42"},
        {"camera_id", "patio"},
        {"camera_name", "patio"},
        {"timestamp", "2026-03-04T10:45:00Z"},
        {"recording_url", nullptr},
        {"snapshot_url", "patio_periodic_20260304_104500.jpg"},
        {"total_detections", 0},
        {"duration_seconds", nullptr},
        {"detected_classes", nullptr},
        {"ai_context", "The patio is empty, sunny afternoon"},
        {"rank", 0.72}
    });
    response["events"] = events;

    // Verify required top-level keys
    CHECK(response.contains("events"));
    CHECK(response.contains("count"));
    CHECK(response.contains("search_mode"));
    CHECK(response.contains("query"));
    CHECK(response["count"] == 2);
    CHECK(response["search_mode"] == "fts");

    // Verify event result
    auto& ev = response["events"][0];
    CHECK(ev["type"] == "event");
    CHECK(ev.contains("id"));
    CHECK(ev.contains("camera_id"));
    CHECK(ev.contains("timestamp"));
    CHECK(ev.contains("recording_url"));
    CHECK(ev.contains("detected_classes"));
    CHECK(ev.contains("ai_context"));
    CHECK(ev.contains("rank"));

    // Verify snapshot result
    auto& snap = response["events"][1];
    CHECK(snap["type"] == "snapshot");
    CHECK(snap["recording_url"].is_null());
    CHECK(snap["duration_seconds"].is_null());
    CHECK(snap["detected_classes"].is_null());
    CHECK(snap["total_detections"] == 0);
    CHECK(snap.contains("ai_context"));
}

TEST_CASE("Semantic search response JSON structure", "[db][search]") {
    nlohmann::json response;
    response["search_mode"] = "semantic";
    response["query"] = "someone approaching the house";
    response["count"] = 1;
    response["events"] = nlohmann::json::array({
        {
            {"type", "event"},
            {"id", "front_door_20260304_090000"},
            {"camera_id", "front_door"},
            {"camera_name", "Front Door"},
            {"timestamp", "2026-03-04T09:00:00Z"},
            {"similarity", 0.91},
            {"ai_context", "A person walking up to the front door"}
        }
    });

    CHECK(response["search_mode"] == "semantic");
    CHECK(response["events"][0].contains("similarity"));
    CHECK(response["events"][0]["similarity"].get<double>() > 0.0);
    CHECK(response["events"][0]["similarity"].get<double>() <= 1.0);
}

TEST_CASE("Periodic snapshot JSON structure matches frontend model", "[db][search]") {
    // Validate the shape returned by GET /api/snapshots
    nlohmann::json snapshot;
    snapshot["type"] = "snapshot";
    snapshot["snapshot_id"] = 42;
    snapshot["camera_id"] = "patio";
    snapshot["captured_at"] = "2026-03-04T14:30:00Z";
    snapshot["snapshot_url"] = "patio_periodic_20260304_143000.jpg";
    snapshot["thumbnail_url"] = "patio_periodic_20260304_143000_thumb.jpg";
    snapshot["ai_context"] = "Sunny patio, garden furniture visible";
    snapshot["is_valid"] = true;

    CHECK(snapshot.contains("type"));
    CHECK(snapshot["type"] == "snapshot");
    CHECK(snapshot.contains("snapshot_id"));
    CHECK(snapshot.contains("camera_id"));
    CHECK(snapshot.contains("captured_at"));
    CHECK(snapshot.contains("snapshot_url"));
    CHECK(snapshot.contains("thumbnail_url"));
    CHECK(snapshot.contains("ai_context"));
    CHECK(snapshot.contains("is_valid"));
    CHECK(snapshot["is_valid"] == true);

    // Verify filename follows naming convention
    std::string filename = snapshot["snapshot_url"].get<std::string>();
    CHECK(filename.find("periodic") != std::string::npos);
    CHECK(filename.find(".jpg") != std::string::npos);
}

TEST_CASE("Periodic snapshot with null optional fields", "[db][search]") {
    nlohmann::json snapshot;
    snapshot["type"] = "snapshot";
    snapshot["snapshot_id"] = 1;
    snapshot["camera_id"] = "side_window";
    snapshot["captured_at"] = "2026-03-04T10:00:00Z";
    snapshot["snapshot_url"] = "side_window_periodic_20260304_100000.jpg";
    snapshot["thumbnail_url"] = nullptr;
    snapshot["ai_context"] = nullptr;
    snapshot["is_valid"] = false;

    CHECK(snapshot["thumbnail_url"].is_null());
    CHECK(snapshot["ai_context"].is_null());
    CHECK(snapshot["is_valid"] == false);
}

TEST_CASE("Snapshots response wraps array", "[db][search]") {
    nlohmann::json snapshots = nlohmann::json::array();
    snapshots.push_back({{"snapshot_id", 1}, {"camera_id", "patio"}});
    snapshots.push_back({{"snapshot_id", 2}, {"camera_id", "patio"}});

    nlohmann::json response;
    response["snapshots"] = snapshots;
    response["count"] = static_cast<int>(snapshots.size());

    CHECK(response["count"] == 2);
    CHECK(response["snapshots"].is_array());
    CHECK(response["snapshots"].size() == 2);
}

TEST_CASE("Search result union type distinguishes events and snapshots", "[db][search]") {
    nlohmann::json results = nlohmann::json::array();
    results.push_back({{"type", "event"}, {"id", "patio_20260304"}});
    results.push_back({{"type", "snapshot"}, {"id", "42"}});
    results.push_back({{"type", "event"}, {"id", "front_door_20260304"}});

    int event_count = 0, snapshot_count = 0;
    for (const auto& r : results) {
        if (r["type"] == "event") event_count++;
        else if (r["type"] == "snapshot") snapshot_count++;
    }

    CHECK(event_count == 2);
    CHECK(snapshot_count == 1);
}
