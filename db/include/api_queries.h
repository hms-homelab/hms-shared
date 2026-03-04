#pragma once

#include <string>
#include <vector>
#include <optional>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include "db_pool.h"
#include "config_manager.h"

namespace yolo {

using json = nlohmann::json;

namespace api_queries {

/// Query all detection events with optional filters.
/// Ports: api_queries.py:get_all_events()
json get_all_events(
    DbPool& db,
    const std::optional<std::string>& start_date = std::nullopt,
    const std::optional<std::string>& end_date = std::nullopt,
    const std::optional<std::string>& camera_id = std::nullopt,
    int limit = 100
);

/// Query single event with all detection details.
/// Ports: api_queries.py:get_event_detail()
json get_event_detail(DbPool& db, const std::string& event_id);

/// Get aggregated timeline data for a specific camera and date.
/// Returns hourly event counts for timeline rendering.
/// Ports: api_queries.py:get_timeline_data()
json get_timeline_data(
    DbPool& db,
    const std::string& camera_id,
    const std::string& date  // "YYYY-MM-DD"
);

/// Get the timestamp of the most recent event for a camera.
/// Ports: api_queries.py:get_camera_last_event()
std::optional<std::string> get_camera_last_event(
    DbPool& db,
    const std::string& camera_id
);

/// Get camera status list with last event times.
/// Used by GET /api/cameras/status
json get_cameras_status(
    DbPool& db,
    const std::unordered_map<std::string, CameraConfig>& cameras
);

// --- Search ---

/// Parameters for full-text + semantic search
struct SearchParams {
    std::string query;
    std::vector<std::string> class_filter;
    std::optional<std::string> camera_id;
    std::optional<std::string> start_date, end_date;
    int limit = 50;
    std::string mode = "auto";  // "auto" | "fts" | "semantic"
};

/// Full-text search over ai_vision_context + periodic_snapshots
json search_events_fts(DbPool& db, const SearchParams& params);

/// Semantic (vector) search — takes pre-computed query embedding
json search_events_semantic(DbPool& db, const SearchParams& params,
                            const std::vector<float>& query_embedding);

/// Get periodic snapshots for a camera + date (for timeline display)
json get_periodic_snapshots(DbPool& db, const std::string& camera_id,
                            const std::string& date);

/// Insert a periodic snapshot record
void insert_periodic_snapshot(DbPool& db,
                              const std::string& camera_id,
                              const std::string& snapshot_filename,
                              const std::string& thumbnail_filename,
                              const std::string& context_text,
                              const std::vector<float>& embedding,
                              const std::string& source_model,
                              bool is_valid);

} // namespace api_queries
} // namespace yolo
