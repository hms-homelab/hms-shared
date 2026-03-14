#include "api_queries.h"
#include "config_manager.h"
#include "time_utils.h"
#include <spdlog/spdlog.h>
#include <pqxx/pqxx>
#include <sstream>

namespace hms::api_queries {

namespace {

json field_or_null(const pqxx::row& row, int col) {
    if (row[col].is_null()) return nullptr;
    return row[col].c_str();
}

json field_as_double_or_null(const pqxx::row& row, int col) {
    if (row[col].is_null()) return nullptr;
    return row[col].as<double>();
}

json field_as_int_or_null(const pqxx::row& row, int col) {
    if (row[col].is_null()) return nullptr;
    return row[col].as<int>();
}

json timestamp_or_null(const pqxx::row& row, int col) {
    if (row[col].is_null()) return nullptr;
    return time_utils::pg_timestamp_to_iso8601(row[col].c_str());
}

} // anonymous namespace

json get_all_events(
    DbPool& db,
    const std::optional<std::string>& start_date,
    const std::optional<std::string>& end_date,
    const std::optional<std::string>& camera_id,
    int limit
) {
    try {
        auto conn = db.acquire();
        pqxx::nontransaction txn(*conn);

        // Dynamic WHERE clause - use txn.quote() for safe embedding (no $1 params needed here)
        std::ostringstream query;
        query << R"(
            SELECT
                de.event_id,
                de.camera_id,
                de.camera_name,
                de.started_at,
                de.ended_at,
                de.duration_seconds,
                de.total_detections,
                de.status,
                de.recording_url,
                de.snapshot_url,
                STRING_AGG(DISTINCT d.class_name, ', ' ORDER BY d.class_name) as detected_classes,
                MAX(d.confidence) as max_confidence,
                avc.context_text as ai_context
            FROM detection_events de
            LEFT JOIN detections d ON de.event_id = d.event_id
            LEFT JOIN ai_vision_context avc ON de.event_id = avc.event_id
        )";

        std::vector<std::string> conditions;
        if (camera_id)  conditions.push_back("de.camera_id = " + txn.quote(*camera_id));
        if (start_date) conditions.push_back("de.started_at >= " + txn.quote(*start_date));
        if (end_date)   conditions.push_back("de.started_at < " + txn.quote(*end_date));

        if (!conditions.empty()) {
            query << " WHERE ";
            for (size_t i = 0; i < conditions.size(); ++i) {
                if (i > 0) query << " AND ";
                query << conditions[i];
            }
        }

        query << R"(
            GROUP BY de.event_id, de.camera_id, de.camera_name, de.started_at,
                     de.ended_at, de.duration_seconds, de.total_detections,
                     de.status, de.recording_url, de.snapshot_url, avc.context_text
            ORDER BY de.started_at DESC
            LIMIT )" << limit;

        auto result = txn.exec(query.str());

        json events = json::array();
        for (const auto& row : result) {
            json event;
            event["event_id"]        = field_or_null(row, 0);
            event["camera_id"]       = field_or_null(row, 1);
            event["camera_name"]     = field_or_null(row, 2);
            event["started_at"]      = timestamp_or_null(row, 3);
            event["ended_at"]        = timestamp_or_null(row, 4);
            event["duration_seconds"]= field_as_double_or_null(row, 5);
            event["total_detections"]= field_as_int_or_null(row, 6);
            event["status"]          = field_or_null(row, 7);
            event["recording_url"]   = field_or_null(row, 8);
            event["snapshot_url"]    = field_or_null(row, 9);
            event["detected_classes"]= field_or_null(row, 10);
            event["max_confidence"]  = field_as_double_or_null(row, 11);
            event["ai_context"]      = field_or_null(row, 12);
            events.push_back(std::move(event));
        }

        return events;

    } catch (const std::exception& e) {
        spdlog::error("Error querying all events: {}", e.what());
        return json::array();
    }
}

json get_event_detail(DbPool& db, const std::string& event_id) {
    try {
        auto conn = db.acquire();
        pqxx::nontransaction txn(*conn);

        // libpqxx 7.x: use pqxx::params for parameterized queries
        auto event_result = txn.exec(R"(
            SELECT
                de.event_id,
                de.camera_id,
                de.camera_name,
                de.started_at,
                de.ended_at,
                de.duration_seconds,
                de.total_detections,
                de.status,
                de.recording_url,
                de.snapshot_url,
                avc.context_text as ai_context
            FROM detection_events de
            LEFT JOIN ai_vision_context avc ON de.event_id = avc.event_id
            WHERE de.event_id = $1
        )", pqxx::params{event_id});

        if (event_result.empty()) return nullptr;

        const auto& row = event_result[0];
        json event;
        event["event_id"]         = field_or_null(row, 0);
        event["camera_id"]        = field_or_null(row, 1);
        event["camera_name"]      = field_or_null(row, 2);
        event["started_at"]       = timestamp_or_null(row, 3);
        event["ended_at"]         = timestamp_or_null(row, 4);
        event["duration_seconds"] = field_as_double_or_null(row, 5);
        event["total_detections"] = field_as_int_or_null(row, 6);
        event["status"]           = field_or_null(row, 7);
        event["recording_url"]    = field_or_null(row, 8);
        event["snapshot_url"]     = field_or_null(row, 9);
        event["ai_context"]       = field_or_null(row, 10);

        auto det_result = txn.exec(R"(
            SELECT
                id as detection_id,
                class_name,
                confidence,
                bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                frame_number,
                detected_at
            FROM detections
            WHERE event_id = $1
            ORDER BY frame_number, confidence DESC
        )", pqxx::params{event_id});

        json detections = json::array();
        for (const auto& det : det_result) {
            detections.push_back({
                {"detection_id",  field_as_int_or_null(det, 0)},
                {"class_name",    field_or_null(det, 1)},
                {"confidence",    field_as_double_or_null(det, 2)},
                {"bbox_x1",       field_as_int_or_null(det, 3)},
                {"bbox_y1",       field_as_int_or_null(det, 4)},
                {"bbox_x2",       field_as_int_or_null(det, 5)},
                {"bbox_y2",       field_as_int_or_null(det, 6)},
                {"frame_number",  field_as_int_or_null(det, 7)},
                {"detected_at",   timestamp_or_null(det, 8)},
            });
        }

        return json{{"event", std::move(event)}, {"detections", std::move(detections)}};

    } catch (const std::exception& e) {
        spdlog::error("Error querying event detail for {}: {}", event_id, e.what());
        return nullptr;
    }
}

json get_timeline_data(
    DbPool& db,
    const std::string& camera_id,
    const std::string& date
) {
    auto make_empty_timeline = [&](const std::string& d) {
        json hours = json::array();
        for (int h = 0; h < 24; ++h)
            hours.push_back({{"hour", h}, {"event_count", 0}, {"total_detections", 0}});
        return json{{"camera_id", camera_id}, {"date", d}, {"hours", hours}};
    };

    try {
        auto date_tp = time_utils::from_date_string(date);
        if (!date_tp) {
            spdlog::error("Invalid date format for timeline: {}", date);
            return make_empty_timeline(date);
        }

        auto start     = time_utils::start_of_day(*date_tp);
        auto end       = time_utils::end_of_day(*date_tp);
        auto start_str = time_utils::to_iso8601(start);
        auto end_str   = time_utils::to_iso8601(end);

        auto conn = db.acquire();
        pqxx::nontransaction txn(*conn);

        auto result = txn.exec(R"(
            SELECT
                EXTRACT(HOUR FROM started_at) as hour,
                COUNT(*) as event_count,
                COALESCE(SUM(total_detections), 0) as total_detections
            FROM detection_events
            WHERE camera_id = $1
              AND started_at >= $2
              AND started_at < $3
              AND status = 'completed'
            GROUP BY EXTRACT(HOUR FROM started_at)
            ORDER BY hour
        )", pqxx::params{camera_id, start_str, end_str});

        std::unordered_map<int, std::pair<int,int>> hour_data;
        for (const auto& row : result) {
            int h = static_cast<int>(row[0].as<double>());
            hour_data[h] = {row[1].as<int>(), row[2].as<int>()};
        }

        json hours = json::array();
        for (int h = 0; h < 24; ++h) {
            auto it = hour_data.find(h);
            if (it != hour_data.end())
                hours.push_back({{"hour",h},{"event_count",it->second.first},
                                             {"total_detections",it->second.second}});
            else
                hours.push_back({{"hour",h},{"event_count",0},{"total_detections",0}});
        }

        return {{"camera_id", camera_id}, {"date", time_utils::to_iso8601(start)}, {"hours", hours}};

    } catch (const std::exception& e) {
        spdlog::error("Error querying timeline for {}: {}", camera_id, e.what());
        return make_empty_timeline(date);
    }
}

std::optional<std::string> get_camera_last_event(
    DbPool& db,
    const std::string& camera_id
) {
    try {
        auto conn = db.acquire();
        pqxx::nontransaction txn(*conn);

        auto result = txn.exec(R"(
            SELECT MAX(started_at)
            FROM detection_events
            WHERE camera_id = $1
        )", pqxx::params{camera_id});

        if (result.empty() || result[0][0].is_null()) return std::nullopt;
        return time_utils::pg_timestamp_to_iso8601(result[0][0].c_str());

    } catch (const std::exception& e) {
        spdlog::error("Error querying last event for {}: {}", camera_id, e.what());
        return std::nullopt;
    }
}

json get_cameras_status(
    DbPool& db,
    const std::unordered_map<std::string, CameraConfig>& cameras
) {
    json result = json::array();

    if (!cameras.empty()) {
        // Use cameras from config (standalone mode)
        for (const auto& [id, cam] : cameras) {
            if (!cam.enabled) continue;
            json c;
            c["id"]          = id;
            c["name"]        = cam.name;
            c["connected"]   = true;
            c["buffer_size"] = 0;
            auto last = get_camera_last_event(db, id);
            c["last_event_time"] = last ? json(*last) : json(nullptr);
            result.push_back(std::move(c));
        }
    } else {
        // No cameras in config — discover from database (add-on / container mode)
        try {
            auto conn = db.acquire();
            pqxx::nontransaction txn(*conn);
            auto rows = txn.exec(
                "SELECT DISTINCT camera_id FROM detection_events ORDER BY camera_id"
            );

            for (const auto& row : rows) {
                std::string id = row[0].as<std::string>();
                // Convert snake_case to Title Case for display name
                std::string name = id;
                bool cap = true;
                for (char& ch : name) {
                    if (ch == '_') { ch = ' '; cap = true; }
                    else if (cap) { ch = static_cast<char>(std::toupper(ch)); cap = false; }
                }
                json c;
                c["id"]          = id;
                c["name"]        = name;
                c["connected"]   = true;
                c["buffer_size"] = 0;
                auto last = get_camera_last_event(db, id);
                c["last_event_time"] = last ? json(*last) : json(nullptr);
                result.push_back(std::move(c));
            }
        } catch (const std::exception& e) {
            spdlog::error("Error discovering cameras from DB: {}", e.what());
        }
    }

    return result;
}

// --- Search functions ---

json search_events_fts(DbPool& db, const SearchParams& params) {
    try {
        auto conn = db.acquire();
        pqxx::nontransaction txn(*conn);

        // Build tsquery from user input
        std::string ts_query_str = params.query;
        // Replace spaces with & for AND matching
        for (auto& ch : ts_query_str) {
            if (ch == ' ') ch = '&';
        }

        std::ostringstream sql;
        // Search motion events (ai_vision_context)
        sql << R"(
            SELECT * FROM (
                SELECT
                    'event' as type,
                    de.event_id as id,
                    de.camera_id,
                    de.camera_name,
                    de.started_at as timestamp,
                    de.recording_url,
                    de.snapshot_url,
                    de.total_detections,
                    de.duration_seconds,
                    STRING_AGG(DISTINCT d.class_name, ', ' ORDER BY d.class_name) as detected_classes,
                    avc.context_text as ai_context,
                    ts_rank(avc.context_tsv, plainto_tsquery('english', )" << txn.quote(params.query) << R"()) as rank
                FROM ai_vision_context avc
                JOIN detection_events de ON avc.event_id = de.event_id
                LEFT JOIN detections d ON de.event_id = d.event_id
                WHERE avc.context_tsv @@ plainto_tsquery('english', )" << txn.quote(params.query) << R"()
        )";

        if (params.camera_id) sql << " AND de.camera_id = " << txn.quote(*params.camera_id);
        if (params.start_date) sql << " AND de.started_at >= " << txn.quote(*params.start_date);
        if (params.end_date) sql << " AND de.started_at < " << txn.quote(*params.end_date);

        if (!params.class_filter.empty()) {
            sql << " AND EXISTS (SELECT 1 FROM detections d2 WHERE d2.event_id = de.event_id AND d2.class_name IN (";
            for (size_t i = 0; i < params.class_filter.size(); ++i) {
                if (i > 0) sql << ",";
                sql << txn.quote(params.class_filter[i]);
            }
            sql << "))";
        }

        sql << R"(
                GROUP BY de.event_id, de.camera_id, de.camera_name, de.started_at,
                         de.recording_url, de.snapshot_url, de.total_detections,
                         de.duration_seconds, avc.context_text, avc.context_tsv

                UNION ALL

                SELECT
                    'snapshot' as type,
                    ps.id::text as id,
                    ps.camera_id,
                    ps.camera_id as camera_name,
                    ps.captured_at as timestamp,
                    NULL as recording_url,
                    ps.snapshot_filename as snapshot_url,
                    0 as total_detections,
                    NULL::float as duration_seconds,
                    NULL as detected_classes,
                    ps.context_text as ai_context,
                    ts_rank(ps.context_tsv, plainto_tsquery('english', )" << txn.quote(params.query) << R"()) as rank
                FROM periodic_snapshots ps
                WHERE ps.context_tsv @@ plainto_tsquery('english', )" << txn.quote(params.query) << R"()
                  AND ps.is_valid = true
        )";

        if (params.camera_id) sql << " AND ps.camera_id = " << txn.quote(*params.camera_id);
        if (params.start_date) sql << " AND ps.captured_at >= " << txn.quote(*params.start_date);
        if (params.end_date) sql << " AND ps.captured_at < " << txn.quote(*params.end_date);

        sql << ") combined ORDER BY rank DESC LIMIT " << params.limit;

        auto result = txn.exec(sql.str());

        json events = json::array();
        for (const auto& row : result) {
            json item;
            item["type"]             = field_or_null(row, 0);
            item["id"]               = field_or_null(row, 1);
            item["camera_id"]        = field_or_null(row, 2);
            item["camera_name"]      = field_or_null(row, 3);
            item["timestamp"]        = timestamp_or_null(row, 4);
            item["recording_url"]    = field_or_null(row, 5);
            item["snapshot_url"]     = field_or_null(row, 6);
            item["total_detections"] = field_as_int_or_null(row, 7);
            item["duration_seconds"] = field_as_double_or_null(row, 8);
            item["detected_classes"] = field_or_null(row, 9);
            item["ai_context"]       = field_or_null(row, 10);
            item["rank"]             = field_as_double_or_null(row, 11);
            events.push_back(std::move(item));
        }

        return json{
            {"events", events},
            {"count", static_cast<int>(events.size())},
            {"search_mode", "fts"},
            {"query", params.query}
        };

    } catch (const std::exception& e) {
        spdlog::error("Error in FTS search: {}", e.what());
        return json{{"events", json::array()}, {"count", 0}, {"search_mode", "fts"}, {"query", params.query}};
    }
}

json search_events_semantic(DbPool& db, const SearchParams& params,
                            const std::vector<float>& query_embedding) {
    try {
        auto conn = db.acquire();
        pqxx::nontransaction txn(*conn);

        // Format embedding as pgvector literal: [0.1,0.2,...]
        std::ostringstream vec_str;
        vec_str << "[";
        for (size_t i = 0; i < query_embedding.size(); ++i) {
            if (i > 0) vec_str << ",";
            vec_str << query_embedding[i];
        }
        vec_str << "]";
        std::string vec_literal = vec_str.str();

        std::ostringstream sql;
        sql << R"(
            SELECT * FROM (
                SELECT
                    'event' as type,
                    de.event_id as id,
                    de.camera_id,
                    de.camera_name,
                    de.started_at as timestamp,
                    de.recording_url,
                    de.snapshot_url,
                    de.total_detections,
                    de.duration_seconds,
                    STRING_AGG(DISTINCT d.class_name, ', ' ORDER BY d.class_name) as detected_classes,
                    avc.context_text as ai_context,
                    1 - (avc.context_embedding <=> )" << txn.quote(vec_literal) << R"(::vector) as similarity
                FROM ai_vision_context avc
                JOIN detection_events de ON avc.event_id = de.event_id
                LEFT JOIN detections d ON de.event_id = d.event_id
                WHERE avc.context_embedding IS NOT NULL
        )";

        if (params.camera_id) sql << " AND de.camera_id = " << txn.quote(*params.camera_id);
        if (params.start_date) sql << " AND de.started_at >= " << txn.quote(*params.start_date);
        if (params.end_date) sql << " AND de.started_at < " << txn.quote(*params.end_date);

        sql << R"(
                GROUP BY de.event_id, de.camera_id, de.camera_name, de.started_at,
                         de.recording_url, de.snapshot_url, de.total_detections,
                         de.duration_seconds, avc.context_text, avc.context_embedding

                UNION ALL

                SELECT
                    'snapshot' as type,
                    ps.id::text as id,
                    ps.camera_id,
                    ps.camera_id as camera_name,
                    ps.captured_at as timestamp,
                    NULL as recording_url,
                    ps.snapshot_filename as snapshot_url,
                    0 as total_detections,
                    NULL::float as duration_seconds,
                    NULL as detected_classes,
                    ps.context_text as ai_context,
                    1 - (ps.context_embedding <=> )" << txn.quote(vec_literal) << R"(::vector) as similarity
                FROM periodic_snapshots ps
                WHERE ps.context_embedding IS NOT NULL
                  AND ps.is_valid = true
        )";

        if (params.camera_id) sql << " AND ps.camera_id = " << txn.quote(*params.camera_id);
        if (params.start_date) sql << " AND ps.captured_at >= " << txn.quote(*params.start_date);
        if (params.end_date) sql << " AND ps.captured_at < " << txn.quote(*params.end_date);

        sql << ") combined ORDER BY similarity DESC LIMIT " << params.limit;

        auto result = txn.exec(sql.str());

        json events = json::array();
        for (const auto& row : result) {
            json item;
            item["type"]             = field_or_null(row, 0);
            item["id"]               = field_or_null(row, 1);
            item["camera_id"]        = field_or_null(row, 2);
            item["camera_name"]      = field_or_null(row, 3);
            item["timestamp"]        = timestamp_or_null(row, 4);
            item["recording_url"]    = field_or_null(row, 5);
            item["snapshot_url"]     = field_or_null(row, 6);
            item["total_detections"] = field_as_int_or_null(row, 7);
            item["duration_seconds"] = field_as_double_or_null(row, 8);
            item["detected_classes"] = field_or_null(row, 9);
            item["ai_context"]       = field_or_null(row, 10);
            item["similarity"]       = field_as_double_or_null(row, 11);
            events.push_back(std::move(item));
        }

        return json{
            {"events", events},
            {"count", static_cast<int>(events.size())},
            {"search_mode", "semantic"},
            {"query", params.query}
        };

    } catch (const std::exception& e) {
        spdlog::error("Error in semantic search: {}", e.what());
        return json{{"events", json::array()}, {"count", 0}, {"search_mode", "semantic"}, {"query", params.query}};
    }
}

json get_periodic_snapshots(DbPool& db, const std::string& camera_id,
                            const std::string& date) {
    try {
        auto date_tp = time_utils::from_date_string(date);
        if (!date_tp) return json::array();

        auto start_str = time_utils::to_iso8601(time_utils::start_of_day(*date_tp));
        auto end_str   = time_utils::to_iso8601(time_utils::end_of_day(*date_tp));

        auto conn = db.acquire();
        pqxx::nontransaction txn(*conn);

        auto result = txn.exec(R"(
            SELECT id, camera_id, captured_at, snapshot_filename,
                   thumbnail_filename, context_text, is_valid
            FROM periodic_snapshots
            WHERE camera_id = $1 AND captured_at >= $2 AND captured_at < $3
              AND is_valid = true
            ORDER BY captured_at DESC
        )", pqxx::params{camera_id, start_str, end_str});

        json snapshots = json::array();
        for (const auto& row : result) {
            snapshots.push_back({
                {"type", "snapshot"},
                {"snapshot_id", row[0].as<int>()},
                {"camera_id", row[1].c_str()},
                {"captured_at", time_utils::pg_timestamp_to_iso8601(row[2].c_str())},
                {"snapshot_url", row[3].c_str()},
                {"thumbnail_url", row[4].is_null() ? json(nullptr) : json(row[4].c_str())},
                {"ai_context", row[5].is_null() ? json(nullptr) : json(row[5].c_str())},
                {"is_valid", row[6].as<bool>()}
            });
        }

        return snapshots;

    } catch (const std::exception& e) {
        spdlog::error("Error querying periodic snapshots for {}: {}", camera_id, e.what());
        return json::array();
    }
}

void insert_periodic_snapshot(DbPool& db,
                              const std::string& camera_id,
                              const std::string& snapshot_filename,
                              const std::string& thumbnail_filename,
                              const std::string& context_text,
                              const std::vector<float>& embedding,
                              const std::string& source_model,
                              bool is_valid) {
    try {
        auto conn = db.acquire();
        pqxx::work txn(*conn);

        // Format embedding as pgvector literal
        std::string vec_literal;
        if (!embedding.empty()) {
            std::ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < embedding.size(); ++i) {
                if (i > 0) oss << ",";
                oss << embedding[i];
            }
            oss << "]";
            vec_literal = oss.str();
        }

        if (!vec_literal.empty()) {
            txn.exec(R"(
                INSERT INTO periodic_snapshots
                    (camera_id, captured_at, snapshot_filename, thumbnail_filename,
                     context_text, context_embedding, source_model, is_valid)
                VALUES ($1, NOW(), $2, $3, $4, $5::vector, $6, $7)
            )", pqxx::params{
                camera_id, snapshot_filename, thumbnail_filename,
                context_text, vec_literal, source_model, is_valid
            });
        } else {
            txn.exec(R"(
                INSERT INTO periodic_snapshots
                    (camera_id, captured_at, snapshot_filename, thumbnail_filename,
                     context_text, source_model, is_valid)
                VALUES ($1, NOW(), $2, $3, $4, $5, $6)
            )", pqxx::params{
                camera_id, snapshot_filename, thumbnail_filename,
                context_text, source_model, is_valid
            });
        }

        txn.commit();
        spdlog::debug("Inserted periodic snapshot {} for {}", snapshot_filename, camera_id);

    } catch (const std::exception& e) {
        spdlog::error("Error inserting periodic snapshot for {}: {}", camera_id, e.what());
    }
}

} // namespace hms::api_queries
