# Changelog

## v1.2.0 (2026-03-04)

### Added
- **Search APIs**: full-text search (FTS) and semantic vector search over ai_vision_context + periodic_snapshots
- **Periodic snapshot queries**: insert and query periodic snapshots for timeline display
- **Config**: `ollama_url` field in config manager
- 280 lines of Catch2 tests for search and snapshot queries

## v1.1.0 (2026-03-02)

### Added
- **GPU config**: `gpu_enabled` field in CameraConfig and detection config
- Config manager tests for GPU fields

## v1.0.0 (2026-02-27)

### Initial release
- **common**: time_utils (ISO 8601 formatting)
- **config**: config_manager (YAML parsing for cameras, MQTT, DB, detection, timeline, logging)
- **db**: db_pool (PostgreSQL connection pool with pqxx), event_logger, api_queries
- **mqtt**: mqtt_client (Paho MQTT C++ wrapper)
- 24 Catch2 unit tests
