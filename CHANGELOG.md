# Changelog

## v1.5.1 (2026-03-14)

### Fixed
- **OpenAI GPT-5+ compatibility**: Use `max_completion_tokens` instead of `max_tokens` in OpenAI
  chat completions request (required by GPT-5.2 and newer models)

## v1.5.0 (2026-03-14)

### Changed
- **Namespace migration**: `yolo::` -> `hms::` across all modules (config, common, db, mqtt)
  - `hms::ConfigManager`, `hms::MqttClient`, `hms::DbPool`, `hms::EventLogger`
  - `hms::time_utils`, `hms::api_queries`
  - Config struct defaults (model paths, topic prefixes, log paths) unchanged
- All tests updated to use `hms::` namespace

## v1.4.0 (2026-03-14)

### Added
- **LLM module** (`llm/`): Multi-provider LLM client (`hms::LLMClient`) in `hms` namespace
  - Ollama (`/api/generate`), OpenAI (`/v1/chat/completions`), Google Gemini (`/v1beta/models/:generateContent`), Anthropic Claude (`/v1/messages`)
  - Configurable temperature, max_tokens, timeout
  - Ollama model eviction via `keep_alive_seconds` (default 0 = unload from VRAM after call)
  - Prompt template file loading and `{placeholder}` substitution
- **`hms_llm` CMake target**: Standalone static library (nlohmann_json + curl only), no DB/MQTT/config deps. Consumers can link just `hms_llm` without pulling the full `hms_shared` target.
- CMakeLists.txt bumped to v1.4.0, added `find_package(CURL)` dependency

## v1.3.0 (2026-03-11)

### Added
- **periodic_vision config**: Separate `LlavaConfig` for periodic snapshot vision model (e.g. moondream), parsed from `periodic_vision` YAML section

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
