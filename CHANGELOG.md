# Changelog

## v1.6.13 (2026-09-01)

### Added
- **`generateWithTools` can force a specific tool.** New optional `force_tool`
  argument, after `abort_flag` so every existing call site is unchanged. Empty
  (the default) leaves the choice to the model, exactly as before.

  Offered a tool it does not think it needs, a model will often answer from its
  own training instead, and the reply comes back fluent, plausible and
  ungrounded. `cpapdash-ai`'s `WebSearch` already worked around this by talking
  to the OpenAI Responses API directly with `"tool_choice":"required"` and
  measured it: unforced returned 0 citations, forced returned 3. This brings the
  same control to the normal conversational path.

  The per-provider shape lives in `tool_format::applyToolChoice`, alongside the
  other provider differences and unit-tested there, rather than inline in
  `generateWithTools` where it cannot be tested without a live HTTP call.

  | Provider | Field |
  |---|---|
  | OpenAI | `tool_choice: {type: function, function: {name}}` |
  | Anthropic | `tool_choice: {type: tool, name}` |
  | Gemini | `toolConfig.functionCallingConfig: {mode: ANY, allowedFunctionNames: [name]}` |
  | Ollama | **not supported** |

  Gemini needs both halves: `mode: ANY` alone means "call something", which
  would let it pick any tool on the list rather than the named one.

- **`supportsForcedTool()`**, because **Ollama cannot force.** Its `/api/chat`
  has no `tool_choice` equivalent, so `force_tool` is ignored there and the
  model may still answer in prose without calling anything. That difference is
  silent in the request body, so it is not faked — a fake would move the failure
  from "unsupported" to "looked supported and quietly did not happen". A caller
  whose correctness depends on the call actually happening must ask this and
  refuse to start, rather than discover it one ungrounded answer at a time.

## v1.6.12 (2026-08-15)

### Fixed
- **`generateWithTools` was broken for Ollama on every round after the first.**
  `buildOllamaMessages` serialised assistant `tool_calls` through the OpenAI
  serializer, which renders `arguments` as a JSON **string**. Ollama requires a
  JSON **object** and rejects the entire request with

      {"error":"Value looks like object, but can't find closing '}' symbol"}

  which reads like a malformed body rather than a type mismatch in one field.

  The first round never carries `tool_calls`, so it always worked; the break
  only appeared on the second round, when the assistant's own tool call is
  echoed back alongside the tool result. Any Ollama tool loop therefore died
  the moment it actually used a tool. Found by running a real agent loop
  against the relay, not by a unit test.

  OpenAI still gets the string form, so the two builders must not be collapsed
  back into one. Both directions now have a test.

## v1.6.11 (2026-08-14)

### Added
- **`LLMClient::generateStream()`** — streaming chat completion, text only, on
  all four providers. `on_delta` is called per text fragment as it arrives and
  returning `false` from it stops the transfer, which is a consumer's stop
  button and is distinct from `abort_flag` (another thread cancelling). The
  complete text is returned in the result as well, so a caller that only wants
  the whole answer can ignore the deltas.

  Deliberately takes no `tools` parameter. Tool rounds have nothing to show a
  user, so they stay on `generateWithTools`; this is for the final answer turn,
  which is where the waiting happens.

- **`tool_format::parseStreamLine(provider, line)`** — extracts the text delta
  from one raw stream line. Every provider here is line-oriented (Ollama sends
  NDJSON, the other three send SSE), so line assembly happens once in
  `httpPostStream` and each provider only has to interpret a line. Exposed
  through `llm_tool_format.h` so streaming is testable against captured frames
  with no HTTP involved, matching how the tool parsers are already tested.

### Notes for consumers
- **Reasoning is not answer text, and both ride the same stream.** Anthropic
  sends `thinking_delta` and `input_json_delta` next to `text_delta`, and
  reasoning models on Ollama (measured on `gpt-oss:120b`) emit a `thinking`
  field alongside an *empty* `content` for the whole reasoning phase. Only
  `content` and `text_delta` are treated as answer text, so a model's private
  planning never reaches a caller. Both cases are covered by tests.
- **Time to first token is not time to first word on a reasoning model.**
  Measured against the relay: 119 thinking frames arrive between 0.28s and
  0.94s, and the first actual content frame lands at 0.94s with the answer
  complete at 1.36s. A UI that shows nothing until the first content delta will
  look frozen for about a second.
- Gemini uses `:streamGenerateContent?alt=sse`. Without `alt=sse` that endpoint
  streams a JSON array rather than server-sent events, which no line parser
  would handle.
- A malformed frame costs one fragment, not the answer: it is skipped and the
  transfer continues.
- Verified live against OpenAI (`gpt-4.1-nano`, 38 deltas, first at 0.81s) and
  the Ollama relay (`gpt-oss:120b-cloud`, 70 deltas), including the stop button,
  `abort_flag`, and a 401 producing zero deltas rather than emitting the error
  body as text.

## v1.6.10 (2026-08-14)

### Documentation
- **`LLMProvider::OPENAI` means any OpenAI-compatible server**, not
  api.openai.com specifically. `endpoint` has always been a base URL that the
  provider appends its own path to, so OpenAI, OpenRouter, LM Studio, vLLM,
  llama.cpp and Ollama's own `/v1` surface all work by pointing `endpoint` at
  them. Nothing about that behaviour changed; it was simply never written down,
  and a consumer reading "OpenAI/ChatGPT" reasonably concluded a local server
  was out of scope.
- `OLLAMA` is documented as being separate because it speaks its own
  `/api/generate` protocol, not because Ollama is unreachable through `OPENAI`.


## v1.6.9 (2026-08-08)

### Fixed
- **DbPool connection-slot leak**: `DbPool::acquire()` permanently leaked a pool
  slot when a connection failed its `SELECT 1` liveness check *and* the
  subsequent `create_connection()` reconnect also threw (e.g. Postgres briefly
  unreachable). The popped connection was not yet wrapped in a `ConnectionGuard`,
  so the `throw;` destroyed it without returning it to the pool. With a small
  `pool_size`, a single DB outage could leak every slot, after which all
  `acquire()` calls failed with "DB pool exhausted — no connection available
  after 10s" until the process was restarted. The slot is now returned to the
  pool before rethrowing, making the pool self-healing across transient outages.

## v1.6.5 (2026-03-18)

### Added
- `generateWithTools()` — LLM function calling for all 4 providers (Ollama, OpenAI, Gemini, Anthropic)
- `embed()` — text embedding generation (Ollama, OpenAI)
- `toVectorLiteral()` — pgvector literal formatter
- New structs: `ToolDefinition`, `ToolCall`, `ChatMessage`, `LLMToolResponse`
- Internal `llm_tool_format.h` for testable serialization/parsing helpers
- `llm_tests` Catch2 target (30 tests for tool serialization, response parsing, embeddings)

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
