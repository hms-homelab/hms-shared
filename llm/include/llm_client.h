#pragma once

#include <atomic>
#include <functional>
#include <optional>
#include <string>
#include <vector>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

namespace hms {

/**
 * Supported LLM providers.
 *
 * `endpoint` is always a BASE URL and the provider decides what path to append
 * to it. That is what makes OPENAI mean "any server speaking the OpenAI chat
 * API" rather than api.openai.com specifically: OpenAI itself, OpenRouter,
 * LM Studio, vLLM, llama.cpp's server, Groq, Together, or Ollama's own /v1
 * surface all work by pointing `endpoint` at them.
 *
 * OLLAMA is kept separate because it speaks its own /api/generate protocol,
 * not because Ollama cannot be reached through OPENAI. It can.
 */
enum class LLMProvider {
    OLLAMA,     // Ollama native: <endpoint>/api/generate (no auth)
    OPENAI,     // ANY OpenAI-compatible server: <endpoint>/v1/chat/completions (Bearer)
    GEMINI,     // Google Gemini: <endpoint>/v1beta/models/{model}:generateContent (API key)
    ANTHROPIC   // Claude: <endpoint>/v1/messages (x-api-key header)
};

/**
 * LLM configuration
 */
struct LLMConfig {
    bool enabled = false;
    LLMProvider provider = LLMProvider::OLLAMA;
    /// BASE URL only, no path. The provider appends its own (see LLMProvider).
    std::string endpoint = "http://localhost:11434";
    std::string model = "llama3.1:8b-instruct-q4_K_M";
    std::string api_key;
    double temperature = 0.3;
    int max_tokens = 1024;
    long timeout_seconds = 120;
    long connect_timeout_seconds = 10;
    int keep_alive_seconds = 0;   // Ollama only: 0 = unload model after call (evict from VRAM)
};

/**
 * Image data for vision requests
 */
struct LLMImage {
    std::string base64_data;
    std::string mime_type = "image/jpeg";
};

/**
 * Response from a vision/generate call with metadata
 */
struct LLMResponse {
    std::optional<std::string> text;
    bool was_aborted = false;
    double elapsed_seconds = 0;
};

/**
 * Tool definition for function calling
 */
struct ToolDefinition {
    std::string name;
    std::string description;
    nlohmann::json parameters;  // JSON Schema
};

/**
 * A tool call returned by the model
 */
struct ToolCall {
    std::string id;               // provider-assigned ID (for matching results)
    std::string name;
    nlohmann::json arguments;
};

/**
 * Multi-turn chat message (supports tool-use conversations)
 */
struct ChatMessage {
    std::string role;             // "system", "user", "assistant", "tool"
    std::string content;
    std::vector<ToolCall> tool_calls;   // when role=="assistant"
    std::string tool_call_id;           // when role=="tool"
};

/**
 * Response from generateWithTools()
 */
struct LLMToolResponse {
    std::optional<std::string> text;
    std::vector<ToolCall> tool_calls;
    bool was_aborted = false;
    double elapsed_seconds = 0;
    std::string stop_reason;      // "end_turn", "tool_use", "stop", etc.
};

/**
 * Receives each text delta as it arrives from a streaming call.
 *
 * Return false to stop the transfer. That is the caller's own stop button, and
 * it is distinct from `abort_flag`: the flag is for another thread cancelling
 * the request, this is for the consumer deciding it has heard enough.
 */
using StreamCallback = std::function<bool(const std::string& delta)>;

/**
 * LLMClient - Multi-provider LLM client for HMS services
 *
 * Supports Ollama, OpenAI/ChatGPT, Google Gemini, and Anthropic Claude.
 * All calls are blocking. Thread-safe (each call uses its own curl handle).
 *
 * Usage:
 *   hms::LLMConfig config;
 *   config.provider = hms::LLMProvider::OLLAMA;
 *   config.endpoint = "http://192.168.2.5:11434";
 *   config.model = "llama3.1:8b-instruct-q4_K_M";
 *
 *   hms::LLMClient client(config);
 *   auto result = client.generate("Summarize this data: ...");
 */
class LLMClient {
public:
    explicit LLMClient(const LLMConfig& config);
    ~LLMClient() = default;

    LLMClient(const LLMClient&) = delete;
    LLMClient& operator=(const LLMClient&) = delete;

    /**
     * Generate text from a prompt
     *
     * @param prompt Complete prompt text
     * @return Generated text, or nullopt on failure
     */
    std::optional<std::string> generate(const std::string& prompt);

    /**
     * Generate text from a prompt with images (vision)
     *
     * @param prompt Complete prompt text
     * @param images Vector of base64-encoded images
     * @param abort_flag Optional atomic flag to abort the request mid-flight
     * @return LLMResponse with text, abort status, and elapsed time
     */
    LLMResponse generateVision(const std::string& prompt,
                                const std::vector<LLMImage>& images,
                                const std::atomic<bool>* abort_flag = nullptr);

    /**
     * Tool-use: multi-turn conversation with function calling
     *
     * Supports all 4 providers. Returns tool_calls when the model wants to
     * invoke functions, or text when the model produces a final answer.
     *
     * `force_tool` names a tool the model MUST call on this request. Empty (the
     * default) leaves the choice to the model, which is what every existing
     * caller gets.
     *
     * *** OLLAMA CANNOT FORCE. *** Its /api/chat has no tool_choice equivalent,
     * so `force_tool` is IGNORED there and the model may answer in prose
     * without calling anything. That is a silent difference, and a caller whose
     * correctness depends on the call actually happening -- one that treats the
     * tool result as the only grounded source, say -- must check the provider
     * itself rather than trusting this parameter. It is not faked here, because
     * a fake would move the failure from "unsupported" to "looked supported and
     * quietly did not happen".
     *
     * Forcing matters more than it sounds: offered a tool it thinks it does not
     * need, a model will often answer from its own training instead, and the
     * reply comes back fluent, plausible and ungrounded.
     */
    LLMToolResponse generateWithTools(
        const std::vector<ChatMessage>& messages,
        const std::vector<ToolDefinition>& tools,
        const std::atomic<bool>* abort_flag = nullptr,
        const std::string& force_tool = "");

    /// True when this client's provider can honour `force_tool`. False for
    /// Ollama. Exposed so a caller can refuse to start rather than discover it
    /// one ungrounded answer at a time.
    bool supportsForcedTool() const;

    /**
     * Streaming chat completion, text only.
     *
     * Supports all 4 providers. `on_delta` is called on the calling thread for
     * each text fragment as it arrives, and returning false from it stops the
     * transfer. The complete text is ALSO returned in the result, so a caller
     * that only wants the whole answer can pass a callback that always returns
     * true and ignore the deltas.
     *
     * Deliberately has no `tools` parameter. Tool rounds have nothing to show a
     * user, so they stay on generateWithTools; this is for the final answer
     * turn, where the waiting happens.
     *
     * @param messages   Conversation so far, same shape as generateWithTools
     * @param on_delta   Called per fragment; return false to stop
     * @param abort_flag Optional atomic flag to abort the request mid-flight
     * @return LLMResponse with the complete text, abort status, elapsed time
     */
    LLMResponse generateStream(const std::vector<ChatMessage>& messages,
                                const StreamCallback& on_delta,
                                const std::atomic<bool>* abort_flag = nullptr);

    /**
     * Generate text embeddings (Ollama, OpenAI only)
     *
     * @param text Text to embed
     * @return Vector of floats (dimension depends on model)
     * @throws std::runtime_error on failure or unsupported provider
     */
    std::vector<float> embed(const std::string& text);
    std::vector<float> embed(const std::string& text, const std::string& model);

    /**
     * Format a float vector as a pgvector literal: [0.1,0.2,0.3]
     */
    static std::string toVectorLiteral(const std::vector<float>& vec);

    /**
     * Base64-encode binary data (e.g. JPEG image bytes)
     */
    static std::string base64Encode(const std::vector<unsigned char>& data);

    /**
     * Force Ollama to unload a model from VRAM (keep_alive=0)
     */
    static void forceUnloadModel(const std::string& ollama_endpoint,
                                  const std::string& model_name);

    /**
     * Check if client is configured and ready
     */
    bool isEnabled() const { return config_.enabled; }

    const LLMConfig& config() const { return config_; }

    /**
     * Parse provider string to enum
     * Accepts: "ollama", "openai", "chatgpt", "gemini", "google", "anthropic", "claude"
     */
    static LLMProvider parseProvider(const std::string& provider_str);

    /**
     * Get provider display name
     */
    static std::string providerName(LLMProvider provider);

    /**
     * Load prompt template from file, replacing {placeholder} tokens.
     *
     * @param filepath Path to prompt template file
     * @return File contents, or empty string on failure
     */
    static std::string loadPromptFile(const std::string& filepath);

    /**
     * Substitute {key} placeholders in a template with values.
     *
     * @param tmpl Template string with {key} placeholders
     * @param values Map of key -> value replacements
     * @return String with all placeholders replaced
     */
    static std::string substituteTemplate(
        const std::string& tmpl,
        const std::vector<std::pair<std::string, std::string>>& values);

private:
    LLMConfig config_;

    std::optional<std::string> generateOllama(const std::string& prompt);
    std::optional<std::string> generateOpenAI(const std::string& prompt);
    std::optional<std::string> generateGemini(const std::string& prompt);
    std::optional<std::string> generateAnthropic(const std::string& prompt);

    std::optional<std::string> generateOllamaVision(const std::string& prompt, const std::vector<LLMImage>& images, const std::atomic<bool>* abort_flag = nullptr);
    std::optional<std::string> generateOpenAIVision(const std::string& prompt, const std::vector<LLMImage>& images, const std::atomic<bool>* abort_flag = nullptr);
    std::optional<std::string> generateGeminiVision(const std::string& prompt, const std::vector<LLMImage>& images, const std::atomic<bool>* abort_flag = nullptr);
    std::optional<std::string> generateAnthropicVision(const std::string& prompt, const std::vector<LLMImage>& images, const std::atomic<bool>* abort_flag = nullptr);

    std::optional<std::string> httpPost(const std::string& url,
                                         const std::string& body,
                                         struct curl_slist* headers,
                                         const std::atomic<bool>* abort_flag = nullptr,
                                         bool* was_aborted = nullptr);

    /**
     * POST that hands the response back one line at a time as it arrives.
     *
     * Both NDJSON (Ollama) and SSE (everyone else) are line-oriented, so line
     * assembly lives here once and each provider only has to interpret a line.
     * `on_line` returning false stops the transfer.
     */
    bool httpPostStream(const std::string& url,
                         const std::string& body,
                         struct curl_slist* headers,
                         const std::function<bool(const std::string& line)>& on_line,
                         const std::atomic<bool>* abort_flag = nullptr,
                         bool* was_aborted = nullptr);

    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp);
};

} // namespace hms
