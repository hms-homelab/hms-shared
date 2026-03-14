#pragma once

#include <atomic>
#include <functional>
#include <optional>
#include <string>
#include <vector>
#include <curl/curl.h>

namespace hms {

/**
 * Supported LLM providers
 */
enum class LLMProvider {
    OLLAMA,     // Local Ollama: /api/generate (no auth)
    OPENAI,     // OpenAI/ChatGPT: /v1/chat/completions (Bearer token)
    GEMINI,     // Google Gemini: /v1beta/models/{model}:generateContent (API key)
    ANTHROPIC   // Claude: /v1/messages (x-api-key header)
};

/**
 * LLM configuration
 */
struct LLMConfig {
    bool enabled = false;
    LLMProvider provider = LLMProvider::OLLAMA;
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

    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp);
};

} // namespace hms
