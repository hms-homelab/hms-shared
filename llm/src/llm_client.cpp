#include "llm_client.h"
#include "llm_tool_format.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

#include <nlohmann/json.hpp>

namespace hms {

// ─── Abort progress callback ────────────────────────────────────────────────

static int abortProgressCallback(void* clientp, curl_off_t, curl_off_t, curl_off_t, curl_off_t) {
    auto* flag = static_cast<const std::atomic<bool>*>(clientp);
    return (flag && flag->load(std::memory_order_acquire)) ? 1 : 0;
}

// ─── Construction ────────────────────────────────────────────────────────────

LLMClient::LLMClient(const LLMConfig& config) : config_(config) {}

// ─── Static helpers ──────────────────────────────────────────────────────────

size_t LLMClient::WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    auto* str = static_cast<std::string*>(userp);
    str->append(static_cast<char*>(contents), size * nmemb);
    return size * nmemb;
}

LLMProvider LLMClient::parseProvider(const std::string& provider_str) {
    std::string lower = provider_str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower == "openai" || lower == "chatgpt") return LLMProvider::OPENAI;
    if (lower == "gemini" || lower == "google")   return LLMProvider::GEMINI;
    if (lower == "anthropic" || lower == "claude") return LLMProvider::ANTHROPIC;
    return LLMProvider::OLLAMA;
}

std::string LLMClient::providerName(LLMProvider provider) {
    switch (provider) {
        case LLMProvider::OLLAMA:    return "ollama";
        case LLMProvider::OPENAI:    return "openai";
        case LLMProvider::GEMINI:    return "gemini";
        case LLMProvider::ANTHROPIC: return "anthropic";
    }
    return "unknown";
}

std::string LLMClient::loadPromptFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "LLM: Cannot open prompt file: " << filepath << std::endl;
        return {};
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

std::string LLMClient::substituteTemplate(
    const std::string& tmpl,
    const std::vector<std::pair<std::string, std::string>>& values) {

    std::string result = tmpl;
    for (const auto& [key, value] : values) {
        std::string placeholder = "{" + key + "}";
        size_t pos = 0;
        while ((pos = result.find(placeholder, pos)) != std::string::npos) {
            result.replace(pos, placeholder.size(), value);
            pos += value.size();
        }
    }
    return result;
}

// ─── Generate (dispatch) ────────────────────────────────────────────────────

std::optional<std::string> LLMClient::generate(const std::string& prompt) {
    switch (config_.provider) {
        case LLMProvider::OLLAMA:    return generateOllama(prompt);
        case LLMProvider::OPENAI:    return generateOpenAI(prompt);
        case LLMProvider::GEMINI:    return generateGemini(prompt);
        case LLMProvider::ANTHROPIC: return generateAnthropic(prompt);
    }
    return std::nullopt;
}

// ─── Generate Vision (dispatch with timing and abort) ───────────────────────

LLMResponse LLMClient::generateVision(const std::string& prompt,
                                        const std::vector<LLMImage>& images,
                                        const std::atomic<bool>* abort_flag) {
    LLMResponse result;
    auto start = std::chrono::steady_clock::now();

    // Check abort before starting
    if (abort_flag && abort_flag->load(std::memory_order_acquire)) {
        result.was_aborted = true;
        return result;
    }

    bool was_aborted = false;
    std::optional<std::string> text;

    switch (config_.provider) {
        case LLMProvider::OLLAMA:    text = generateOllamaVision(prompt, images, abort_flag); break;
        case LLMProvider::OPENAI:    text = generateOpenAIVision(prompt, images, abort_flag); break;
        case LLMProvider::GEMINI:    text = generateGeminiVision(prompt, images, abort_flag); break;
        case LLMProvider::ANTHROPIC: text = generateAnthropicVision(prompt, images, abort_flag); break;
    }

    auto end = std::chrono::steady_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(end - start).count();
    result.text = text;

    // Check if we were aborted
    if (abort_flag && abort_flag->load(std::memory_order_acquire)) {
        result.was_aborted = true;
    }

    return result;
}

// ─── HTTP POST ──────────────────────────────────────────────────────────────

std::optional<std::string> LLMClient::httpPost(const std::string& url,
                                                const std::string& body,
                                                struct curl_slist* headers,
                                                const std::atomic<bool>* abort_flag,
                                                bool* was_aborted) {
    if (was_aborted) *was_aborted = false;

    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "LLM: curl_easy_init() failed" << std::endl;
        return std::nullopt;
    }

    std::string response;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(body.size()));
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, config_.timeout_seconds);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, config_.connect_timeout_seconds);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);

    if (abort_flag) {
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, abortProgressCallback);
        curl_easy_setopt(curl, CURLOPT_XFERINFODATA, abort_flag);
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    }

    CURLcode res = curl_easy_perform(curl);

    if (res == CURLE_ABORTED_BY_CALLBACK) {
        std::cerr << "LLM: request aborted by callback" << std::endl;
        if (was_aborted) *was_aborted = true;
        curl_easy_cleanup(curl);
        return std::nullopt;
    }

    if (res != CURLE_OK) {
        std::cerr << "LLM: HTTP request failed: " << curl_easy_strerror(res) << std::endl;
        curl_easy_cleanup(curl);
        return std::nullopt;
    }

    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_easy_cleanup(curl);

    if (http_code != 200) {
        std::cerr << "LLM: HTTP " << http_code << " from " << providerName(config_.provider) << std::endl;
        if (!response.empty()) {
            std::cerr << "LLM: " << response.substr(0, 500) << std::endl;
        }
        return std::nullopt;
    }

    return response;
}

// ─── OLLAMA ─────────────────────────────────────────────────────────────────
// POST /api/generate {"model":"…","prompt":"…","stream":false}
// Response: {"response":"…"}

std::optional<std::string> LLMClient::generateOllama(const std::string& prompt) {
    std::string url = config_.endpoint + "/api/generate";

    nlohmann::json req = {
        {"model", config_.model},
        {"prompt", prompt},
        {"stream", false},
        {"keep_alive", config_.keep_alive_seconds},
        {"options", {{"temperature", config_.temperature}}}
    };

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    std::string body = req.dump();
    auto response = httpPost(url, body, headers);
    curl_slist_free_all(headers);

    if (!response) return std::nullopt;

    try {
        auto j = nlohmann::json::parse(response.value());
        if (j.contains("response")) return j["response"].get<std::string>();
    } catch (const std::exception& e) {
        std::cerr << "LLM: Failed to parse Ollama response: " << e.what() << std::endl;
    }
    return std::nullopt;
}

// ─── OPENAI ─────────────────────────────────────────────────────────────────
// POST /v1/chat/completions
// {"model":"…","messages":[{"role":"user","content":"…"}],"temperature":0.3}
// Response: {"choices":[{"message":{"content":"…"}}]}

std::optional<std::string> LLMClient::generateOpenAI(const std::string& prompt) {
    std::string url = config_.endpoint + "/v1/chat/completions";

    nlohmann::json req = {
        {"model", config_.model},
        {"messages", {{{"role", "user"}, {"content", prompt}}}},
        {"temperature", config_.temperature},
        {"max_completion_tokens", config_.max_tokens}
    };

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    std::string auth = "Authorization: Bearer " + config_.api_key;
    headers = curl_slist_append(headers, auth.c_str());

    std::string body = req.dump();
    auto response = httpPost(url, body, headers);
    curl_slist_free_all(headers);

    if (!response) return std::nullopt;

    try {
        auto j = nlohmann::json::parse(response.value());
        if (j.contains("choices") && !j["choices"].empty()) {
            return j["choices"][0]["message"]["content"].get<std::string>();
        }
    } catch (const std::exception& e) {
        std::cerr << "LLM: Failed to parse OpenAI response: " << e.what() << std::endl;
    }
    return std::nullopt;
}

// ─── GEMINI ─────────────────────────────────────────────────────────────────
// POST /v1beta/models/{model}:generateContent?key={api_key}
// {"contents":[{"parts":[{"text":"…"}]}],"generationConfig":{"temperature":…}}
// Response: {"candidates":[{"content":{"parts":[{"text":"…"}]}}]}

std::optional<std::string> LLMClient::generateGemini(const std::string& prompt) {
    std::string url = config_.endpoint + "/v1beta/models/" + config_.model +
                      ":generateContent?key=" + config_.api_key;

    nlohmann::json req = {
        {"contents", {{{"parts", {{{"text", prompt}}}}}}},
        {"generationConfig", {
            {"temperature", config_.temperature},
            {"maxOutputTokens", config_.max_tokens}
        }}
    };

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    std::string body = req.dump();
    auto response = httpPost(url, body, headers);
    curl_slist_free_all(headers);

    if (!response) return std::nullopt;

    try {
        auto j = nlohmann::json::parse(response.value());
        if (j.contains("candidates") && !j["candidates"].empty()) {
            return j["candidates"][0]["content"]["parts"][0]["text"].get<std::string>();
        }
    } catch (const std::exception& e) {
        std::cerr << "LLM: Failed to parse Gemini response: " << e.what() << std::endl;
    }
    return std::nullopt;
}

// ─── ANTHROPIC ──────────────────────────────────────────────────────────────
// POST /v1/messages
// {"model":"…","max_tokens":1024,"messages":[{"role":"user","content":"…"}]}
// Headers: x-api-key, anthropic-version
// Response: {"content":[{"text":"…"}]}

std::optional<std::string> LLMClient::generateAnthropic(const std::string& prompt) {
    std::string url = config_.endpoint + "/v1/messages";

    nlohmann::json req = {
        {"model", config_.model},
        {"max_tokens", config_.max_tokens},
        {"messages", {{{"role", "user"}, {"content", prompt}}}}
    };

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    std::string api_header = "x-api-key: " + config_.api_key;
    headers = curl_slist_append(headers, api_header.c_str());
    headers = curl_slist_append(headers, "anthropic-version: 2023-06-01");

    std::string body = req.dump();
    auto response = httpPost(url, body, headers);
    curl_slist_free_all(headers);

    if (!response) return std::nullopt;

    try {
        auto j = nlohmann::json::parse(response.value());
        if (j.contains("content") && !j["content"].empty()) {
            return j["content"][0]["text"].get<std::string>();
        }
    } catch (const std::exception& e) {
        std::cerr << "LLM: Failed to parse Anthropic response: " << e.what() << std::endl;
    }
    return std::nullopt;
}

// ─── OLLAMA VISION ──────────────────────────────────────────────────────────
// POST /api/generate with "images" array of base64 strings

std::optional<std::string> LLMClient::generateOllamaVision(const std::string& prompt,
                                                             const std::vector<LLMImage>& images,
                                                             const std::atomic<bool>* abort_flag) {
    std::string url = config_.endpoint + "/api/generate";

    nlohmann::json req = {
        {"model", config_.model},
        {"prompt", prompt},
        {"stream", false},
        {"keep_alive", config_.keep_alive_seconds},
        {"options", {{"temperature", config_.temperature}}}
    };

    if (!images.empty()) {
        nlohmann::json img_array = nlohmann::json::array();
        for (const auto& img : images) {
            img_array.push_back(img.base64_data);
        }
        req["images"] = img_array;
    }

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    std::string body = req.dump();
    auto response = httpPost(url, body, headers, abort_flag);
    curl_slist_free_all(headers);

    if (!response) return std::nullopt;

    try {
        auto j = nlohmann::json::parse(response.value());
        if (j.contains("response")) return j["response"].get<std::string>();
    } catch (const std::exception& e) {
        std::cerr << "LLM: Failed to parse Ollama vision response: " << e.what() << std::endl;
    }
    return std::nullopt;
}

// ─── OPENAI VISION ──────────────────────────────────────────────────────────
// POST /v1/chat/completions with image_url content parts

std::optional<std::string> LLMClient::generateOpenAIVision(const std::string& prompt,
                                                             const std::vector<LLMImage>& images,
                                                             const std::atomic<bool>* abort_flag) {
    std::string url = config_.endpoint + "/v1/chat/completions";

    nlohmann::json content = nlohmann::json::array();
    for (const auto& img : images) {
        content.push_back({
            {"type", "image_url"},
            {"image_url", {{"url", "data:" + img.mime_type + ";base64," + img.base64_data}}}
        });
    }
    content.push_back({{"type", "text"}, {"text", prompt}});

    nlohmann::json req = {
        {"model", config_.model},
        {"messages", {{{"role", "user"}, {"content", content}}}},
        {"temperature", config_.temperature},
        {"max_tokens", config_.max_tokens}
    };

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    std::string auth = "Authorization: Bearer " + config_.api_key;
    headers = curl_slist_append(headers, auth.c_str());

    std::string body = req.dump();
    auto response = httpPost(url, body, headers, abort_flag);
    curl_slist_free_all(headers);

    if (!response) return std::nullopt;

    try {
        auto j = nlohmann::json::parse(response.value());
        if (j.contains("choices") && !j["choices"].empty()) {
            return j["choices"][0]["message"]["content"].get<std::string>();
        }
    } catch (const std::exception& e) {
        std::cerr << "LLM: Failed to parse OpenAI vision response: " << e.what() << std::endl;
    }
    return std::nullopt;
}

// ─── GEMINI VISION ──────────────────────────────────────────────────────────
// POST with inline_data parts for images

std::optional<std::string> LLMClient::generateGeminiVision(const std::string& prompt,
                                                             const std::vector<LLMImage>& images,
                                                             const std::atomic<bool>* abort_flag) {
    std::string url = config_.endpoint + "/v1beta/models/" + config_.model +
                      ":generateContent?key=" + config_.api_key;

    nlohmann::json parts = nlohmann::json::array();
    for (const auto& img : images) {
        parts.push_back({{"inline_data", {{"mime_type", img.mime_type}, {"data", img.base64_data}}}});
    }
    parts.push_back({{"text", prompt}});

    nlohmann::json req = {
        {"contents", {{{"parts", parts}}}},
        {"generationConfig", {{"temperature", config_.temperature}, {"maxOutputTokens", config_.max_tokens}}}
    };

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    std::string body = req.dump();
    auto response = httpPost(url, body, headers, abort_flag);
    curl_slist_free_all(headers);

    if (!response) return std::nullopt;

    try {
        auto j = nlohmann::json::parse(response.value());
        if (j.contains("candidates") && !j["candidates"].empty()) {
            return j["candidates"][0]["content"]["parts"][0]["text"].get<std::string>();
        }
    } catch (const std::exception& e) {
        std::cerr << "LLM: Failed to parse Gemini vision response: " << e.what() << std::endl;
    }
    return std::nullopt;
}

// ─── ANTHROPIC VISION ───────────────────────────────────────────────────────
// POST /v1/messages with image content blocks

std::optional<std::string> LLMClient::generateAnthropicVision(const std::string& prompt,
                                                                const std::vector<LLMImage>& images,
                                                                const std::atomic<bool>* abort_flag) {
    std::string url = config_.endpoint + "/v1/messages";

    nlohmann::json content = nlohmann::json::array();
    for (const auto& img : images) {
        content.push_back({
            {"type", "image"},
            {"source", {{"type", "base64"}, {"media_type", img.mime_type}, {"data", img.base64_data}}}
        });
    }
    content.push_back({{"type", "text"}, {"text", prompt}});

    nlohmann::json req = {
        {"model", config_.model},
        {"max_tokens", config_.max_tokens},
        {"messages", {{{"role", "user"}, {"content", content}}}}
    };

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    std::string api_header = "x-api-key: " + config_.api_key;
    headers = curl_slist_append(headers, api_header.c_str());
    headers = curl_slist_append(headers, "anthropic-version: 2023-06-01");

    std::string body = req.dump();
    auto response = httpPost(url, body, headers, abort_flag);
    curl_slist_free_all(headers);

    if (!response) return std::nullopt;

    try {
        auto j = nlohmann::json::parse(response.value());
        if (j.contains("content") && !j["content"].empty()) {
            return j["content"][0]["text"].get<std::string>();
        }
    } catch (const std::exception& e) {
        std::cerr << "LLM: Failed to parse Anthropic vision response: " << e.what() << std::endl;
    }
    return std::nullopt;
}

// ─── Base64 Encode ──────────────────────────────────────────────────────────

std::string LLMClient::base64Encode(const std::vector<unsigned char>& data) {
    static constexpr char table[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    std::string encoded;
    encoded.reserve(((data.size() + 2) / 3) * 4);

    size_t i = 0;
    for (; i + 2 < data.size(); i += 3) {
        uint32_t n = (static_cast<uint32_t>(data[i]) << 16) |
                     (static_cast<uint32_t>(data[i + 1]) << 8) |
                      static_cast<uint32_t>(data[i + 2]);
        encoded += table[(n >> 18) & 0x3F];
        encoded += table[(n >> 12) & 0x3F];
        encoded += table[(n >> 6)  & 0x3F];
        encoded += table[n & 0x3F];
    }

    if (i + 1 == data.size()) {
        uint32_t n = static_cast<uint32_t>(data[i]) << 16;
        encoded += table[(n >> 18) & 0x3F];
        encoded += table[(n >> 12) & 0x3F];
        encoded += '=';
        encoded += '=';
    } else if (i + 2 == data.size()) {
        uint32_t n = (static_cast<uint32_t>(data[i]) << 16) |
                     (static_cast<uint32_t>(data[i + 1]) << 8);
        encoded += table[(n >> 18) & 0x3F];
        encoded += table[(n >> 12) & 0x3F];
        encoded += table[(n >> 6)  & 0x3F];
        encoded += '=';
    }

    return encoded;
}

// ─── generateWithTools (dispatch with timing and abort) ─────────────────────

LLMToolResponse LLMClient::generateWithTools(
    const std::vector<ChatMessage>& messages,
    const std::vector<ToolDefinition>& tools,
    const std::atomic<bool>* abort_flag) {

    LLMToolResponse result;
    auto start = std::chrono::steady_clock::now();

    if (abort_flag && abort_flag->load(std::memory_order_acquire)) {
        result.was_aborted = true;
        return result;
    }

    nlohmann::json req;
    std::string url;
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    switch (config_.provider) {
        case LLMProvider::OLLAMA: {
            url = config_.endpoint + "/api/chat";
            req["model"] = config_.model;
            req["stream"] = false;
            req["messages"] = tool_format::buildOllamaMessages(messages);
            req["tools"] = tool_format::buildOllamaTools(tools);
            req["options"] = {{"temperature", config_.temperature}};
            req["keep_alive"] = config_.keep_alive_seconds;
            break;
        }
        case LLMProvider::OPENAI: {
            url = config_.endpoint + "/v1/chat/completions";
            req["model"] = config_.model;
            req["messages"] = tool_format::buildOpenAIMessages(messages);
            req["tools"] = tool_format::buildOpenAITools(tools);
            req["temperature"] = config_.temperature;
            req["max_completion_tokens"] = config_.max_tokens;
            std::string auth = "Authorization: Bearer " + config_.api_key;
            headers = curl_slist_append(headers, auth.c_str());
            break;
        }
        case LLMProvider::ANTHROPIC: {
            url = config_.endpoint + "/v1/messages";
            auto am = tool_format::buildAnthropicMessages(messages);
            req["model"] = config_.model;
            req["max_tokens"] = config_.max_tokens;
            req["messages"] = am.messages;
            req["tools"] = tool_format::buildAnthropicTools(tools);
            if (!am.system_prompt.empty()) {
                req["system"] = am.system_prompt;
            }
            std::string api_header = "x-api-key: " + config_.api_key;
            headers = curl_slist_append(headers, api_header.c_str());
            headers = curl_slist_append(headers, "anthropic-version: 2023-06-01");
            break;
        }
        case LLMProvider::GEMINI: {
            url = config_.endpoint + "/v1beta/models/" + config_.model +
                  ":generateContent?key=" + config_.api_key;
            req["contents"] = tool_format::buildGeminiMessages(messages);
            req["tools"] = tool_format::buildGeminiTools(tools);
            req["generationConfig"] = {
                {"temperature", config_.temperature},
                {"maxOutputTokens", config_.max_tokens}
            };
            break;
        }
    }

    std::string body = req.dump();
    auto response = httpPost(url, body, headers, abort_flag);
    curl_slist_free_all(headers);

    auto end = std::chrono::steady_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(end - start).count();

    if (abort_flag && abort_flag->load(std::memory_order_acquire)) {
        result.was_aborted = true;
        return result;
    }

    if (!response) return result;

    try {
        auto j = nlohmann::json::parse(response.value());
        switch (config_.provider) {
            case LLMProvider::OLLAMA:    result = tool_format::parseOllamaToolResponse(j); break;
            case LLMProvider::OPENAI:    result = tool_format::parseOpenAIToolResponse(j); break;
            case LLMProvider::ANTHROPIC: result = tool_format::parseAnthropicToolResponse(j); break;
            case LLMProvider::GEMINI:    result = tool_format::parseGeminiToolResponse(j); break;
        }
    } catch (const std::exception& e) {
        std::cerr << "LLM: Failed to parse tool response from "
                  << providerName(config_.provider) << ": " << e.what() << std::endl;
    }

    // Restore timing (parsing functions don't set it)
    result.elapsed_seconds = std::chrono::duration<double>(end - start).count();
    return result;
}

// ─── embed ──────────────────────────────────────────────────────────────────

std::vector<float> LLMClient::embed(const std::string& text) {
    return embed(text, config_.model);
}

std::vector<float> LLMClient::embed(const std::string& text, const std::string& model) {
    nlohmann::json req;
    std::string url;
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    switch (config_.provider) {
        case LLMProvider::OLLAMA: {
            url = config_.endpoint + "/api/embeddings";
            req["model"] = model;
            req["prompt"] = text;
            break;
        }
        case LLMProvider::OPENAI: {
            url = config_.endpoint + "/v1/embeddings";
            req["model"] = model;
            req["input"] = text;
            std::string auth = "Authorization: Bearer " + config_.api_key;
            headers = curl_slist_append(headers, auth.c_str());
            break;
        }
        case LLMProvider::ANTHROPIC:
        case LLMProvider::GEMINI:
            curl_slist_free_all(headers);
            throw std::runtime_error(
                "LLM: embed() not supported for " + providerName(config_.provider) +
                ". Use a separate Ollama or OpenAI-backed LLMClient for embeddings.");
    }

    std::string body = req.dump();
    auto response = httpPost(url, body, headers);
    curl_slist_free_all(headers);

    if (!response) {
        throw std::runtime_error("LLM: embed() HTTP request failed");
    }

    try {
        auto j = nlohmann::json::parse(response.value());
        switch (config_.provider) {
            case LLMProvider::OLLAMA:  return tool_format::parseOllamaEmbedding(j);
            case LLMProvider::OPENAI:  return tool_format::parseOpenAIEmbedding(j);
            default: break;
        }
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error(std::string("LLM: embed() parse error: ") + e.what());
    }

    return {};
}

// ─── toVectorLiteral ────────────────────────────────────────────────────────

std::string LLMClient::toVectorLiteral(const std::vector<float>& vec) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) oss << ",";
        oss << vec[i];
    }
    oss << "]";
    return oss.str();
}

// ─── Force Unload Model ─────────────────────────────────────────────────────

void LLMClient::forceUnloadModel(const std::string& ollama_endpoint,
                                  const std::string& model_name) {
    nlohmann::json body = {
        {"model", model_name},
        {"keep_alive", 0}
    };
    std::string body_str = body.dump();
    std::string url = ollama_endpoint + "/api/generate";
    std::string response_body;

    CURL* curl = curl_easy_init();
    if (!curl) return;

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body_str.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(body_str.size()));
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 3L);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);

    auto res = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res == CURLE_OK) {
        std::cerr << "LLM: force-unloaded " << model_name << " from Ollama" << std::endl;
    } else {
        std::cerr << "LLM: failed to force-unload " << model_name << ": "
                  << curl_easy_strerror(res) << std::endl;
    }
}

} // namespace hms

// ═══════════════════════════════════════════════════════════════════════════
// tool_format — serialization and parsing helpers (testable via llm_tool_format.h)
// ═══════════════════════════════════════════════════════════════════════════

namespace hms::tool_format {

using json = nlohmann::json;

// ─── Tool serialization ────────────────────────────────────────────────────

json buildOllamaTools(const std::vector<ToolDefinition>& tools) {
    json arr = json::array();
    for (const auto& t : tools) {
        arr.push_back({
            {"type", "function"},
            {"function", {
                {"name", t.name},
                {"description", t.description},
                {"parameters", t.parameters}
            }}
        });
    }
    return arr;
}

json buildOpenAITools(const std::vector<ToolDefinition>& tools) {
    // Same format as Ollama (OpenAI-compatible)
    return buildOllamaTools(tools);
}

json buildAnthropicTools(const std::vector<ToolDefinition>& tools) {
    json arr = json::array();
    for (const auto& t : tools) {
        arr.push_back({
            {"name", t.name},
            {"description", t.description},
            {"input_schema", t.parameters}
        });
    }
    return arr;
}

json buildGeminiTools(const std::vector<ToolDefinition>& tools) {
    json decls = json::array();
    for (const auto& t : tools) {
        decls.push_back({
            {"name", t.name},
            {"description", t.description},
            {"parameters", t.parameters}
        });
    }
    return json::array({{{"functionDeclarations", decls}}});
}

// ─── Message serialization ─────────────────────────────────────────────────

static json serializeToolCalls_OpenAI(const std::vector<ToolCall>& calls) {
    json arr = json::array();
    for (const auto& tc : calls) {
        arr.push_back({
            {"id", tc.id},
            {"type", "function"},
            {"function", {
                {"name", tc.name},
                {"arguments", tc.arguments.dump()}
            }}
        });
    }
    return arr;
}

json buildOllamaMessages(const std::vector<ChatMessage>& messages) {
    json arr = json::array();
    for (const auto& m : messages) {
        json msg = {{"role", m.role}, {"content", m.content}};
        if (!m.tool_calls.empty()) {
            msg["tool_calls"] = serializeToolCalls_OpenAI(m.tool_calls);
        }
        arr.push_back(msg);
    }
    return arr;
}

json buildOpenAIMessages(const std::vector<ChatMessage>& messages) {
    json arr = json::array();
    for (const auto& m : messages) {
        json msg = {{"role", m.role}, {"content", m.content}};
        if (!m.tool_calls.empty()) {
            msg["tool_calls"] = serializeToolCalls_OpenAI(m.tool_calls);
        }
        if (!m.tool_call_id.empty()) {
            msg["tool_call_id"] = m.tool_call_id;
        }
        arr.push_back(msg);
    }
    return arr;
}

AnthropicMessageResult buildAnthropicMessages(const std::vector<ChatMessage>& messages) {
    AnthropicMessageResult result;
    result.messages = json::array();

    for (const auto& m : messages) {
        if (m.role == "system") {
            result.system_prompt = m.content;
            continue;
        }

        if (m.role == "assistant" && !m.tool_calls.empty()) {
            json content = json::array();
            if (!m.content.empty()) {
                content.push_back({{"type", "text"}, {"text", m.content}});
            }
            for (const auto& tc : m.tool_calls) {
                content.push_back({
                    {"type", "tool_use"},
                    {"id", tc.id},
                    {"name", tc.name},
                    {"input", tc.arguments}
                });
            }
            result.messages.push_back({{"role", "assistant"}, {"content", content}});
        } else if (m.role == "tool") {
            // Anthropic: tool results go as user messages with content blocks
            json content = json::array();
            content.push_back({
                {"type", "tool_result"},
                {"tool_use_id", m.tool_call_id},
                {"content", m.content}
            });
            result.messages.push_back({{"role", "user"}, {"content", content}});
        } else {
            result.messages.push_back({{"role", m.role}, {"content", m.content}});
        }
    }
    return result;
}

json buildGeminiMessages(const std::vector<ChatMessage>& messages) {
    json arr = json::array();
    for (const auto& m : messages) {
        if (m.role == "system") continue;  // Gemini uses systemInstruction, handled separately

        std::string role = (m.role == "assistant") ? "model" : m.role;

        if (m.role == "tool") {
            // Gemini: tool results are function role with functionResponse parts
            json parts = json::array();
            json response_content;
            try {
                response_content = json::parse(m.content);
            } catch (...) {
                response_content = {{"result", m.content}};
            }
            parts.push_back({
                {"functionResponse", {
                    {"name", m.tool_call_id},  // tool_call_id stores function name for Gemini
                    {"response", response_content}
                }}
            });
            arr.push_back({{"role", "function"}, {"parts", parts}});
        } else if (m.role == "assistant" && !m.tool_calls.empty()) {
            json parts = json::array();
            if (!m.content.empty()) {
                parts.push_back({{"text", m.content}});
            }
            for (const auto& tc : m.tool_calls) {
                parts.push_back({
                    {"functionCall", {
                        {"name", tc.name},
                        {"args", tc.arguments}
                    }}
                });
            }
            arr.push_back({{"role", "model"}, {"parts", parts}});
        } else {
            arr.push_back({
                {"role", role},
                {"parts", json::array({{{"text", m.content}}})}
            });
        }
    }
    return arr;
}

// ─── Response parsing ──────────────────────────────────────────────────────

LLMToolResponse parseOllamaToolResponse(const json& j) {
    LLMToolResponse result;

    if (j.contains("message")) {
        const auto& msg = j["message"];
        if (msg.contains("content") && !msg["content"].get<std::string>().empty()) {
            result.text = msg["content"].get<std::string>();
        }
        if (msg.contains("tool_calls")) {
            for (const auto& tc : msg["tool_calls"]) {
                ToolCall call;
                call.name = tc["function"]["name"].get<std::string>();
                call.arguments = tc["function"]["arguments"];
                if (tc.contains("id")) {
                    call.id = tc["id"].get<std::string>();
                }
                result.tool_calls.push_back(std::move(call));
            }
        }
    }
    result.stop_reason = result.tool_calls.empty() ? "stop" : "tool_calls";
    return result;
}

LLMToolResponse parseOpenAIToolResponse(const json& j) {
    LLMToolResponse result;

    if (j.contains("choices") && !j["choices"].empty()) {
        const auto& choice = j["choices"][0];
        if (choice.contains("finish_reason")) {
            result.stop_reason = choice["finish_reason"].get<std::string>();
        }
        if (choice.contains("message")) {
            const auto& msg = choice["message"];
            if (msg.contains("content") && !msg["content"].is_null()) {
                result.text = msg["content"].get<std::string>();
            }
            if (msg.contains("tool_calls")) {
                for (const auto& tc : msg["tool_calls"]) {
                    ToolCall call;
                    call.id = tc["id"].get<std::string>();
                    call.name = tc["function"]["name"].get<std::string>();
                    // OpenAI returns arguments as a string
                    std::string args_str = tc["function"]["arguments"].get<std::string>();
                    call.arguments = json::parse(args_str);
                    result.tool_calls.push_back(std::move(call));
                }
            }
        }
    }
    return result;
}

LLMToolResponse parseAnthropicToolResponse(const json& j) {
    LLMToolResponse result;

    if (j.contains("stop_reason")) {
        result.stop_reason = j["stop_reason"].get<std::string>();
    }
    if (j.contains("content")) {
        for (const auto& block : j["content"]) {
            if (block["type"] == "text") {
                result.text = block["text"].get<std::string>();
            } else if (block["type"] == "tool_use") {
                ToolCall call;
                call.id = block["id"].get<std::string>();
                call.name = block["name"].get<std::string>();
                call.arguments = block["input"];
                result.tool_calls.push_back(std::move(call));
            }
        }
    }
    return result;
}

LLMToolResponse parseGeminiToolResponse(const json& j) {
    LLMToolResponse result;

    if (j.contains("candidates") && !j["candidates"].empty()) {
        const auto& candidate = j["candidates"][0];
        if (candidate.contains("finishReason")) {
            result.stop_reason = candidate["finishReason"].get<std::string>();
        }
        if (candidate.contains("content") && candidate["content"].contains("parts")) {
            for (const auto& part : candidate["content"]["parts"]) {
                if (part.contains("text")) {
                    result.text = part["text"].get<std::string>();
                } else if (part.contains("functionCall")) {
                    ToolCall call;
                    call.name = part["functionCall"]["name"].get<std::string>();
                    call.arguments = part["functionCall"]["args"];
                    result.tool_calls.push_back(std::move(call));
                }
            }
        }
    }
    return result;
}

// ─── Embedding response parsing ────────────────────────────────────────────

std::vector<float> parseOllamaEmbedding(const json& j) {
    std::vector<float> result;
    if (j.contains("embedding")) {
        const auto& emb = j["embedding"];
        result.reserve(emb.size());
        for (const auto& v : emb) {
            result.push_back(v.get<float>());
        }
    }
    return result;
}

std::vector<float> parseOpenAIEmbedding(const json& j) {
    std::vector<float> result;
    if (j.contains("data") && !j["data"].empty()) {
        const auto& emb = j["data"][0]["embedding"];
        result.reserve(emb.size());
        for (const auto& v : emb) {
            result.push_back(v.get<float>());
        }
    }
    return result;
}

} // namespace hms::tool_format
