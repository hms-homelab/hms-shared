#include "llm_client.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

// Use nlohmann/json if available (hms-shared already depends on it),
// otherwise fall back to jsoncpp
#include <nlohmann/json.hpp>

namespace hms {

// ─── Construction ────────────────────────────────────────────────────────────

LLMClient::LLMClient(const LLMConfig& config) : config_(config) {
    curl_ = curl_easy_init();
}

LLMClient::~LLMClient() {
    if (curl_) curl_easy_cleanup(curl_);
}

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
    if (!curl_) {
        std::cerr << "LLM: curl not initialized" << std::endl;
        return std::nullopt;
    }

    switch (config_.provider) {
        case LLMProvider::OLLAMA:    return generateOllama(prompt);
        case LLMProvider::OPENAI:    return generateOpenAI(prompt);
        case LLMProvider::GEMINI:    return generateGemini(prompt);
        case LLMProvider::ANTHROPIC: return generateAnthropic(prompt);
    }
    return std::nullopt;
}

// ─── HTTP POST ──────────────────────────────────────────────────────────────

std::optional<std::string> LLMClient::httpPost(const std::string& url,
                                                const std::string& body,
                                                struct curl_slist* headers) {
    std::string response;

    curl_easy_reset(curl_);
    curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl_, CURLOPT_POST, 1L);
    curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl_, CURLOPT_POSTFIELDSIZE, static_cast<long>(body.size()));
    curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl_, CURLOPT_TIMEOUT, config_.timeout_seconds);
    curl_easy_setopt(curl_, CURLOPT_CONNECTTIMEOUT, config_.connect_timeout_seconds);

    CURLcode res = curl_easy_perform(curl_);
    if (res != CURLE_OK) {
        std::cerr << "LLM: HTTP request failed: " << curl_easy_strerror(res) << std::endl;
        return std::nullopt;
    }

    long http_code = 0;
    curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &http_code);
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
        {"max_tokens", config_.max_tokens}
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

} // namespace hms
