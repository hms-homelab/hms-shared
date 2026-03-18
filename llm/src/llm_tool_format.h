#pragma once

// Internal header — exposes tool/message serialization and response parsing
// so that llm_client_test.cpp can unit-test them without live HTTP calls.

#include "llm_client.h"

#include <nlohmann/json.hpp>

namespace hms::tool_format {

// ─── Tool serialization ────────────────────────────────────────────────────

nlohmann::json buildOllamaTools(const std::vector<ToolDefinition>& tools);
nlohmann::json buildOpenAITools(const std::vector<ToolDefinition>& tools);
nlohmann::json buildAnthropicTools(const std::vector<ToolDefinition>& tools);
nlohmann::json buildGeminiTools(const std::vector<ToolDefinition>& tools);

// ─── Message serialization ─────────────────────────────────────────────────

nlohmann::json buildOllamaMessages(const std::vector<ChatMessage>& messages);
nlohmann::json buildOpenAIMessages(const std::vector<ChatMessage>& messages);

// Returns {system, messages} pair — system prompt extracted from messages
struct AnthropicMessageResult {
    std::string system_prompt;
    nlohmann::json messages;
};
AnthropicMessageResult buildAnthropicMessages(const std::vector<ChatMessage>& messages);

nlohmann::json buildGeminiMessages(const std::vector<ChatMessage>& messages);

// ─── Response parsing ──────────────────────────────────────────────────────

LLMToolResponse parseOllamaToolResponse(const nlohmann::json& j);
LLMToolResponse parseOpenAIToolResponse(const nlohmann::json& j);
LLMToolResponse parseAnthropicToolResponse(const nlohmann::json& j);
LLMToolResponse parseGeminiToolResponse(const nlohmann::json& j);

// ─── Embedding response parsing ────────────────────────────────────────────

std::vector<float> parseOllamaEmbedding(const nlohmann::json& j);
std::vector<float> parseOpenAIEmbedding(const nlohmann::json& j);

} // namespace hms::tool_format
