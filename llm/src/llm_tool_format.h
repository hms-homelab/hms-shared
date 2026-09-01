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

/**
 * Set the provider's "you must call this tool" field on an outgoing request.
 *
 * Every provider spells it differently and Gemini nests it under a different
 * top-level key entirely, so the shape lives here with the other provider
 * differences rather than inline in generateWithTools -- which cannot be tested
 * without a live HTTP call, and this is exactly the kind of per-provider detail
 * that is wrong silently.
 *
 * A no-op when `tool` is empty, and A NO-OP FOR OLLAMA, whose /api/chat has no
 * equivalent. The caller cannot tell those two apart from the request body, so
 * LLMClient::supportsForcedTool() is what answers "will this actually force".
 */
void applyToolChoice(nlohmann::json& req, LLMProvider provider,
                     const std::string& tool);

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

// ─── Stream parsing ────────────────────────────────────────────────────────

/**
 * Extract the text delta carried by ONE raw line of a streaming response.
 *
 * Every provider here is line-oriented: Ollama sends NDJSON, the other three
 * send SSE. This function owns the differences between them, which is what
 * makes streaming testable against captured lines with no HTTP involved.
 *
 * Returns nullopt for every line that carries no text, and there are a lot of
 * those: blank separators, SSE comments, `event:` lines, OpenAI's `[DONE]`
 * sentinel, Ollama's final `done:true` frame, Anthropic's non-text events, and
 * anything that fails to parse. A caller can therefore treat nullopt as
 * "nothing to emit" and never has to know why.
 */
std::optional<std::string> parseStreamLine(LLMProvider provider, const std::string& line);

// ─── Embedding response parsing ────────────────────────────────────────────

std::vector<float> parseOllamaEmbedding(const nlohmann::json& j);
std::vector<float> parseOpenAIEmbedding(const nlohmann::json& j);

} // namespace hms::tool_format
