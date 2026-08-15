#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "llm_client.h"
#include "../src/llm_tool_format.h"

using namespace hms;
using namespace hms::tool_format;
using json = nlohmann::json;

// ═══════════════════════════════════════════════════════════════════════════
// toVectorLiteral
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("toVectorLiteral — empty vector", "[llm][embed]") {
    REQUIRE(LLMClient::toVectorLiteral({}) == "[]");
}

TEST_CASE("toVectorLiteral — single element", "[llm][embed]") {
    auto s = LLMClient::toVectorLiteral({0.5f});
    REQUIRE(s.front() == '[');
    REQUIRE(s.back() == ']');
    REQUIRE(s.find(',') == std::string::npos);
}

TEST_CASE("toVectorLiteral — multiple elements", "[llm][embed]") {
    auto s = LLMClient::toVectorLiteral({0.1f, 0.2f, 0.3f});
    REQUIRE(s.front() == '[');
    REQUIRE(s.back() == ']');
    // Should have exactly 2 commas
    REQUIRE(std::count(s.begin(), s.end(), ',') == 2);
}

// ═══════════════════════════════════════════════════════════════════════════
// Tool serialization
// ═══════════════════════════════════════════════════════════════════════════

static ToolDefinition sampleTool() {
    return {"get_weather", "Get current weather", json({
        {"type", "object"},
        {"properties", {{"city", {{"type", "string"}}}}},
        {"required", json::array({"city"})}
    })};
}

TEST_CASE("buildOllamaTools — OpenAI-compatible format", "[llm][tools]") {
    auto arr = buildOllamaTools({sampleTool()});
    REQUIRE(arr.size() == 1);
    REQUIRE(arr[0]["type"] == "function");
    REQUIRE(arr[0]["function"]["name"] == "get_weather");
    REQUIRE(arr[0]["function"]["description"] == "Get current weather");
    REQUIRE(arr[0]["function"]["parameters"]["type"] == "object");
}

TEST_CASE("buildOpenAITools — same as Ollama", "[llm][tools]") {
    auto ollama = buildOllamaTools({sampleTool()});
    auto openai = buildOpenAITools({sampleTool()});
    REQUIRE(ollama == openai);
}

TEST_CASE("buildAnthropicTools — uses input_schema", "[llm][tools]") {
    auto arr = buildAnthropicTools({sampleTool()});
    REQUIRE(arr.size() == 1);
    REQUIRE(arr[0].contains("input_schema"));
    REQUIRE(!arr[0].contains("parameters"));
    REQUIRE(arr[0]["name"] == "get_weather");
}

TEST_CASE("buildGeminiTools — wrapped in functionDeclarations", "[llm][tools]") {
    auto arr = buildGeminiTools({sampleTool()});
    REQUIRE(arr.size() == 1);
    REQUIRE(arr[0].contains("functionDeclarations"));
    auto decls = arr[0]["functionDeclarations"];
    REQUIRE(decls.size() == 1);
    REQUIRE(decls[0]["name"] == "get_weather");
}

// ═══════════════════════════════════════════════════════════════════════════
// Message serialization
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("buildOllamaMessages — basic roles", "[llm][messages]") {
    std::vector<ChatMessage> msgs = {
        {"system", "You are helpful.", {}, ""},
        {"user", "Hello", {}, ""},
    };
    auto arr = buildOllamaMessages(msgs);
    REQUIRE(arr.size() == 2);
    REQUIRE(arr[0]["role"] == "system");
    REQUIRE(arr[1]["role"] == "user");
    REQUIRE(arr[1]["content"] == "Hello");
}

TEST_CASE("buildOllamaMessages — tool_calls arguments stay an OBJECT", "[llm][messages]") {
    // Ollama rejects the whole request when `arguments` is a JSON string, with
    // an error that reads like a malformed body rather than a type mismatch:
    //   {"error":"Value looks like object, but can't find closing '}' symbol"}
    // The first round never carries tool_calls, so this only breaks on the
    // SECOND round, which is why it went unnoticed until an agent loop ran.
    ChatMessage assistant;
    assistant.role = "assistant";
    assistant.content = "";
    assistant.tool_calls = {{"call_1", "get_night", json({{"date", "2026-08-14"}})}};

    auto arr = buildOllamaMessages({assistant});
    REQUIRE(arr.size() == 1);
    const auto& args = arr[0]["tool_calls"][0]["function"]["arguments"];
    REQUIRE(args.is_object());
    REQUIRE_FALSE(args.is_string());
    REQUIRE(args["date"] == "2026-08-14");
}

TEST_CASE("buildOpenAIMessages — tool_calls arguments stay a STRING", "[llm][messages]") {
    // The mirror of the case above: OpenAI wants the opposite, so the two
    // builders must NOT be collapsed back into one.
    ChatMessage assistant;
    assistant.role = "assistant";
    assistant.tool_calls = {{"call_1", "get_night", json({{"date", "2026-08-14"}})}};

    auto arr = buildOpenAIMessages({assistant});
    const auto& args = arr[0]["tool_calls"][0]["function"]["arguments"];
    REQUIRE(args.is_string());
    REQUIRE(args.get<std::string>().find("2026-08-14") != std::string::npos);
}

TEST_CASE("buildOpenAIMessages — tool result includes tool_call_id", "[llm][messages]") {
    std::vector<ChatMessage> msgs = {
        {"tool", "{\"temp\": 72}", {}, "call_123"},
    };
    auto arr = buildOpenAIMessages(msgs);
    REQUIRE(arr[0]["tool_call_id"] == "call_123");
    REQUIRE(arr[0]["role"] == "tool");
}

TEST_CASE("buildAnthropicMessages — system extracted", "[llm][messages]") {
    std::vector<ChatMessage> msgs = {
        {"system", "Be concise.", {}, ""},
        {"user", "Hi", {}, ""},
    };
    auto result = buildAnthropicMessages(msgs);
    REQUIRE(result.system_prompt == "Be concise.");
    REQUIRE(result.messages.size() == 1);
    REQUIRE(result.messages[0]["role"] == "user");
}

TEST_CASE("buildAnthropicMessages — tool result as user content block", "[llm][messages]") {
    std::vector<ChatMessage> msgs = {
        {"tool", "72 degrees", {}, "toolu_abc"},
    };
    auto result = buildAnthropicMessages(msgs);
    REQUIRE(result.messages.size() == 1);
    REQUIRE(result.messages[0]["role"] == "user");
    auto content = result.messages[0]["content"];
    REQUIRE(content[0]["type"] == "tool_result");
    REQUIRE(content[0]["tool_use_id"] == "toolu_abc");
}

TEST_CASE("buildAnthropicMessages — assistant with tool_calls", "[llm][messages]") {
    ToolCall tc;
    tc.id = "toolu_123";
    tc.name = "get_weather";
    tc.arguments = {{"city", "NYC"}};
    std::vector<ChatMessage> msgs = {
        {"assistant", "", {tc}, ""},
    };
    auto result = buildAnthropicMessages(msgs);
    auto content = result.messages[0]["content"];
    REQUIRE(content.size() == 1);
    REQUIRE(content[0]["type"] == "tool_use");
    REQUIRE(content[0]["id"] == "toolu_123");
    REQUIRE(content[0]["name"] == "get_weather");
    REQUIRE(content[0]["input"]["city"] == "NYC");
}

TEST_CASE("buildGeminiMessages — role mapping", "[llm][messages]") {
    std::vector<ChatMessage> msgs = {
        {"user", "Hi", {}, ""},
        {"assistant", "Hello!", {}, ""},
    };
    auto arr = buildGeminiMessages(msgs);
    REQUIRE(arr[0]["role"] == "user");
    REQUIRE(arr[1]["role"] == "model");
    REQUIRE(arr[1]["parts"][0]["text"] == "Hello!");
}

TEST_CASE("buildGeminiMessages — system messages skipped", "[llm][messages]") {
    std::vector<ChatMessage> msgs = {
        {"system", "Ignored", {}, ""},
        {"user", "Hi", {}, ""},
    };
    auto arr = buildGeminiMessages(msgs);
    REQUIRE(arr.size() == 1);
    REQUIRE(arr[0]["role"] == "user");
}

TEST_CASE("buildGeminiMessages — tool result as functionResponse", "[llm][messages]") {
    std::vector<ChatMessage> msgs = {
        {"tool", "{\"temp\": 72}", {}, "get_weather"},
    };
    auto arr = buildGeminiMessages(msgs);
    REQUIRE(arr[0]["role"] == "function");
    REQUIRE(arr[0]["parts"][0]["functionResponse"]["name"] == "get_weather");
    REQUIRE(arr[0]["parts"][0]["functionResponse"]["response"]["temp"] == 72);
}

// ═══════════════════════════════════════════════════════════════════════════
// Response parsing
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("parseOllamaToolResponse — text only", "[llm][parse]") {
    json j = {{"message", {{"role", "assistant"}, {"content", "Hello!"}}}};
    auto r = parseOllamaToolResponse(j);
    REQUIRE(r.text == "Hello!");
    REQUIRE(r.tool_calls.empty());
    REQUIRE(r.stop_reason == "stop");
}

TEST_CASE("parseOllamaToolResponse — tool call", "[llm][parse]") {
    json j = {{"message", {
        {"role", "assistant"},
        {"content", ""},
        {"tool_calls", json::array({
            {{"function", {{"name", "get_weather"}, {"arguments", {{"city", "NYC"}}}}}}
        })}
    }}};
    auto r = parseOllamaToolResponse(j);
    REQUIRE(r.tool_calls.size() == 1);
    REQUIRE(r.tool_calls[0].name == "get_weather");
    REQUIRE(r.tool_calls[0].arguments["city"] == "NYC");
    REQUIRE(r.stop_reason == "tool_calls");
}

TEST_CASE("parseOpenAIToolResponse — tool call with string arguments", "[llm][parse]") {
    json j = {{"choices", json::array({
        {{"finish_reason", "tool_calls"},
         {"message", {
            {"role", "assistant"},
            {"content", nullptr},
            {"tool_calls", json::array({
                {{"id", "call_abc"}, {"type", "function"},
                 {"function", {{"name", "get_weather"}, {"arguments", "{\"city\":\"NYC\"}"}}}}
            })}
         }}}
    })}};
    auto r = parseOpenAIToolResponse(j);
    REQUIRE(r.tool_calls.size() == 1);
    REQUIRE(r.tool_calls[0].id == "call_abc");
    REQUIRE(r.tool_calls[0].name == "get_weather");
    REQUIRE(r.tool_calls[0].arguments["city"] == "NYC");
    REQUIRE(r.stop_reason == "tool_calls");
}

TEST_CASE("parseOpenAIToolResponse — text response", "[llm][parse]") {
    json j = {{"choices", json::array({
        {{"finish_reason", "stop"},
         {"message", {{"role", "assistant"}, {"content", "The weather is 72F."}}}}
    })}};
    auto r = parseOpenAIToolResponse(j);
    REQUIRE(r.text == "The weather is 72F.");
    REQUIRE(r.tool_calls.empty());
    REQUIRE(r.stop_reason == "stop");
}

TEST_CASE("parseAnthropicToolResponse — tool_use block", "[llm][parse]") {
    json j = {
        {"stop_reason", "tool_use"},
        {"content", json::array({
            {{"type", "text"}, {"text", "Let me check the weather."}},
            {{"type", "tool_use"}, {"id", "toolu_abc"}, {"name", "get_weather"},
             {"input", {{"city", "NYC"}}}}
        })}
    };
    auto r = parseAnthropicToolResponse(j);
    REQUIRE(r.text == "Let me check the weather.");
    REQUIRE(r.tool_calls.size() == 1);
    REQUIRE(r.tool_calls[0].id == "toolu_abc");
    REQUIRE(r.tool_calls[0].name == "get_weather");
    REQUIRE(r.stop_reason == "tool_use");
}

TEST_CASE("parseAnthropicToolResponse — end_turn text only", "[llm][parse]") {
    json j = {
        {"stop_reason", "end_turn"},
        {"content", json::array({
            {{"type", "text"}, {"text", "The weather is 72F."}}
        })}
    };
    auto r = parseAnthropicToolResponse(j);
    REQUIRE(r.text == "The weather is 72F.");
    REQUIRE(r.tool_calls.empty());
    REQUIRE(r.stop_reason == "end_turn");
}

TEST_CASE("parseGeminiToolResponse — functionCall", "[llm][parse]") {
    json j = {{"candidates", json::array({
        {{"finishReason", "STOP"},
         {"content", {{"parts", json::array({
            {{"functionCall", {{"name", "get_weather"}, {"args", {{"city", "NYC"}}}}}}
         })}}}}
    })}};
    auto r = parseGeminiToolResponse(j);
    REQUIRE(r.tool_calls.size() == 1);
    REQUIRE(r.tool_calls[0].name == "get_weather");
    REQUIRE(r.tool_calls[0].arguments["city"] == "NYC");
}

TEST_CASE("parseGeminiToolResponse — text", "[llm][parse]") {
    json j = {{"candidates", json::array({
        {{"finishReason", "STOP"},
         {"content", {{"parts", json::array({
            {{"text", "72 degrees"}}
         })}}}}
    })}};
    auto r = parseGeminiToolResponse(j);
    REQUIRE(r.text == "72 degrees");
    REQUIRE(r.tool_calls.empty());
}

// ═══════════════════════════════════════════════════════════════════════════
// Embedding response parsing
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("parseOllamaEmbedding — extracts float vector", "[llm][embed]") {
    json j = {{"embedding", json::array({0.1, 0.2, 0.3})}};
    auto vec = parseOllamaEmbedding(j);
    REQUIRE(vec.size() == 3);
    REQUIRE_THAT(vec[0], Catch::Matchers::WithinAbs(0.1, 0.001));
    REQUIRE_THAT(vec[2], Catch::Matchers::WithinAbs(0.3, 0.001));
}

TEST_CASE("parseOllamaEmbedding — empty when no embedding key", "[llm][embed]") {
    json j = {{"error", "model not found"}};
    auto vec = parseOllamaEmbedding(j);
    REQUIRE(vec.empty());
}

TEST_CASE("parseOpenAIEmbedding — extracts from data[0].embedding", "[llm][embed]") {
    json j = {{"data", json::array({
        {{"embedding", json::array({0.4, 0.5, 0.6})}, {"index", 0}}
    })}};
    auto vec = parseOpenAIEmbedding(j);
    REQUIRE(vec.size() == 3);
    REQUIRE_THAT(vec[1], Catch::Matchers::WithinAbs(0.5, 0.001));
}

TEST_CASE("parseOpenAIEmbedding — empty when no data", "[llm][embed]") {
    json j = {{"error", {{"message", "invalid"}}}};
    auto vec = parseOpenAIEmbedding(j);
    REQUIRE(vec.empty());
}

// ═══════════════════════════════════════════════════════════════════════════
// Stream parsing
//
// Frames below are shaped as the providers actually send them. The cases that
// matter most are the ones that must NOT yield text: a stream carries far more
// framing than prose, and anything mistaken for prose ends up on a user's
// screen.
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("parseStreamLine — Ollama content frame", "[llm][stream]") {
    auto d = parseStreamLine(LLMProvider::OLLAMA,
        R"({"model":"gpt-oss:120b-cloud","message":{"role":"assistant","content":"Your AHI"},"done":false})");
    REQUIRE(d.has_value());
    REQUIRE(*d == "Your AHI");
}

TEST_CASE("parseStreamLine — Ollama final done frame carries no text", "[llm][stream]") {
    auto d = parseStreamLine(LLMProvider::OLLAMA,
        R"({"model":"m","message":{"role":"assistant","content":""},"done":true,"total_duration":51})");
    REQUIRE_FALSE(d.has_value());
}

TEST_CASE("parseStreamLine — Ollama thinking field is NOT answer text", "[llm][stream]") {
    // Captured from gpt-oss:120b on the relay 2026-08-14. Reasoning models on
    // Ollama emit `thinking` alongside an EMPTY `content` for the whole
    // reasoning phase (119 such frames in one measured turn). Emitting it would
    // put the model's private planning in front of the user.
    auto d = parseStreamLine(LLMProvider::OLLAMA,
        R"({"model":"gpt-oss:120b","message":{"role":"assistant","content":"","thinking":"We just need to respond"},"done":false})");
    REQUIRE_FALSE(d.has_value());
}

TEST_CASE("parseStreamLine — Ollama takes no SSE framing", "[llm][stream]") {
    // Ollama is NDJSON. A data: prefix would mean we pointed the wrong parser
    // at the stream, and it must not silently half-work.
    auto d = parseStreamLine(LLMProvider::OLLAMA,
        R"(data: {"message":{"content":"hi"}})");
    REQUIRE_FALSE(d.has_value());
}

TEST_CASE("parseStreamLine — OpenAI delta", "[llm][stream]") {
    auto d = parseStreamLine(LLMProvider::OPENAI,
        R"(data: {"choices":[{"index":0,"delta":{"content":"was 4.1"},"finish_reason":null}]})");
    REQUIRE(d.has_value());
    REQUIRE(*d == "was 4.1");
}

TEST_CASE("parseStreamLine — OpenAI role-only opening frame", "[llm][stream]") {
    auto d = parseStreamLine(LLMProvider::OPENAI,
        R"(data: {"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]})");
    REQUIRE_FALSE(d.has_value());
}

TEST_CASE("parseStreamLine — OpenAI [DONE] sentinel is not JSON", "[llm][stream]") {
    REQUIRE_FALSE(parseStreamLine(LLMProvider::OPENAI, "data: [DONE]").has_value());
}

TEST_CASE("parseStreamLine — OpenAI finish frame", "[llm][stream]") {
    auto d = parseStreamLine(LLMProvider::OPENAI,
        R"(data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]})");
    REQUIRE_FALSE(d.has_value());
}

TEST_CASE("parseStreamLine — Anthropic text_delta", "[llm][stream]") {
    auto d = parseStreamLine(LLMProvider::ANTHROPIC,
        R"(data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"last night"}})");
    REQUIRE(d.has_value());
    REQUIRE(*d == "last night");
}

TEST_CASE("parseStreamLine — Anthropic thinking_delta is NOT answer text", "[llm][stream]") {
    // Rides the same channel as prose. Emitting it would put the model's
    // reasoning in front of the user.
    auto d = parseStreamLine(LLMProvider::ANTHROPIC,
        R"(data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}})");
    REQUIRE_FALSE(d.has_value());
}

TEST_CASE("parseStreamLine — Anthropic input_json_delta is NOT answer text", "[llm][stream]") {
    auto d = parseStreamLine(LLMProvider::ANTHROPIC,
        R"(data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"a\":"}})");
    REQUIRE_FALSE(d.has_value());
}

TEST_CASE("parseStreamLine — Anthropic non-delta events", "[llm][stream]") {
    REQUIRE_FALSE(parseStreamLine(LLMProvider::ANTHROPIC,
        R"(data: {"type":"message_start","message":{"id":"msg_1"}})").has_value());
    REQUIRE_FALSE(parseStreamLine(LLMProvider::ANTHROPIC,
        R"(data: {"type":"message_stop"})").has_value());
    REQUIRE_FALSE(parseStreamLine(LLMProvider::ANTHROPIC, "event: content_block_delta").has_value());
}

TEST_CASE("parseStreamLine — Gemini single part", "[llm][stream]") {
    auto d = parseStreamLine(LLMProvider::GEMINI,
        R"(data: {"candidates":[{"content":{"parts":[{"text":"Tuesday"}],"role":"model"}}]})");
    REQUIRE(d.has_value());
    REQUIRE(*d == "Tuesday");
}

TEST_CASE("parseStreamLine — Gemini concatenates multiple parts in order", "[llm][stream]") {
    auto d = parseStreamLine(LLMProvider::GEMINI,
        R"(data: {"candidates":[{"content":{"parts":[{"text":"Tues"},{"text":"day"}]}}]})");
    REQUIRE(d.has_value());
    REQUIRE(*d == "Tuesday");
}

TEST_CASE("parseStreamLine — SSE noise yields nothing", "[llm][stream]") {
    for (auto provider : {LLMProvider::OPENAI, LLMProvider::ANTHROPIC, LLMProvider::GEMINI}) {
        REQUIRE_FALSE(parseStreamLine(provider, "").has_value());          // separator
        REQUIRE_FALSE(parseStreamLine(provider, ": keep-alive").has_value());  // comment
        REQUIRE_FALSE(parseStreamLine(provider, "id: 42").has_value());    // unused field
        REQUIRE_FALSE(parseStreamLine(provider, "data:").has_value());     // empty payload
    }
}

TEST_CASE("parseStreamLine — data: with no space after the colon", "[llm][stream]") {
    // The space is optional in the SSE spec even though every provider sends it.
    auto d = parseStreamLine(LLMProvider::OPENAI,
        R"(data:{"choices":[{"delta":{"content":"x"}}]})");
    REQUIRE(d.has_value());
    REQUIRE(*d == "x");
}

TEST_CASE("parseStreamLine — malformed JSON does not throw", "[llm][stream]") {
    // A truncated frame must cost one fragment, never the whole answer.
    REQUIRE_FALSE(parseStreamLine(LLMProvider::OPENAI, R"(data: {"choices":[{"del)").has_value());
    REQUIRE_FALSE(parseStreamLine(LLMProvider::OLLAMA, "{not json at all").has_value());
}

TEST_CASE("parseStreamLine — empty content string is not a delta", "[llm][stream]") {
    // Emitting these would fire the consumer's callback for nothing.
    REQUIRE_FALSE(parseStreamLine(LLMProvider::OPENAI,
        R"(data: {"choices":[{"delta":{"content":""}}]})").has_value());
    REQUIRE_FALSE(parseStreamLine(LLMProvider::OLLAMA,
        R"({"message":{"content":""},"done":false})").has_value());
}

TEST_CASE("parseStreamLine — whitespace-only content IS a delta", "[llm][stream]") {
    // The space between two words arrives as its own frame. Dropping it would
    // run the answer together.
    auto d = parseStreamLine(LLMProvider::OPENAI,
        R"(data: {"choices":[{"delta":{"content":" "}}]})");
    REQUIRE(d.has_value());
    REQUIRE(*d == " ");
}
