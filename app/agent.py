"""
Agent loop: model decides tool_call vs final_answer, each turn is JSON-schema
constrained so we always get parseable structured output back.
"""
import json

from app import config, tools
from app.llm_client import chat_completion, parse_agent_turn
from app.schemas import AGENT_TURN_JSON_SCHEMA, ChatResponse, Source


async def run_agent(user_message: str) -> ChatResponse:
    messages = [
        {
            "role": "system",
            "content": config.SYSTEM_PROMPT
            + "\n\n"
            + tools.TOOL_DESCRIPTIONS_FOR_PROMPT
            + "\n\nRespond ONLY with a JSON object matching this shape:\n"
            '{"action": "tool_call", "tool_name": "...", "tool_args": {...}} '
            'OR {"action": "final_answer", "final_answer": "...", "confidence": "low|medium|high"}',
        },
        {"role": "user", "content": user_message},
    ]

    used_tools: list[str] = []
    collected_sources: list[Source] = []

    for _ in range(config.MAX_TOOL_ITERATIONS):
        message = await chat_completion(
            messages, json_schema=AGENT_TURN_JSON_SCHEMA, max_tokens=config.AGENT_MAX_TOKENS
        )
        content = message.get("content", "")
        turn = parse_agent_turn(content)

        if turn.get("action") == "tool_call":
            tool_name = turn.get("tool_name", "")
            tool_args = turn.get("tool_args", {}) or {}
            used_tools.append(tool_name)

            result = tools.execute_tool(tool_name, tool_args)

            if tool_name == "retrieve_documents" and "results" in result:
                for r in result["results"]:
                    collected_sources.append(Source(**r))

            # feed the tool result back to the model and continue the loop
            messages.append({"role": "assistant", "content": json.dumps(turn)})
            messages.append({"role": "user", "content": f"Tool result: {json.dumps(result)}"})
            continue

        # action == final_answer (or fallback)
        return ChatResponse(
            answer=turn.get("final_answer", "I couldn't produce an answer."),
            used_tools=used_tools,
            sources=collected_sources,
            confidence=turn.get("confidence", "medium"),
        )

    # Exhausted iterations without a final answer
    return ChatResponse(
        answer="I wasn't able to finish reasoning about this within the tool-call budget. "
        "Try rephrasing or narrowing your question.",
        used_tools=used_tools,
        sources=collected_sources,
        confidence="low",
    )