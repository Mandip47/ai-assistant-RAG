"""
Agent loop: retrieves relevant context eagerly (rather than depending on the
model reliably deciding to call the retrieve_documents tool — small/quantized
models are inconsistent about that), remembers conversation history per
session, and still supports explicit tool calls (e.g. calculator) via the
JSON-schema-constrained turn format.
"""
import asyncio
import json
import logging

from app import config, tools, sessions
from app.llm_client import chat_completion, parse_agent_turn, LLMError
from app.schemas import AGENT_TURN_JSON_SCHEMA, ChatResponse, Source

logger = logging.getLogger(__name__)


async def run_agent(user_message: str, session_id: str = "default") -> ChatResponse:
    history = sessions.get_history(session_id)
    collected_sources: list[Source] = []

    # Eager retrieval: always pull relevant context up front for every
    # message, instead of relying on the model to correctly emit a
    # retrieve_documents tool_call. This is what makes ingested documents
    # actually get used reliably, even with a small local model.
    #
    # Guarded: a retrieval failure (e.g. the vector store is temporarily
    # unreachable) must not take down the whole chat turn — that would skip
    # the LLM call entirely and, critically, skip sessions.append_turn below,
    # which makes it *look* like the conversation "forgot" when really the
    # turn never completed at all. Degrade to answering without context
    # instead, and let the model say it doesn't have the info.
    try:
        retrieval = await asyncio.to_thread(tools.execute_tool, "retrieve_documents", {"query": user_message})
    except Exception:
        logger.exception("retrieve_documents failed; answering without retrieved context")
        retrieval = {"results": [], "error": "retrieval unavailable"}
    context_block = ""
    if retrieval.get("results"):
        for r in retrieval["results"]:
            collected_sources.append(Source(**r))
        context_lines = "\n".join(f"- {r['text']}" for r in retrieval["results"])
        context_block = f"\n\nRelevant context retrieved from the knowledge base for this question:\n{context_lines}"

    system_content = (
        config.SYSTEM_PROMPT
        + context_block
        + "\n\n"
        + tools.TOOL_DESCRIPTIONS_FOR_PROMPT
        + "\n\nNote: relevant knowledge-base context (if any) has already been retrieved and is "
        "included above — you usually do NOT need to call retrieve_documents yourself. "
        "Use the calculator tool if the question needs arithmetic.\n\n"
        "Respond ONLY with a JSON object matching this shape:\n"
        '{"action": "tool_call", "tool_name": "...", "tool_args": {...}} '
        'OR {"action": "final_answer", "final_answer": "...", "confidence": "low|medium|high"}'
    )

    messages = [{"role": "system", "content": system_content}] + history + [
        {"role": "user", "content": user_message}
    ]

    used_tools: list[str] = []

    for _ in range(config.MAX_TOOL_ITERATIONS):
        try:
            message = await chat_completion(
                messages, json_schema=AGENT_TURN_JSON_SCHEMA, max_tokens=config.AGENT_MAX_TOKENS
            )
        except LLMError:
            answer = (
                "The assistant is temporarily unavailable (all model providers failed "
                "after retrying). Please try again shortly."
            )
            return ChatResponse(answer=answer, used_tools=used_tools, sources=collected_sources, confidence="low")

        content = message.get("content", "")
        turn = parse_agent_turn(content)

        if turn.get("action") == "tool_call":
            tool_name = turn.get("tool_name", "")
            tool_args = turn.get("tool_args", {}) or {}
            used_tools.append(tool_name)

            result = await asyncio.to_thread(tools.execute_tool, tool_name, tool_args)

            if tool_name == "retrieve_documents" and "results" in result:
                for r in result["results"]:
                    collected_sources.append(Source(**r))

            messages.append({"role": "assistant", "content": json.dumps(turn)})
            messages.append({"role": "user", "content": f"Tool result: {json.dumps(result)}"})
            continue

        answer = turn.get("final_answer", "I couldn't produce an answer.")
        sessions.append_turn(session_id, user_message, answer)
        return ChatResponse(
            answer=answer,
            used_tools=used_tools,
            sources=collected_sources,
            confidence=turn.get("confidence", "medium"),
        )

    answer = (
        "I wasn't able to finish reasoning about this within the tool-call budget. "
        "Try rephrasing or narrowing your question."
    )
    sessions.append_turn(session_id, user_message, answer)
    return ChatResponse(answer=answer, used_tools=used_tools, sources=collected_sources, confidence="low")