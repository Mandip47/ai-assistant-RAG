"""
Client for vllm_metal's OpenAI-compatible /v1/chat/completions endpoint,
using the official `openai` SDK (works against any OpenAI-compatible server
by pointing base_url at it).

Handles:
- prompt engineering (system prompt + temperature/top_p tuning)
- structured output via guided decoding, passed as a vendor extension
  (`guided_json`) through `extra_body`. vllm_metal may or may not honor
  this strictly — parse_agent_turn() below degrades gracefully either way.
"""
import json

from openai import OpenAI, APIError, APIConnectionError

from app import config

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=config.LLM_BASE_URL, api_key="not-needed")
    return _client


class LLMError(RuntimeError):
    pass


async def chat_completion(
    messages: list[dict],
    json_schema: dict | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
) -> dict:
    """
    Calls the OpenAI-compatible chat completions endpoint.
    If json_schema is provided, we ask for guided/constrained JSON via the
    `guided_json` extra_body field (a vLLM-family extension).
    Returns the message dict: {"role": ..., "content": "..."}
    """
    client = _get_client()
    extra_body = {"guided_json": json_schema} if json_schema is not None else {}

    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL_NAME,
            messages=messages,
            temperature=temperature if temperature is not None else config.LLM_TEMPERATURE,
            top_p=top_p if top_p is not None else config.LLM_TOP_P,
            max_tokens=max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS,
            frequency_penalty=config.LLM_FREQUENCY_PENALTY,
            presence_penalty=config.LLM_PRESENCE_PENALTY,
            extra_body=extra_body,
        )
    except APIConnectionError as e:
        raise LLMError(f"Failed to reach LLM server at {config.LLM_BASE_URL}: {e}") from e
    except APIError as e:
        raise LLMError(f"LLM server returned an error: {e}") from e

    choice = response.choices[0].message
    return {"role": choice.role, "content": choice.content}


def parse_agent_turn(raw_content: str) -> dict:
    """
    Extracts and parses only the FIRST valid JSON object in the response,
    ignoring anything after it. This matters because vllm_metal doesn't
    strictly honor guided_json, so small models can sometimes keep
    regenerating past a complete object instead of stopping cleanly.
    """
    raw_content = raw_content.strip()
    start = raw_content.find("{")
    if start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(raw_content, start)
            return obj
        except json.JSONDecodeError:
            pass
    # No valid JSON object found at all: treat (truncated) raw text as the answer.
    return {"action": "final_answer", "final_answer": raw_content[:500], "confidence": "low"}