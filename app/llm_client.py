"""
Client for OpenAI-compatible chat completion endpoints (vLLM / vllm_metal /
any compatible server), with:
- true async calls (AsyncOpenAI) so concurrent requests don't block each other
- retry with exponential backoff per provider (tenacity)
- automatic fallback to a secondary provider if the primary is exhausted
- structured output via guided decoding (`guided_json` extra_body extension)
"""
import json
import logging

from openai import AsyncOpenAI, APIError, APIConnectionError, APITimeoutError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)

from app import config

logger = logging.getLogger(__name__)

_clients: dict[str, AsyncOpenAI] = {}


def _get_client(base_url: str) -> AsyncOpenAI:
    if base_url not in _clients:
        _clients[base_url] = AsyncOpenAI(base_url=base_url, api_key="not-needed")
    return _clients[base_url]


class LLMError(RuntimeError):
    """Raised only after all configured providers have been exhausted."""
    pass


_RETRYABLE = (APIConnectionError, APITimeoutError, APIError)


@retry(
    stop=stop_after_attempt(config.RETRY_MAX_ATTEMPTS),
    wait=wait_exponential(min=config.RETRY_WAIT_MIN_SECONDS, max=config.RETRY_WAIT_MAX_SECONDS),
    retry=retry_if_exception_type(_RETRYABLE),
    reraise=True,
)
async def _call_provider(
    provider: dict,
    messages: list[dict],
    json_schema: dict | None,
    temperature: float,
    top_p: float,
    max_tokens: int,
):
    client = _get_client(provider["base_url"])
    extra_body = {"guided_json": json_schema} if json_schema is not None else {}
    return await client.chat.completions.create(
        model=provider["model"],
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        frequency_penalty=config.LLM_FREQUENCY_PENALTY,
        presence_penalty=config.LLM_PRESENCE_PENALTY,
        extra_body=extra_body,
    )


async def chat_completion(
    messages: list[dict],
    json_schema: dict | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
) -> dict:
    """
    Tries each configured provider in order (primary, then fallback if set),
    retrying each with exponential backoff before moving to the next.
    Raises LLMError only if every provider fails.
    """
    last_error: Exception | None = None

    for provider in config.LLM_PROVIDERS:
        try:
            response = await _call_provider(
                provider,
                messages,
                json_schema,
                temperature if temperature is not None else config.LLM_TEMPERATURE,
                top_p if top_p is not None else config.LLM_TOP_P,
                max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS,
            )
            choice = response.choices[0].message
            return {"role": choice.role, "content": choice.content}
        except RetryError as e:
            last_error = e.last_attempt.exception()
            logger.warning("Provider %s exhausted retries: %s", provider["base_url"], last_error)
        except _RETRYABLE as e:
            last_error = e
            logger.warning("Provider %s failed: %s", provider["base_url"], e)

    raise LLMError(f"All LLM providers failed. Last error: {last_error}")


def parse_agent_turn(raw_content: str) -> dict:
    """
    Extracts and parses only the FIRST valid JSON object in the response,
    ignoring anything after it. This matters because some OpenAI-compatible
    servers don't strictly honor guided_json, so smaller models can keep
    regenerating past a complete object instead of stopping cleanly.
    """
    raw_content = (raw_content or "").strip()
    start = raw_content.find("{")
    if start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(raw_content, start)
            return obj
        except json.JSONDecodeError:
            pass
    return {"action": "final_answer", "final_answer": raw_content[:500], "confidence": "low"}