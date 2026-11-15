"""Thin wrapper around the Claude API, used by the generative AI features.

Every feature also has an offline fallback, so the app works without an API key.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

MODEL = "claude-opus-5-5"
_client = None


def credentials_available() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return (Path.home() / ".config" / "anthropic").exists()  # `ant auth login` profile


def set_api_key(key: str) -> None:
    global _client
    os.environ["ANTHROPIC_API_KEY"] = key.strip()
    _client = None


def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic()
    return _client


class LLMError(RuntimeError):
    pass


def ask(system: str, prompt: str, schema: dict | None = None, effort: str = "medium",
        max_tokens: int = 16000):
    """Send one request to Claude. Returns text, or a parsed dict when `schema` is given."""
    import anthropic

    kwargs = dict(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        thinking={"type": "adaptive"},
        output_config={"effort": effort},
        # If a safety classifier declines, the API retries on a fallback model in the same call
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if schema is not None:
        kwargs["output_config"]["format"] = {"type": "json_schema", "schema": schema}
    try:
        response = _get_client().beta.messages.create(**kwargs)
    except anthropic.AuthenticationError as e:
        raise LLMError("The API key was rejected. Check it in the sidebar.") from e
    except anthropic.RateLimitError as e:
        raise LLMError("Rate limited by the API. Try again in a minute.") from e
    except anthropic.APIStatusError as e:
        raise LLMError(f"API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise LLMError("Could not reach the Claude API. Check your internet connection.") from e

    if response.stop_reason == "refusal":
        raise LLMError("The model declined this request.")
    text = "".join(b.text for b in response.content if b.type == "text")
    if schema is not None:
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError("The model returned output that could not be parsed.") from e
    return text
