"""LLM provider selection for the question generator and the judge.

Order: ANTHROPIC_API_KEY (Anthropic SDK, direct) then OPENROUTER_API_KEY (OpenRouter's
OpenAI-compatible chat completions, Claude models). Both return a Caller:
(system, user) -> text. Every call's token usage and cost (when the provider reports it)
accumulate in USAGE so runs can report what they spent.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Callable

Caller = Callable[[str, str], str]

DEFAULT_MODEL = "claude-opus-5-5"
OPENROUTER_MODELS = {"claude-opus-5-5": "anthropic/claude-opus-5.5", "claude-sonnet-5-5": "anthropic/claude-sonnet-5.5",
                     "claude-haiku-5-5": "anthropic/claude-haiku-5.5"}
USAGE = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "errors": 0}


def provider() -> str | None:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "anthropic"
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    return None


def _post_json(url: str, payload: dict, headers: dict, *, timeout: int = 300, retries: int = 4) -> dict:
    data = json.dumps(payload).encode()
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(2 ** (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt < retries:
                time.sleep(2 ** (attempt + 1))
                continue
            raise
    raise RuntimeError("unreachable")


def openrouter_caller(model: str = DEFAULT_MODEL, effort: str = "medium", max_tokens: int = 4000) -> Caller:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    or_model = OPENROUTER_MODELS.get(model, model)

    def call(system: str, user: str) -> str:
        out = _post_json(
            "https://openrouter.ai/api/v1/chat/completions",
            {"model": or_model, "max_tokens": max_tokens, "reasoning": {"effort": effort}, "usage": {"include": True},
             "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
            {"Authorization": f"Bearer {key}", "X-Title": "ragtuner"},
        )
        USAGE["calls"] += 1
        u = out.get("usage") or {}
        USAGE["input_tokens"] += int(u.get("prompt_tokens", 0))
        USAGE["output_tokens"] += int(u.get("completion_tokens", 0))
        USAGE["cost_usd"] += float(u.get("cost", 0.0) or 0.0)
        if "error" in out:
            USAGE["errors"] += 1
            raise RuntimeError(f"openrouter error: {out['error']}")
        choice = out["choices"][0]
        if choice.get("finish_reason") in ("content_filter",):
            raise RuntimeError("model refused")
        return choice["message"].get("content") or ""

    return call


def anthropic_caller(model: str = DEFAULT_MODEL, effort: str = "medium", max_tokens: int = 4000) -> Caller:
    import anthropic

    client = anthropic.Anthropic()

    def call(system: str, user: str) -> str:
        resp = client.beta.messages.create(
            model=model, max_tokens=max_tokens, system=system, output_config={"effort": effort},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            messages=[{"role": "user", "content": user}],
        )
        USAGE["calls"] += 1
        USAGE["input_tokens"] += resp.usage.input_tokens
        USAGE["output_tokens"] += resp.usage.output_tokens
        if resp.stop_reason == "refusal":
            raise RuntimeError("model refused")
        return "".join(b.text for b in resp.content if b.type == "text")

    return call


def default_caller(model: str = DEFAULT_MODEL, effort: str = "medium") -> Caller:
    p = provider()
    if p == "anthropic":
        return anthropic_caller(model, effort)
    if p == "openrouter":
        return openrouter_caller(model, effort)
    raise RuntimeError("no LLM credentials: set ANTHROPIC_API_KEY or OPENROUTER_API_KEY")
