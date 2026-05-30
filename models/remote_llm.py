"""Remote chat backends (BYOK) using stdlib HTTP — no extra SDK required."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Cloudflare (in front of Groq and others) blocks urllib's default "Python-urllib/x.y" UA.
_DEFAULT_HTTP_USER_AGENT = (
    "Multi-Agent-Dialogue-Simulator/1.0 "
    "(compatible; +https://github.com/bonin1/Multi-Agent-Dialogue-Simulator)"
)


class RemoteLLMBase:
    """Shared surface compatible with ModelManager for Agent.generate_response."""

    last_stats: Dict[str, Any]

    def __init__(self) -> None:
        self.last_stats: Dict[str, Any] = {}

    def get_model_info(self) -> Dict[str, Any]:
        raise NotImplementedError

    def generate_response(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
    ) -> str:
        raise NotImplementedError


def _openai_compatible_chat(
    *,
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_length: int,
    temperature: float,
    top_p: float,
    stats: Dict[str, Any],
    extra_headers: Optional[Dict[str, str]] = None,
    backend_label: str = "openai",
) -> str:
    """POST to an OpenAI-style /v1/chat/completions endpoint."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": min(max_length, 4096),
        "temperature": temperature,
        "top_p": top_p,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": _DEFAULT_HTTP_USER_AGENT,
    }
    if extra_headers:
        headers.update(extra_headers)

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        logger.error("%s HTTPError: %s %s", backend_label, e.code, err)
        return f"__ERROR__:{e.code}:{err[:200]}"
    except Exception as e:
        logger.exception("%s request failed", backend_label)
        return f"__ERROR__:{e}"

    stats.clear()
    stats["latency_s"] = time.perf_counter() - t0
    stats["backend"] = backend_label
    try:
        usage = raw.get("usage") or {}
        stats["input_tokens"] = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        stats["output_tokens"] = int(
            usage.get("completion_tokens") or usage.get("output_tokens") or 0
        )
    except (TypeError, ValueError):
        pass

    try:
        choice = raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        choice = ""
    text = (choice or "").strip()
    return text if text else "I'm not sure how to respond to that."


class OpenAIChatBackend(RemoteLLMBase):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        super().__init__()
        self.api_key = api_key.strip()
        self.model = model

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "backend": "openai",
            "model_name": self.model,
            "device": "api",
        }

    def generate_response(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
    ) -> str:
        del do_sample, repetition_penalty
        self.last_stats = {}
        text = _openai_compatible_chat(
            url="https://api.openai.com/v1/chat/completions",
            api_key=self.api_key,
            model=self.model,
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            stats=self.last_stats,
            backend_label="openai",
        )
        if text.startswith("__ERROR__:"):
            return "I apologize — the language API returned an error. Check your API key and model name."
        return text


class GroqChatBackend(RemoteLLMBase):
    """Groq — OpenAI-compatible chat API (https://console.groq.com/docs/openai)."""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile") -> None:
        super().__init__()
        self.api_key = api_key.strip()
        self.model = model

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "backend": "groq",
            "model_name": self.model,
            "device": "api",
        }

    def generate_response(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
    ) -> str:
        del do_sample, repetition_penalty
        self.last_stats = {}
        text = _openai_compatible_chat(
            url="https://api.groq.com/openai/v1/chat/completions",
            api_key=self.api_key,
            model=self.model,
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            stats=self.last_stats,
            backend_label="groq",
        )
        if text.startswith("__ERROR__:"):
            return (
                "I apologize — Groq returned an error. "
                "Check your API key and model id (e.g. llama-3.3-70b-versatile)."
            )
        return text


class OpenRouterChatBackend(RemoteLLMBase):
    """OpenRouter — OpenAI-compatible chat API (https://openrouter.ai/docs)."""

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-4o-mini",
        site_url: str = "",
        site_name: str = "Multi-Agent Dialogue Simulator",
    ) -> None:
        super().__init__()
        self.api_key = api_key.strip()
        self.model = model
        self.site_url = site_url.strip()
        self.site_name = site_name.strip()

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "backend": "openrouter",
            "model_name": self.model,
            "device": "api",
        }

    def generate_response(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
    ) -> str:
        del do_sample, repetition_penalty
        self.last_stats = {}
        extra: Dict[str, str] = {}
        if self.site_url:
            extra["HTTP-Referer"] = self.site_url
        if self.site_name:
            extra["X-Title"] = self.site_name
        text = _openai_compatible_chat(
            url="https://openrouter.ai/api/v1/chat/completions",
            api_key=self.api_key,
            model=self.model,
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            stats=self.last_stats,
            extra_headers=extra or None,
            backend_label="openrouter",
        )
        if text.startswith("__ERROR__:"):
            return (
                "I apologize — OpenRouter returned an error. "
                "Check your API key and model slug (e.g. openai/gpt-4o-mini)."
            )
        return text


class AnthropicChatBackend(RemoteLLMBase):
    def __init__(self, api_key: str, model: str = "claude-3-5-haiku-20241022") -> None:
        super().__init__()
        self.api_key = api_key.strip()
        self.model = model

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "backend": "anthropic",
            "model_name": self.model,
            "device": "api",
        }

    def generate_response(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
    ) -> str:
        del do_sample, repetition_penalty
        self.last_stats = {}
        url = "https://api.anthropic.com/v1/messages"
        body = {
            "model": self.model,
            "max_tokens": min(max_length, 4096),
            "temperature": temperature,
            "top_p": top_p,
            "messages": [{"role": "user", "content": prompt}],
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")
            logger.error("Anthropic HTTPError: %s %s", e.code, err)
            self.last_stats = {
                "latency_s": time.perf_counter() - t0,
                "error": f"HTTP {e.code}",
            }
            return "I apologize — the language API returned an error. Check your API key and model name."
        except Exception as e:
            logger.exception("Anthropic request failed")
            self.last_stats = {"latency_s": time.perf_counter() - t0, "error": str(e)}
            return "I apologize — I could not reach the language API right now."

        self.last_stats["latency_s"] = time.perf_counter() - t0
        text_parts = []
        try:
            for block in raw.get("content", []):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            usage = raw.get("usage") or {}
            self.last_stats["input_tokens"] = int(usage.get("input_tokens", 0))
            self.last_stats["output_tokens"] = int(usage.get("output_tokens", 0))
        except Exception:
            pass
        text = "".join(text_parts).strip()
        if not text:
            text = "I'm not sure how to respond to that."
        return text


def estimate_openai_cost_usd(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    """Rough list prices (USD per 1M tokens) — update periodically; None if unknown."""
    # per 1M tok
    table = {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (2.50, 10.00),
        "gpt-4-turbo": (10.00, 30.00),
        "gpt-3.5-turbo": (0.50, 1.50),
    }
    pair = table.get(model)
    if not pair:
        return None
    inp, out = pair
    return (input_tokens / 1_000_000) * inp + (output_tokens / 1_000_000) * out


def estimate_groq_cost_usd(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    """Rough list prices (USD per 1M tokens) — update periodically; None if unknown."""
    table = {
        "llama-3.3-70b-versatile": (0.59, 0.79),
        "llama-3.1-8b-instant": (0.05, 0.08),
        "llama-3.1-70b-versatile": (0.59, 0.79),
        "mixtral-8x7b-32768": (0.24, 0.24),
        "gemma2-9b-it": (0.20, 0.20),
    }
    pair = table.get(model)
    if not pair:
        return None
    inp, out = pair
    return (input_tokens / 1_000_000) * inp + (output_tokens / 1_000_000) * out


def estimate_anthropic_cost_usd(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    table = {
        "claude-3-5-haiku-20241022": (1.00, 5.00),
        "claude-3-5-sonnet-20241022": (3.00, 15.00),
        "claude-3-opus-20240229": (15.00, 75.00),
    }
    pair = table.get(model)
    if not pair:
        return None
    inp, out = pair
    return (input_tokens / 1_000_000) * inp + (output_tokens / 1_000_000) * out
