"""Connect local or remote LLM backends (shared by sidebar and Settings)."""

from __future__ import annotations

import os
from typing import Any, Dict

import streamlit as st

from models.model_manager import ModelManager
from models.remote_llm import (
    AnthropicChatBackend,
    GroqChatBackend,
    OpenAIChatBackend,
    OpenRouterChatBackend,
)


def _has_key(state_key: str, env_name: str) -> bool:
    return bool(
        (st.session_state.get(state_key) or "").strip()
        or (os.environ.get(env_name) or "").strip()
    )


def hydrate_api_keys_from_secrets_and_env() -> None:
    """Fill session_state API keys from Streamlit secrets / env when UI fields are empty."""
    mapping = {
        "groq_api_key": "GROQ_API_KEY",
        "openai_api_key": "OPENAI_API_KEY",
        "anthropic_api_key": "ANTHROPIC_API_KEY",
        "openrouter_api_key": "OPENROUTER_API_KEY",
        "huggingface_token": "HUGGING_FACE_HUB_TOKEN",
    }
    try:
        secrets = getattr(st, "secrets", None)
        if secrets:
            for state_key, env_name in mapping.items():
                if env_name in secrets and not (st.session_state.get(state_key) or "").strip():
                    st.session_state[state_key] = str(secrets[env_name])
            if "llm_backend" in secrets and not st.session_state.get("llm_backend_user_locked"):
                st.session_state.llm_backend = str(secrets["llm_backend"])
    except Exception:
        pass

    for state_key, env_name in mapping.items():
        env_val = (os.environ.get(env_name) or "").strip()
        if env_val and not (st.session_state.get(state_key) or "").strip():
            st.session_state[state_key] = env_val


def resolve_llm_backend(*, show_hint: bool = True) -> str:
    """
    Pick backend to use. If still 'local' but a remote API key exists, auto-use that API
    so we never start a 10GB Hugging Face download by accident.
    """
    hydrate_api_keys_from_secrets_and_env()
    backend = st.session_state.get("llm_backend", "local")

    if st.session_state.get("llm_backend_user_locked"):
        return backend

    if backend != "local":
        return backend

    remote_priority = [
        ("groq", "groq_api_key", "GROQ_API_KEY"),
        ("openai", "openai_api_key", "OPENAI_API_KEY"),
        ("anthropic", "anthropic_api_key", "ANTHROPIC_API_KEY"),
        ("openrouter", "openrouter_api_key", "OPENROUTER_API_KEY"),
    ]
    for name, state_key, env_name in remote_priority:
        if _has_key(state_key, env_name):
            st.session_state.llm_backend = name
            if show_hint:
                st.info(
                    f"**{name}** API key detected — using cloud API (no local model download). "
                    "To force the huge local model instead, set **LLM backend** to **local** in Settings."
                )
            return name

    return "local"


def load_llm_backend() -> bool:
    """Load local HF model or configure a remote API backend (BYOK)."""
    backend = resolve_llm_backend(show_hint=True)

    try:
        hf_ui = (st.session_state.get("huggingface_token") or "").strip()
        if hf_ui:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_ui

        if backend == "openai":
            key = (st.session_state.get("openai_api_key") or os.environ.get("OPENAI_API_KEY") or "").strip()
            if not key:
                st.error("OpenAI API key required (Settings tab or OPENAI_API_KEY env).")
                return False
            with st.spinner("Connecting to OpenAI…"):
                st.session_state.model_manager = OpenAIChatBackend(
                    key, st.session_state.get("openai_model", "gpt-4o-mini")
                )
            return True

        if backend == "anthropic":
            key = (st.session_state.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY") or "").strip()
            if not key:
                st.error("Anthropic API key required (Settings tab or ANTHROPIC_API_KEY env).")
                return False
            with st.spinner("Connecting to Anthropic…"):
                st.session_state.model_manager = AnthropicChatBackend(
                    key, st.session_state.get("anthropic_model", "claude-3-5-haiku-20241022")
                )
            return True

        if backend == "openrouter":
            key = (
                st.session_state.get("openrouter_api_key")
                or os.environ.get("OPENROUTER_API_KEY")
                or ""
            ).strip()
            if not key:
                st.error("OpenRouter API key required (Settings tab or OPENROUTER_API_KEY env).")
                return False
            with st.spinner("Connecting to OpenRouter…"):
                st.session_state.model_manager = OpenRouterChatBackend(
                    key,
                    st.session_state.get("openrouter_model", "openai/gpt-4o-mini"),
                    site_url=st.session_state.get("openrouter_site_url", ""),
                    site_name=st.session_state.get("openrouter_site_name", "Multi-Agent Dialogue Simulator"),
                )
            return True

        if backend == "groq":
            key = (st.session_state.get("groq_api_key") or os.environ.get("GROQ_API_KEY") or "").strip()
            if not key:
                st.error(
                    "Groq API key required. Settings → LLM backend = groq → paste key → "
                    "click **Connect LLM** (or sidebar **Load AI Model**)."
                )
                return False
            with st.spinner("Connecting to Groq…"):
                st.session_state.model_manager = GroqChatBackend(
                    key, st.session_state.get("groq_model", "llama-3.3-70b-versatile")
                )
            return True

        st.warning(
            "Loading **local** Hugging Face model (~10GB download first time). "
            "For Groq/OpenAI instead: Settings → pick **groq** → paste key → **Connect LLM**."
        )
        with st.spinner("Loading local AI model… This may take a long time on first run."):
            model_id = st.session_state.get("local_model_name", "teknium/OpenHermes-2.5-Mistral-7B")
            st.session_state.model_manager = ModelManager(model_id)
        return True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return False


def backend_status_line() -> Dict[str, Any]:
    """Short status for UI banners."""
    hydrate_api_keys_from_secrets_and_env()
    backend = st.session_state.get("llm_backend", "local")
    connected = st.session_state.get("model_manager") is not None
    return {"backend": backend, "connected": connected}
