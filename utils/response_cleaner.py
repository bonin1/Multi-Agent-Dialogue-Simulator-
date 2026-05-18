"""Strip model / template junk from agent dialogue lines."""

from __future__ import annotations

import re


def clean_dialogue_text(text: str) -> str:
    """Return speech-only text suitable for the conversation UI."""
    if not text:
        return text

    s = text.strip()

    # ChatML / instruction leaks
    s = re.sub(r"<\|[^|]+\|>", "", s)
    s = re.sub(r"</?(?:s|system|user|assistant|instruction)[^>]*>", "", s, flags=re.I)

    # diff / style template artifacts (common with some instruction-tuned models)
    s = re.sub(r"diff\s*\{[^}]*\}", "", s, flags=re.I)
    s = re.sub(r"\{style\s*=\s*[\"'][^\"']*[\"']\s*\}", "", s, flags=re.I)
    s = re.sub(r"-?\{\}\{-?>\}", "", s)
    s = re.sub(r"\[-?\]->\]", "", s)
    s = re.sub(r"\{\s*\}", "", s)

    # Lone bracket fragments left by broken templates
    s = re.sub(r"\[\s*(?=[A-Za-z])", "", s)
    s = re.sub(r"(?<=[.!?])\s*\]", "", s)

    # Trailing metadata accidentally echoed by the model
    s = re.sub(
        r",?\s*\d+\.\d+s,?\s*tok\s+\d+\+\d+,?\s*[\w./-]+$",
        "",
        s,
        flags=re.I,
    )

    s = re.sub(r"\s+", " ", s).strip(" ,-[]")

    # Known weak model fallbacks → drop so caller can substitute
    low = s.lower()
    if low in (
        "i need more time to process this information.",
        "i apologize, but i'm having trouble generating a response right now.",
        "i'm not sure how to respond to that.",
    ):
        return ""

    # Hard cap: spoken dialogue only (max ~2 sentences / ~280 chars)
    if len(s) > 280:
        parts = re.split(r"(?<=[.!?])\s+", s)
        s = " ".join(parts[:2]).strip()
        if len(s) > 280:
            s = s[:277].rsplit(" ", 1)[0] + "…"

    return s
