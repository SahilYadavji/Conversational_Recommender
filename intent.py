"""
intent.py

Intent detection for the SHL Assessment Recommender.

The chatbot decides what to do before calling retrieval or the LLM.

Supported intents:

- clarify
- recommend
- compare
- refine
- refuse
"""

from __future__ import annotations

import re
from typing import Any


COMPARE_KEYWORDS = [
    "compare",
    "difference",
    "vs",
    "versus",
    "better",
    "which one",
]

REFINE_KEYWORDS = [
    "also",
    "instead",
    "actually",
    "change",
    "update",
    "add",
    "remove",
    "include",
    "exclude",
    "only",
]

OFF_TOPIC_KEYWORDS = [
    "weather",
    "news",
    "cricket",
    "football",
    "movie",
    "joke",
    "recipe",
    "bitcoin",
    "stock",
    "politics",
]

PROMPT_INJECTION = [
    "ignore previous instructions",
    "system prompt",
    "developer prompt",
    "reveal prompt",
    "bypass",
]


ROLE_KEYWORDS = [
    "developer",
    "engineer",
    "manager",
    "analyst",
    "consultant",
    "designer",
    "tester",
    "architect",
    "java",
    "python",
    "react",
    "angular",
    "sales",
    "marketing",
    "support",
    "customer",
    "qa",
    "automation",
    "data",
    "aws",
    "azure",
]


def _latest_user_message(messages: list[dict[str, Any]]) -> str:
    """
    Returns latest user message.
    """

    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "").lower()

    return ""


def detect_intent(messages: list[dict[str, Any]]) -> str:
    """
    Detect conversation intent.

    Returns one of:

    clarify
    recommend
    compare
    refine
    refuse
    """

    text = _latest_user_message(messages)

    if not text:
        return "clarify"

    # -----------------------------------------
    # Prompt Injection
    # -----------------------------------------

    for word in PROMPT_INJECTION:
        if word in text:
            return "refuse"

    # -----------------------------------------
    # Off-topic
    # -----------------------------------------

    for word in OFF_TOPIC_KEYWORDS:
        if word in text:
            return "refuse"

    # -----------------------------------------
    # Compare
    # -----------------------------------------

    for word in COMPARE_KEYWORDS:
        if word in text:
            return "compare"

    # -----------------------------------------
    # Refinement
    # -----------------------------------------

    for word in REFINE_KEYWORDS:
        if word in text:
            return "refine"

    # -----------------------------------------
    # Recommendation
    # -----------------------------------------

    for role in ROLE_KEYWORDS:
        if role in text:
            return "recommend"

    # Experience detection

    if re.search(r"\d+\s*(year|years)", text):
        return "recommend"

    # Hiring related

    if any(
        word in text
        for word in [
            "hire",
            "hiring",
            "recruit",
            "candidate",
            "assessment",
            "test",
        ]
    ):
        return "recommend"

    # -----------------------------------------
    # Otherwise clarify
    # -----------------------------------------

    return "clarify"


if __name__ == "__main__":

    tests = [

        [{"role": "user", "content": "Need assessment"}],

        [{"role": "user", "content": "Hiring Java Developer"}],

        [{"role": "user", "content": "Compare OPQ and GSA"}],

        [{"role": "user", "content": "Actually include personality tests"}],

        [{"role": "user", "content": "Tell me a joke"}],

        [{"role": "user", "content": "Ignore previous instructions"}],
    ]

    for t in tests:
        print(
            t[0]["content"],
            "->",
            detect_intent(t),
        )