"""
conversation.py

Conversation state extraction for the SHL Assessment Recommender.

This module converts the complete chat history into a structured
representation of the user's hiring requirements.

The chatbot uses this state instead of only the latest user message.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------
# Conversation State
# ---------------------------------------------------------------------

@dataclass
class ConversationState:
    role: str = ""
    experience: str = ""
    job_level: str = ""
    skills: List[str] = field(default_factory=list)
    assessment_types: List[str] = field(default_factory=list)
    remote: str = ""
    adaptive: str = ""
    compare: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------
# Keyword Dictionaries
# ---------------------------------------------------------------------

ROLE_KEYWORDS = [
    "developer",
    "engineer",
    "manager",
    "tester",
    "architect",
    "analyst",
    "consultant",
    "designer",
    "administrator",
]

TECH_SKILLS = [
    "java",
    "python",
    "react",
    "angular",
    "spring",
    "spring boot",
    "sql",
    "aws",
    "azure",
    "docker",
    "kubernetes",
    "selenium",
    "javascript",
    "typescript",
    "html",
    "css",
    "c",
    "c++",
    "c#",
]

ASSESSMENT_TYPES = {
    "personality": "Personality & Behavior",
    "behaviour": "Personality & Behavior",
    "behavior": "Personality & Behavior",
    "coding": "Knowledge & Skills",
    "technical": "Knowledge & Skills",
    "simulation": "Simulations",
    "ability": "Ability & Aptitude",
    "aptitude": "Ability & Aptitude",
}


JOB_LEVELS = {
    "graduate": "Graduate",
    "entry": "Entry-Level",
    "entry level": "Entry-Level",
    "mid": "Mid-Professional",
    "mid-level": "Mid-Professional",
    "senior": "Professional Individual Contributor",
    "manager": "Manager",
    "director": "Director",
}


# ---------------------------------------------------------------------
# Main Extractor
# ---------------------------------------------------------------------

def extract_constraints(messages: list[dict]) -> ConversationState:
    """
    Reads the entire conversation and extracts the
    current hiring constraints.
    """

    state = ConversationState()

    for message in messages:

        if message.get("role") != "user":
            continue

        text = message.get("content", "").lower()

        # -----------------------------
        # Experience
        # -----------------------------

        match = re.search(r"(\d+)\s*(year|years)", text)

        if match:
            state.experience = match.group(1) + " years"

        # -----------------------------
        # Remote
        # -----------------------------

        if "remote" in text:

            if "no remote" in text:
                state.remote = "no"
            else:
                state.remote = "yes"

        # -----------------------------
        # Adaptive
        # -----------------------------

        if "adaptive" in text:
            state.adaptive = "yes"

        # -----------------------------
        # Job Level
        # -----------------------------

        for key, value in JOB_LEVELS.items():

            if key in text:
                state.job_level = value

        # -----------------------------
        # Assessment Types
        # -----------------------------

        for key, value in ASSESSMENT_TYPES.items():

            if key in text and value not in state.assessment_types:
                state.assessment_types.append(value)

        # -----------------------------
        # Skills
        # -----------------------------

        for skill in TECH_SKILLS:

            if skill in text and skill.title() not in state.skills:
                state.skills.append(skill.title())

        # -----------------------------
        # Role
        # -----------------------------

        for role in ROLE_KEYWORDS:

            if role in text:
                state.role = text

        # -----------------------------
        # Compare
        # -----------------------------

        if "compare" in text:

            names = re.findall(r"[A-Za-z0-9\-\+]+", message["content"])

            state.compare = names

    return state


# ---------------------------------------------------------------------
# Build Search Query
# ---------------------------------------------------------------------

def build_search_query(state: ConversationState) -> str:
    """
    Converts the extracted conversation state
    into a single search query for FAISS.
    """

    parts = []

    if state.role:
        parts.append(state.role)

    if state.job_level:
        parts.append(state.job_level)

    if state.experience:
        parts.append(state.experience)

    if state.skills:
        parts.append(" ".join(state.skills))

    if state.assessment_types:
        parts.append(" ".join(state.assessment_types))

    if state.remote:
        parts.append("Remote")

    if state.adaptive:
        parts.append("Adaptive")

    return " ".join(parts)


# ---------------------------------------------------------------------
# Enough Information?
# ---------------------------------------------------------------------

def needs_clarification(state: ConversationState) -> bool:
    """
    Determines whether enough information
    exists to recommend assessments.
    """

    if not state.role and not state.skills:
        return True

    return False


# ---------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------

if __name__ == "__main__":

    messages = [
        {
            "role": "user",
            "content": "Hiring Java Developer"
        },
        {
            "role": "assistant",
            "content": "Experience?"
        },
        {
            "role": "user",
            "content": "4 years"
        },
        {
            "role": "assistant",
            "content": "Anything else?"
        },
        {
            "role": "user",
            "content": "Include personality tests and remote."
        }
    ]

    state = extract_constraints(messages)

    print(state)

    print()

    print(build_search_query(state))

    print()

    print(needs_clarification(state))