"""
models.py

Pydantic models for the SHL Assessment Recommender API.

Responsibilities
----------------
- Validate incoming requests.
- Define API response schemas.
- Provide automatic OpenAPI documentation.
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------
# Chat Models
# ---------------------------------------------------------------------


class Message(BaseModel):
    """
    Represents a single chat message.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    role: Literal["user", "assistant"] = Field(
        ...,
        description="Message sender.",
        examples=["user"],
    )

    content: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Message content.",
        examples=["I need an assessment for software engineers."],
    )


class ChatRequest(BaseModel):
    """
    Incoming chat request.
    """

    model_config = ConfigDict(extra="forbid")

    messages: List[Message] = Field(
        ...,
        min_length=1,
        description="Conversation history.",
    )


# ---------------------------------------------------------------------
# Recommendation Models
# ---------------------------------------------------------------------


class Recommendation(BaseModel):
    """
    SHL assessment recommendation.
    """

    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
    )

    name: str = Field(
        ...,
        description="Assessment name.",
    )

    url: str = Field(
        ...,
        description="Official SHL assessment URL.",
    )

    test_type: str = Field(
        ...,
        description="Assessment categories.",
    )


# ---------------------------------------------------------------------
# Response Model
# ---------------------------------------------------------------------


class ChatResponse(BaseModel):
    """
    API response returned to the frontend.
    """

    model_config = ConfigDict(extra="ignore")

    reply: str = Field(
        ...,
        description="LLM-generated response.",
    )

    recommendations: List[Recommendation] = Field(
        default_factory=list,
        description="Recommended SHL assessments.",
    )

    end_of_conversation: bool = Field(
        False,
        description=(
            "True when the assistant has enough information "
            "to make recommendations."
        ),
    )