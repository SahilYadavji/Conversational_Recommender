"""
chatbot.py

Main chatbot orchestration for the SHL Assessment Recommender.
"""

from retriever import retriever
from llm import ask_llm
from prompt_builder import build_prompt

from conversation import (
    extract_constraints,
    needs_clarification,
)

from intent import detect_intent


def chat(messages):
    """
    Main chatbot pipeline.

    Flow:
    1. Extract conversation constraints
    2. Detect intent
    3. Clarify if needed
    4. Retrieve relevant assessments
    5. Build prompt
    6. Ask Gemini
    7. Return structured response
    """

    # --------------------------------------------------
    # Extract conversation state
    # --------------------------------------------------

    state = extract_constraints(messages)

    # --------------------------------------------------
    # Detect intent
    # --------------------------------------------------

    intent = detect_intent(messages)

    # --------------------------------------------------
    # Refuse off-topic requests
    # --------------------------------------------------

    if intent == "refuse":

        return {
            "reply": (
                "I'm an SHL Assessment Recommendation Assistant. "
                "I can only answer questions related to SHL "
                "assessments and assessment recommendations."
            ),
            "recommendations": [],
            "end_of_conversation": False,
        }

    # --------------------------------------------------
    # Clarification logic
    # --------------------------------------------------

    if intent == "clarify" or needs_clarification(state):

        if not state.role:

            question = (
                "What role are you hiring for? "
                "(For example: Java Developer, Sales Manager, QA Engineer)"
            )

        elif not state.experience:

            question = (
                "What experience level are you hiring for? "
                "(Entry-Level, Mid-Level, Senior, etc.)"
            )

        elif not state.assessment_types:

            question = (
                "Are you looking for technical, personality, "
                "ability, or simulation assessments?"
            )

        else:

            question = (
                "Do you have any additional requirements such as "
                "remote testing or adaptive assessments?"
            )

        return {
            "reply": question,
            "recommendations": [],
            "end_of_conversation": False,
        }

    # --------------------------------------------------
    # Retrieve assessments
    # --------------------------------------------------

    results = retriever.retrieve(state)

    if not results:

        return {
            "reply": (
                "I couldn't find any matching SHL assessments. "
                "Could you provide more information about the role "
                "or required skills?"
            ),
            "recommendations": [],
            "end_of_conversation": False,
        }

    # --------------------------------------------------
    # Build prompt
    # --------------------------------------------------

    prompt = build_prompt(messages, results)

    # --------------------------------------------------
    # Ask Gemini
    # --------------------------------------------------

    reply = ask_llm(prompt)

    # --------------------------------------------------
    # Build structured recommendations
    # --------------------------------------------------

    recommendations = []

    for item in results[:10]:

        recommendations.append(
            {
                "name": item.get("name", ""),
                "url": item.get("link", ""),
                "test_type": ", ".join(item.get("keys", [])),
            }
        )

    # --------------------------------------------------
    # Decide conversation completion
    # --------------------------------------------------

    end = intent == "recommend" and len(recommendations) > 0

    # --------------------------------------------------
    # Final response
    # --------------------------------------------------

    return {
        "reply": reply,
        "recommendations": recommendations,
        "end_of_conversation": end,
    }