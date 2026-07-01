"""
retriever.py

Hybrid retrieval layer for the SHL Assessment Recommender.

Responsibilities:
- Build the search query from conversation state.
- Call semantic search (FAISS).
- Filter and rerank results using SHL catalog metadata.
"""

from typing import List, Dict

from conversation import ConversationState, build_search_query
from search import search_assessments


class Retriever:
    """
    Hybrid retriever combining semantic search
    with metadata-based reranking.
    """

    def __init__(self, top_k: int = 10):
        self.top_k = top_k

    def retrieve(
        self,
        state: ConversationState,
    ) -> List[Dict]:

        query = build_search_query(state)

        results = search_assessments(query, top_k=25)

        ranked = []

        for item in results:

            score = float(item.get("score", 0))

            # -------------------------------------
            # Job Level Boost
            # -------------------------------------

            if (
                state.job_level
                and state.job_level in item.get("job_levels", [])
            ):
                score += 0.25

            # -------------------------------------
            # Assessment Type Boost
            # -------------------------------------

            if state.assessment_types:

                matched = set(state.assessment_types).intersection(
                    set(item.get("keys", []))
                )

                score += len(matched) * 0.20

            # -------------------------------------
            # Remote Boost
            # -------------------------------------

            if (
                state.remote == "yes"
                and item.get("remote", "").lower() == "yes"
            ):
                score += 0.10

            # -------------------------------------
            # Adaptive Boost
            # -------------------------------------

            if (
                state.adaptive == "yes"
                and item.get("adaptive", "").lower() == "yes"
            ):
                score += 0.10

            # -------------------------------------
            # Skill Matching
            # -------------------------------------

            description = item.get("description", "").lower()

            skill_matches = 0

            for skill in state.skills:

                if skill.lower() in description:
                    skill_matches += 1

            score += skill_matches * 0.10

            item["final_score"] = score

            ranked.append(item)

        # -------------------------------------
        # Remove duplicates
        # -------------------------------------

        unique = {}

        for item in ranked:

            name = item["name"]

            if (
                name not in unique
                or item["final_score"] > unique[name]["final_score"]
            ):
                unique[name] = item

        ranked = list(unique.values())

        # -------------------------------------
        # Sort
        # -------------------------------------

        ranked.sort(
            key=lambda x: x["final_score"],
            reverse=True,
        )

        return ranked[: self.top_k]


retriever = Retriever()