from __future__ import annotations

import os
from typing import Any

from typesafe_sdk import Noul, TypeSafeClient


class JEVConfigurationError(RuntimeError):
    pass


class JEVError(RuntimeError):
    pass


class JEVClient:
    """Use TypeSafe JEV to judge retrieval evidence and answer grounding."""

    def __init__(
        self, api_key: str | None = None, model: str = "jev-latest"
    ) -> None:
        self.api_key = (api_key or os.getenv("TYPESAFE_API_KEY", "")).strip()
        self.model = (model or os.getenv("JEV_MODEL", "jev-latest")).strip()
        if not self.api_key:
            raise JEVConfigurationError(
                "JEV API key is missing. Set TYPESAFE_API_KEY in .env or the app sidebar."
            )

    @staticmethod
    def _chunk_payload(chunks: list[Any]) -> list[dict[str, Any]]:
        result = []
        for chunk in chunks:
            item = chunk.to_dict() if hasattr(chunk, "to_dict") else dict(chunk)
            result.append(
                {
                    "text": item.get("text", ""),
                    "source": item.get("source", "unknown"),
                    "page": item.get("page", 0),
                    "chunk_id": item.get("chunk_id", ""),
                }
            )
        return result

    def evaluate(
        self, question: str, chunks: list[Any], answer: str | None = None
    ) -> dict[str, Any]:
        """Return JEV Noul judgments normalized to [0, 1].

        Scores are model judgments, not empirically measured retrieval accuracy.
        When answer is empty, only retrieval relevance and sufficiency are judged.
        """
        state = {
            "question": question,
            "retrieved_chunks": self._chunk_payload(chunks),
            "draft_answer": answer or "(Answer not generated yet; evaluate retrieval only.)",
        }
        questions = {
            "retrieval_relevance": Noul(
                instructions=(
                    "To what degree are the retrieved PDF chunks directly relevant "
                    "to answering the user's question? Return a low value for unrelated "
                    "or misleading chunks and a high value when most retrieved evidence "
                    "directly addresses the question."
                )
            ),
            "evidence_sufficiency": Noul(
                instructions=(
                    "To what degree do the retrieved PDF chunks contain sufficient "
                    "information to answer the question accurately and completely? "
                    "Return a low value when key evidence is missing or conflicting."
                )
            ),
        }
        if answer:
            questions["answer_groundedness"] = Noul(
                instructions=(
                    "To what degree is the draft answer supported by the retrieved PDF "
                    "chunks? Return a low value for unsupported claims, invented facts, "
                    "or claims contradicted by the retrieved evidence."
                )
            )

        try:
            with TypeSafeClient(api_key=self.api_key, model=self.model) as client:
                response = client.system_one(state=state, questions=questions)
        except Exception as exc:
            raise JEVError(
                f"JEV evaluation failed ({type(exc).__name__}). Check the API key, "
                "model name, network connection, and TypeSafe account limits."
            ) from exc

        values: dict[str, Any] = {}
        confidences: list[float] = []
        for output_name, result_name in (
            ("retrieval_relevance", "relevance"),
            ("evidence_sufficiency", "sufficiency"),
            ("answer_groundedness", "groundedness"),
        ):
            if output_name not in response.nouls:
                values[result_name] = None
                continue
            judgement = response.nouls[output_name]
            score = float(judgement.noul)
            values[result_name] = min(1.0, max(0.0, score))
            confidence = getattr(judgement, "confidence", None)
            if confidence is not None:
                try:
                    confidences.append(float(confidence))
                except (TypeError, ValueError):
                    pass

        relevance = values.get("relevance")
        sufficiency = values.get("sufficiency")
        values["needs_retry"] = bool(
            relevance is not None
            and sufficiency is not None
            and (relevance < 0.55 or sufficiency < 0.60)
        )
        values["judge_confidence"] = (
            sum(confidences) / len(confidences) if confidences else None
        )
        values["reasoning"] = (
            "TypeSafe JEV Noul judgments: "
            f"retrieval relevance={relevance if relevance is not None else 'not evaluated'}, "
            f"evidence sufficiency={sufficiency if sufficiency is not None else 'not evaluated'}, "
            f"answer groundedness={values.get('groundedness') if values.get('groundedness') is not None else 'not evaluated'}. "
            "These are model judgments, not a human-verified accuracy percentage."
        )
        return values
