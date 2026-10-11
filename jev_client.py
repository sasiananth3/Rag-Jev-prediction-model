from __future__ import annotations

from typing import Any

import requests


class JEVConfigurationError(RuntimeError):
    pass


class JEVError(RuntimeError):
    pass


class JEVClient:
    """Generic REST adapter. Match its payload and parsing to your JEV API contract."""

    def __init__(self, base_url: str, api_key: str, timeout: int = 60) -> None:
        self.base_url = base_url.strip()
        self.api_key = api_key.strip()
        self.timeout = timeout
        if not self.base_url:
            raise JEVConfigurationError("JEV API URL is missing. Set JEV_API_URL.")
        if not self.api_key:
            raise JEVConfigurationError("JEV API key is missing. Set JEV_API_KEY.")

    @staticmethod
    def _build_payload(
        question: str, chunks: list[Any], answer: str
    ) -> dict[str, Any]:
        data = []
        for chunk in chunks:
            item = chunk.to_dict() if hasattr(chunk, "to_dict") else dict(chunk)
            data.append(
                {
                    "text": item.get("text", ""),
                    "source": item.get("source", "unknown"),
                    "page": item.get("page", 0),
                    "chunk_id": item.get("chunk_id", ""),
                }
            )
        return {
            "question": question,
            "retrieved_chunks": data,
            "answer": answer,
            "criteria": ["relevance", "sufficiency", "groundedness"],
            "response_schema": {
                "relevance": "number between 0 and 1",
                "sufficiency": "number between 0 and 1",
                "groundedness": "number between 0 and 1",
                "reasoning": "string",
                "needs_retry": "boolean",
            },
        }

    @staticmethod
    def _parse_response(data: Any) -> dict[str, Any]:
        if isinstance(data, dict) and isinstance(data.get("result"), dict):
            data = data["result"]
        if not isinstance(data, dict):
            raise JEVError("JEV response must be a JSON object.")
        result: dict[str, Any] = {}
        for key in ("relevance", "sufficiency", "groundedness"):
            value = data.get(key)
            if value is None:
                result[key] = None
                continue
            try:
                value = float(value)
            except (ValueError, TypeError) as exc:
                raise JEVError(f"JEV field '{key}' must be a number from 0 to 1.") from exc
            if not 0 <= value <= 1:
                raise JEVError(f"JEV field '{key}' must be between 0 and 1.")
            result[key] = value
        result["reasoning"] = str(data.get("reasoning", ""))
        result["needs_retry"] = bool(data.get("needs_retry", False))
        if all(result[key] is None for key in ("relevance", "sufficiency", "groundedness")):
            raise JEVError(
                "No evaluation scores found. Adapt _parse_response() to your actual JEV schema."
            )
        return result

    def evaluate(
        self, question: str, chunks: list[Any], answer: str = ""
    ) -> dict[str, Any]:
        payload = self._build_payload(question, chunks, answer)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            response = requests.post(
                self.base_url, headers=headers, json=payload, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise JEVError(f"Could not connect to JEV: {exc}") from exc
        if not response.ok:
            raise JEVError(
                f"JEV HTTP {response.status_code}: {response.text[:1000]}. "
                "Verify endpoint, credentials, and API schema."
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise JEVError("JEV endpoint did not return JSON.") from exc
        return self._parse_response(data)
