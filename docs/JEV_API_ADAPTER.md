# JEV API adapter integration

JEV API deployments may differ. This repository uses a generic HTTP adapter instead of assuming an undocumented SDK or endpoint.

## Assumed request
POST to JEV_API_URL with bearer authorization and a JSON body:

    {
      "question": "question",
      "retrieved_chunks": [
        {"text": "passage", "source": "report.pdf", "page": 2, "chunk_id": "abc"}
      ],
      "answer": "",
      "criteria": ["relevance", "sufficiency", "groundedness"],
      "response_schema": {
        "relevance": "number between 0 and 1",
        "sufficiency": "number between 0 and 1",
        "groundedness": "number between 0 and 1",
        "reasoning": "string",
        "needs_retry": "boolean"
      }
    }

## Assumed response

    {
      "relevance": 0.9,
      "sufficiency": 0.7,
      "groundedness": 0.95,
      "reasoning": "A key detail is missing",
      "needs_retry": true
    }

Update JEVClient._build_payload, request headers, and JEVClient._parse_response to match the actual JEV API contract. Model scores must be compared with human-verified labels before being treated as accuracy metrics.
