# TypeSafe JEV evaluation

This project uses the official TypeSafe Python SDK rather than a guessed generic endpoint.

## Configure
Set TYPESAFE_API_KEY in your local .env file, or enter the key in the JEV API key field in the Streamlit sidebar. The default model is jev-latest. You can optionally set JEV_MODEL to a model alias supported by your TypeSafe account.

## What JEV evaluates
The app sends the user's question, retrieved PDF chunks (including source filenames and page numbers), and optional draft answer as state to TypeSafe JEV. It asks three Noul judgments:
- Retrieval relevance: how directly the retrieved chunks address the question.
- Evidence sufficiency: whether the retrieved context contains enough evidence to answer.
- Answer groundedness: whether the draft answer is supported by the retrieved evidence.

Before answer generation, the app evaluates retrieval relevance and sufficiency. If either is below the configured threshold, the app retrieves a wider set of chunks once. After answer generation, it re-evaluates the final answer for groundedness.

## Interpreting results
Noul values are model judgments on a 0-to-1 scale. They are not empirically measured accuracy, and the SDK's confidence is not proof of correctness. Compare JEV scores with a manually labeled set of questions and relevant pages, and report objective Precision@K and Recall@K next to model-judge values.

## Official references
- TypeSafe SDK: https://github.com/typesafe-ai/typesafe-sdk-python
- TypeSafe Python guide: https://typesafe-jev.com/en/guides/python/
