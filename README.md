# JEV-RAG: Retrieval Evaluation & Optimization

A Streamlit application to upload PDFs, build a searchable knowledge base, answer questions using RAG, and evaluate retrieval evidence using TypeSafe JEV. OpenRouter answer generation defaults to its free-model router.

## Features
- PDF extraction with filename and page metadata
- Persistent ChromaDB with its built-in local ONNX MiniLM embedding function
- OpenRouter chat completions with the default model openrouter/free
- TypeSafe JEV evaluates retrieval relevance, evidence sufficiency, and answer groundedness
- One wider retrieval retry when JEV judges context insufficient
- Dashboard for retrieved evidence, JEV judgments, and objective CSV metrics
- API keys are read from .env or entered in the Streamlit sidebar, never hard-coded

> JEV scores are model judgments, not automatically verified accuracy percentages. Compare them with a human-labeled dataset and objective Precision@K / Recall@K.

## Install (Ubuntu/Linux)
    git clone https://github.com/sasiananth3/Rag-Jev-prediction-model.git
    cd Rag-Jev-prediction-model
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    cp .env.example .env

Edit .env with your API keys, then run:

    streamlit run app.py

Open the local URL printed by Streamlit, usually http://localhost:8501.

## Configure OpenRouter
1. Create an API key in your OpenRouter account.
2. Set OPENROUTER_API_KEY in .env or paste it in the app's sidebar.
3. The default model is openrouter/free. OpenRouter selects an available free model; availability and rate limits can vary.
4. Optionally set OPENROUTER_MODEL to a specific free-model slug supported by your account.

Requests use the OpenAI-compatible endpoint https://openrouter.ai/api/v1/chat/completions.

## Configure TypeSafe JEV
1. Get your TypeSafe JEV API key.
2. Add it as TYPESAFE_API_KEY in .env, or paste it into the TypeSafe JEV API key field in the sidebar.
3. The default model is jev-latest. You can optionally set JEV_MODEL to another JEV model alias supported by your account.
4. No endpoint URL is required: the app uses the official typesafe-sdk Python package.

See docs/JEV_API_ADAPTER.md for the evaluation design.

## Adaptive RAG flow
1. Extract and chunk PDF text, keeping source and page metadata.
2. Retrieve the most relevant chunks from ChromaDB.
3. JEV judges retrieval relevance and whether the evidence is sufficient.
4. If relevance or sufficiency is low, retrieve a wider set once.
5. Generate an answer grounded in retrieved chunks using OpenRouter.
6. JEV judges the final answer's grounding against retrieved evidence.

## Objective evaluation
The Evaluation tab accepts a CSV with question and relevant_sources columns. Separate source filenames with semicolons. It calculates starter source-level precision and recall. For rigorous chunk Recall@K, label relevant chunk IDs or page ranges. JEV judgments and objective metrics are displayed separately.

## Security
The .env file, PDFs, and ChromaDB are ignored by Git. Do not send sensitive PDF excerpts to hosted models unless your data policy permits it.

## Tests
    pytest -q
