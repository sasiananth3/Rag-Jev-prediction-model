# JEV-RAG: Retrieval Evaluation & Optimization

A Streamlit application to upload PDFs, build a searchable knowledge base, ask questions through RAG, and evaluate retrieved evidence using a configurable JEV API adapter. OpenRouter answer generation defaults to `openrouter/free`.

## Features
- PDF extraction with filename/page metadata
- Persistent ChromaDB and local Sentence Transformer embeddings
- OpenRouter chat completion using the free-model router by default
- JEV evaluation for retrieval relevance, evidence sufficiency, and answer groundedness
- One wider retrieval retry if JEV reports insufficient evidence
- Dashboard for retrieved evidence, evaluation results, and CSV metrics
- API keys via `.env` or Streamlit sidebar, not hard-coded in source

> JEV scores are model judgments, not automatically verified accuracy percentages. Compare them with human-labeled examples and objective Precision@K / Recall@K.

## Install (Ubuntu/Linux)
```bash
git clone https://github.com/sasiananth3/Rag-Jev-prediction-model.git
cd Rag-Jev-prediction-model
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```
Edit `.env` with your keys, then run:
```bash
streamlit run app.py
```
Open `http://localhost:8501`.

## OpenRouter configuration
Set `OPENROUTER_API_KEY` in `.env` or paste it into the app sidebar. The default model is `openrouter/free`, which routes to an available free model; availability, rate limits, and capabilities can vary. Optionally set `OPENROUTER_MODEL` to a specific free model slug supported by your account. API endpoint: `https://openrouter.ai/api/v1/chat/completions`.

## JEV configuration
JEV API deployments can differ, so this repo does not assume an undocumented SDK or endpoint. Obtain the actual API URL and key from your JEV provider, then configure `JEV_API_URL` and `JEV_API_KEY`. The generic adapter expects a bearer-authenticated JSON POST containing `question`, `retrieved_chunks`, `answer`, and `criteria`. Expected response fields are `relevance`, `sufficiency`, `groundedness` (0-1), optional `reasoning`, and optional `needs_retry`. Adapt `jev_client.py` to your provider's actual API contract. Without JEV configured, RAG still works but no JEV scores are fabricated. See `docs/JEV_API_ADAPTER.md`.

## Evaluation
The Evaluation tab accepts a CSV with `question` and `relevant_sources` (semicolon-separated PDF filenames). It calculates starter source-level precision and recall. For rigorous chunk-level Recall@K, label relevant chunk IDs/pages. The app separates JEV judgments from objective metrics.

## Security
`.env`, PDFs, and ChromaDB are ignored by Git. Do not send sensitive PDF excerpts to hosted models unless allowed by your data policy.

## Tests
```bash
pytest -q
```
