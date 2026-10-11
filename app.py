from __future__ import annotations

import os
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from jev_client import JEVClient, JEVConfigurationError, JEVError
from llm_client import OpenRouterClient, OpenRouterError
from rag_pipeline import RAGPipeline

load_dotenv()
st.set_page_config(page_title="JEV-RAG", page_icon="🔎", layout="wide")


def _secret(name: str, default: str = "") -> str:
    """Read Streamlit secrets first, then environment variables."""
    try:
        value = st.secrets.get(name, "")
        if value:
            return str(value)
    except (FileNotFoundError, RuntimeError):
        pass
    return os.getenv(name, default)


@st.cache_resource(show_spinner="Loading local embedding model and vector database...")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline(
        persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db"),
        collection_name=os.getenv("CHROMA_COLLECTION", "pdf_knowledge"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
    )


def main() -> None:
    st.title("JEV-RAG")
    st.caption("PDF knowledge retrieval · JEV evaluation · adaptive retrieval")

    with st.sidebar:
        st.header("Settings")
        openrouter_key = st.text_input(
            "OpenRouter API key", value=_secret("OPENROUTER_API_KEY"), type="password"
        )
        openrouter_model = st.text_input(
            "OpenRouter model",
            value=_secret("OPENROUTER_MODEL", "openrouter/free"),
        )
        jev_key = st.text_input(
            "TypeSafe JEV API key", value=_secret("TYPESAFE_API_KEY"), type="password"
        )
        jev_model = st.text_input(
            "JEV model", value=_secret("JEV_MODEL", "jev-latest")
        )
        st.caption("Sidebar keys are session-only. Use .env for local persistence.")
        top_k = st.slider("Chunks per retrieval", 2, 12, 5)
        retry_enabled = st.checkbox(
            "Retry once when JEV says evidence is insufficient", value=True
        )
        st.divider()
        st.caption("Default OpenRouter model: openrouter/free")

    try:
        pipeline = get_pipeline()
    except Exception as exc:
        st.error(f"Could not initialize RAG: {exc}")
        st.info("Check dependencies and network access for the first-time embedding model download.")
        st.stop()

    tab_docs, tab_ask, tab_eval, tab_setup = st.tabs(
        ["📚 Knowledge base", "💬 Ask & evaluate", "📊 Evaluation", "⚙️ Setup"]
    )

    with tab_docs:
        st.subheader("Add PDF documents")
        uploads = st.file_uploader(
            "Choose PDF files", type=["pdf"], accept_multiple_files=True
        )
        left, right = st.columns([1, 2])
        with left:
            if st.button("Index selected PDFs", type="primary", disabled=not uploads):
                try:
                    with st.spinner("Extracting text, chunking and embedding..."):
                        results = pipeline.ingest_uploaded_files(uploads)
                    for item in results:
                        if item["chunks"]:
                            st.success(
                                f"{item['filename']}: indexed {item['chunks']} chunks across "
                                f"{item['pages_with_text']} pages."
                            )
                        else:
                            st.warning(
                                f"{item['filename']}: no extractable text. Scanned PDFs may need OCR."
                            )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")
        with right:
            st.metric("Indexed chunks", pipeline.count_chunks())
            st.caption("Chunks and metadata are stored locally in ChromaDB.")
        if st.button("Clear indexed knowledge"):
            pipeline.clear()
            st.success("Knowledge base cleared.")
            st.rerun()
        sources = pipeline.list_sources()
        if sources:
            st.dataframe(pd.DataFrame(sources), use_container_width=True, hide_index=True)
        else:
            st.info("No PDFs indexed yet.")

    with tab_ask:
        st.subheader("Ask a question about your PDFs")
        question = st.text_area(
            "Question", placeholder="What are the main conclusions in the documents?", height=90
        )
        if st.button(
            "Retrieve, evaluate & answer", type="primary", disabled=not question.strip()
        ):
            if pipeline.count_chunks() == 0:
                st.warning("Index at least one PDF first.")
            elif not openrouter_key:
                st.warning(
                    "Enter an OpenRouter API key in the sidebar or configure "
                    "OPENROUTER_API_KEY in .env."
                )
            else:
                try:
                    with st.spinner("Retrieving evidence and evaluating context..."):
                        chunks = pipeline.retrieve(question, top_k=top_k)
                        evaluation: dict[str, Any] | None = None
                        retry_used = False
                        evaluator = (
                            JEVClient(api_key=jev_key, model=jev_model)
                            if jev_key.strip()
                            else None
                        )
                        if evaluator:
                            evaluation = evaluator.evaluate(question, chunks)
                            if retry_enabled and evaluation.get("needs_retry", False):
                                chunks = pipeline.retrieve(
                                    question, top_k=min(top_k * 2, 20), diversity=True
                                )
                                retry_used = True
                                evaluation = evaluator.evaluate(question, chunks)
                        answerer = OpenRouterClient(
                            api_key=openrouter_key,
                            model=openrouter_model or "openrouter/free",
                        )
                        answer = answerer.answer(question, chunks)
                        if evaluator:
                            evaluation = evaluator.evaluate(question, chunks, answer=answer)
                        else:
                            st.warning(
                                "JEV is not configured. RAG answered, but no JEV score was generated."
                            )
                except (OpenRouterError, JEVError, JEVConfigurationError) as exc:
                    st.error(str(exc))
                    st.stop()
                except Exception as exc:
                    st.error(f"Request failed: {exc}")
                    st.stop()
                st.session_state["last_result"] = {
                    "question": question,
                    "answer": answer,
                    "chunks": [chunk.to_dict() for chunk in chunks],
                    "evaluation": evaluation,
                    "retry_used": retry_used,
                    "model": openrouter_model or "openrouter/free",
                }

        result = st.session_state.get("last_result")
        if result:
            st.markdown("### Answer")
            st.markdown(result["answer"])
            st.caption(
                f"Model: {result['model']} · Wider retrieval retry: "
                f"{'Yes' if result['retry_used'] else 'No'}"
            )
            st.markdown("### Retrieved evidence")
            for index, chunk in enumerate(result["chunks"], 1):
                with st.expander(
                    f"[{index}] {chunk['source']} — page {chunk['page']} · "
                    f"distance {chunk.get('distance')}"
                ):
                    st.write(chunk["text"])
            st.markdown("### JEV evaluation")
            if result["evaluation"]:
                evaluation = result["evaluation"]
                cols = st.columns(3)
                keys = ["relevance", "sufficiency", "groundedness"]
                labels = [
                    "Retrieval relevance (JEV)",
                    "Evidence sufficiency (JEV)",
                    "Answer groundedness (JEV)",
                ]
                for col, key, label in zip(cols, keys, labels):
                    value = evaluation.get(key)
                    col.metric(
                        label, f"{float(value):.2f} / 1.00" if value is not None else "N/A"
                    )
                st.write(evaluation.get("reasoning") or "No explanation supplied by JEV.")
                confidence = evaluation.get("judge_confidence")
                if confidence is not None:
                    st.caption(f"JEV judgment confidence: {float(confidence):.2f}")
                st.caption(
                    "These are JEV model judgments, not measured retrieval accuracy. "
                    "Validate against human-labeled questions and reference passages."
                )
                st.json(evaluation)
            else:
                st.info("Configure TypeSafe JEV in the sidebar to evaluate this answer.")

    with tab_eval:
        st.subheader("Objective retrieval evaluation")
        st.write(
            "Upload a CSV with columns question and relevant_sources. Separate expected PDF "
            "filenames in relevant_sources with semicolons."
        )
        st.caption(
            "This starter metric evaluates source-level matches. For robust chunk-level recall, "
            "label relevant chunks or page ranges."
        )
        labels_file = st.file_uploader("Evaluation labels CSV", type=["csv"], key="labels_csv")
        eval_k = st.slider("K for offline evaluation", 1, 10, 5)
        if labels_file is not None and st.button("Run offline retrieval evaluation"):
            try:
                labels = pd.read_csv(labels_file)
                required = {"question", "relevant_sources"}
                if not required.issubset(labels.columns):
                    st.error("CSV must include question and relevant_sources columns.")
                else:
                    rows = []
                    for _, row in labels.iterrows():
                        relevant = {
                            item.strip()
                            for item in str(row["relevant_sources"]).split(";")
                            if item.strip()
                        }
                        retrieved = pipeline.retrieve(str(row["question"]), top_k=eval_k)
                        sources = [chunk.source for chunk in retrieved]
                        hits = sum(source in relevant for source in sources)
                        rows.append(
                            {
                                "question": row["question"],
                                "precision_at_k_source_level": hits / len(sources)
                                if sources
                                else 0.0,
                                "recall_at_k_source_level": len(set(sources) & relevant)
                                / len(relevant)
                                if relevant
                                else 0.0,
                                "relevant_sources": "; ".join(sorted(relevant)),
                                "retrieved_sources": "; ".join(sources),
                            }
                        )
                    metrics = pd.DataFrame(rows)
                    st.dataframe(metrics, use_container_width=True, hide_index=True)
                    first, second = st.columns(2)
                    first.metric(
                        "Mean source precision@K",
                        f"{metrics['precision_at_k_source_level'].mean():.1%}",
                    )
                    second.metric(
                        "Mean source recall@K",
                        f"{metrics['recall_at_k_source_level'].mean():.1%}",
                    )
                    st.download_button(
                        "Download results CSV",
                        metrics.to_csv(index=False).encode("utf-8"),
                        "retrieval_evaluation.csv",
                        "text/csv",
                    )
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")

    with tab_setup:
        st.subheader("API configuration")
        st.markdown(
            "OpenRouter: set OPENROUTER_API_KEY in .env or paste it into the sidebar. "
            "The default model is openrouter/free; free model availability and limits vary."
        )
        st.markdown(
            "TypeSafe JEV: set TYPESAFE_API_KEY in .env or paste it into the sidebar. "
            "The default model is jev-latest. No JEV endpoint URL is required when using the official SDK."
        )
        st.markdown(
            "Persistence: PDF text is processed in memory; vector chunks persist locally in "
            "ChromaDB. The .env file, PDFs, and database are ignored by Git."
        )


if __name__ == "__main__":
    main()
