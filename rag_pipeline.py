from __future__ import annotations

import hashlib
import io
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import chromadb
import fitz
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievalChunk:
    text: str
    source: str
    page: int
    chunk_id: str
    distance: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def split_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Split text into overlapping character windows, preferring word boundaries."""
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0.")
    text = " ".join(text.split())
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + int(chunk_size * 0.7), end)
            if boundary > start:
                end = boundary
        part = text[start:end].strip()
        if part:
            chunks.append(part)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


class RAGPipeline:
    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: str = "pdf_knowledge",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.persist_directory = str(Path(persist_directory))
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        self.embedder = SentenceTransformer(embedding_model)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    @staticmethod
    def _read_pdf(data: bytes) -> list[tuple[int, str]]:
        pages: list[tuple[int, str]] = []
        with fitz.open(stream=io.BytesIO(data), filetype="pdf") as document:
            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()
                if text:
                    pages.append((page_number, text))
        return pages

    def ingest_uploaded_files(self, uploads: Iterable[Any]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for upload in uploads:
            filename = Path(upload.name).name
            if Path(filename).suffix.lower() != ".pdf":
                continue
            pages = self._read_pdf(upload.getvalue())
            ids: list[str] = []
            documents: list[str] = []
            metadata: list[dict[str, Any]] = []
            for page_number, page_text in pages:
                for index, chunk in enumerate(split_text(page_text)):
                    chunk_id = hashlib.sha256(
                        f"{filename}:{page_number}:{index}:{chunk}".encode("utf-8")
                    ).hexdigest()[:32]
                    ids.append(chunk_id)
                    documents.append(chunk)
                    metadata.append(
                        {"source": filename, "page": page_number, "chunk_index": index}
                    )
            if ids:
                self.collection.upsert(ids=ids, documents=documents, metadatas=metadata)
            results.append(
                {"filename": filename, "chunks": len(ids), "pages_with_text": len(pages)}
            )
        return results

    def retrieve(
        self, question: str, top_k: int = 5, diversity: bool = False
    ) -> list[RetrievalChunk]:
        if not question.strip() or self.collection.count() == 0:
            return []
        query_k = min(max(1, top_k), int(self.collection.count()))
        result = self.collection.query(
            query_texts=[question],
            n_results=query_k,
            include=["documents", "metadatas", "distances"],
        )
        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0]
        chunks: list[RetrievalChunk] = []
        for index, (text, metadata) in enumerate(zip(docs, metadatas)):
            metadata = metadata or {}
            distance = distances[index] if index < len(distances) else None
            chunks.append(
                RetrievalChunk(
                    text=text,
                    source=str(metadata.get("source", "unknown")),
                    page=int(metadata.get("page", 0)),
                    chunk_id=str(ids[index]) if index < len(ids) else "",
                    distance=float(distance) if distance is not None else None,
                )
            )
        if diversity:
            chosen: list[RetrievalChunk] = []
            deferred: list[RetrievalChunk] = []
            seen: set[str] = set()
            for chunk in chunks:
                target = deferred if chunk.source in seen else chosen
                target.append(chunk)
                seen.add(chunk.source)
            chunks = chosen + deferred
        return chunks[:top_k]

    def count_chunks(self) -> int:
        return int(self.collection.count())

    def list_sources(self) -> list[dict[str, Any]]:
        result = self.collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for metadata in result.get("metadatas", []):
            source = (metadata or {}).get("source", "unknown")
            counts[source] = counts.get(source, 0) + 1
        return [
            {"filename": filename, "chunks": count}
            for filename, count in sorted(counts.items())
        ]

    def clear(self) -> None:
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )
