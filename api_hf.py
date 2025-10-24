from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
import os
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer

PERSIST_DIR = Path("chroma_db")
COLLECTION_NAME = "networking_context"
EMBED_MODEL_NAME = "sentence-transformers/msmarco-distilbert-base-v4"
DEFAULT_TOP_K = 4


class AskRequest(BaseModel):
    question: str = Field(..., description="User question to retrieve context for")
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=20, description="Number of contexts to retrieve")


class ContextItem(BaseModel):
    rank: int
    source: Optional[str] = None
    page: Optional[int] = None
    text: str


class AskResponse(BaseModel):
    question: str
    top_k: int
    results: List[ContextItem]
    answer: str
    citations: List[dict]


app = FastAPI(title="Networking RAG")
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: Optional[SentenceTransformer] = None
_collection: Optional[chromadb.Collection] = None
_openai_client: Optional[OpenAI] = None


def _load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    return _model


def _load_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(PERSIST_DIR), settings=Settings(anonymized_telemetry=False))
        try:
            _collection = client.get_collection(COLLECTION_NAME)
        except Exception as e:
            raise e
    return _collection


def _load_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set. Provide it in environment or .env for summarization.")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _embed_query(text: str) -> List[float]:
    model = _load_model()
    vec = model.encode([text], convert_to_numpy=False, normalize_embeddings=True)
    return vec[0].tolist()


def _fetch_context(query_embedding: List[float], top_k: int) -> List[ContextItem]:
    collection = _load_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"],
    )
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    items: List[ContextItem] = []
    for idx, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        source = meta.get("source") if isinstance(meta, dict) else None
        page = meta.get("page") if isinstance(meta, dict) else None
        text = (doc or "").strip()
        items.append(ContextItem(rank=idx, source=source, page=page, text=text))
    return items


def _summarize_with_openai(question: str, contexts: List[ContextItem]) -> str:
    client = _load_openai_client()
    context_text = "\n\n".join(
        [f"[{c.rank}] Source: {c.source or 'unknown'}{(' · p' + str(c.page)) if c.page else ''}\n{c.text}" for c in contexts]
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a concise assistant. Answer the user using ONLY the provided context. "
                "Cite sources inline like [1], [2]. If unsure, say you don't know."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion: {question}",
        },
    ]
    resp = client.chat.completions.create(model="gpt-5-nano", messages=messages)
    return resp.choices[0].message.content.strip()


@app.get("/health")
def health():
    try:
        _load_model()
        _load_collection()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/")
def root_ui():
    index_path = Path("templates/index.html")
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI not found. Ensure templates/index.html exists.")
    return FileResponse(index_path)


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    q = payload.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        q_emb = _embed_query(q)
        contexts = _fetch_context(q_emb, payload.top_k)
        try:
            answer = _summarize_with_openai(q, contexts)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        citations = [
            {"index": c.rank, "source": c.source, "page": c.page}
            for c in contexts
        ]
        return AskResponse(question=q, top_k=payload.top_k, results=contexts, answer=answer, citations=citations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_hf:app", host="127.0.0.1", port=8000, reload=False)
