"""
Q&A Mode API - Handles question answering with context retrieval
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import os

import chromadb
from chromadb.config import Settings
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# Constants
PERSIST_DIR = Path("chroma_db")
COLLECTION_NAME = "networking_context"
EMBED_MODEL_NAME = "sentence-transformers/msmarco-distilbert-base-v4"
DEFAULT_TOP_K = 4


# Pydantic Models
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
    snippet: str
    citations: List[dict]


# Global instances
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
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
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
                "You are a helpful networking and security expert. Provide detailed, comprehensive answers using ONLY the provided context. "
                "Include relevant technical details, examples, and explanations. "
                "Cite sources inline like [1], [2]. If unsure, say you don't know."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion: {question}",
        },
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=1000,
        temperature=0.7
    )
    return resp.choices[0].message.content.strip()


def ask_question(question: str, top_k: int = DEFAULT_TOP_K) -> AskResponse:
    """Main Q&A function"""
    q = question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        q_emb = _embed_query(q)
        contexts = _fetch_context(q_emb, top_k)
        answer = _summarize_with_openai(q, contexts)

        citations = [
            {"index": c.rank, "source": c.source, "page": c.page}
            for c in contexts
        ]
        return AskResponse(question=q, top_k=top_k, results=contexts, snippet=answer, citations=citations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
