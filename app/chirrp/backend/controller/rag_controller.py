import json
import importlib
import os
import urllib.request
from pathlib import Path
from typing import Optional

import psycopg
from dotenv import dotenv_values, load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from chirrp.backend.dbo.db_config import resolve_postgres_url
from chirrp.rag_pipeline.ingest.retrieve_gemini import (
    build_prompt,
    evidence_is_weak,
    pick_top_docs,
    retrieve_chunks_multi_query,
    summarize_with_gemini,
)

router = APIRouter(prefix="/rag", tags=["RAG"])

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_GEN_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={key}"

_embedder = None


def _resolve_database_url() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    root_env_path = repo_root / ".env"
    rag_env_path = repo_root / "rag_pipeline" / ".env"

    load_dotenv(dotenv_path=root_env_path, override=False)
    load_dotenv(dotenv_path=rag_env_path, override=False)

    dsn = resolve_postgres_url()
    if not dsn:
        raise RuntimeError(
            "RAG database configuration missing. Set RAG_DATABASE_URL (or DATABASE_URL), "
            "or define POSTGRES_USER/POSTGRES_PASSWORD/POSTGRES_HOST/POSTGRES_PORT/POSTGRES_DB in .env."
        )

    return dsn


def get_embedder():
    global _embedder
    if _embedder is None:
        from chirrp.rag_pipeline.ingest.embed_gemini import GeminiEmbedder

        _embedder = GeminiEmbedder(output_dim=1536)
    return _embedder


def _call_gemini_plain(question: str) -> str:
    url = GEMINI_GEN_URL.format(key=GEMINI_API_KEY)
    payload = json.dumps(
        {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                "Answer this question using only your general training knowledge. "
                                "Be specific and detailed.\n\n"
                                f"Question: {question}"
                            )
                        }
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.7},
        }
    ).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        return f"General LLM error: {e}"


class AskRequest(BaseModel):
    question: str
    top_k: int = 8
    min_score: float = 0.45
    show_sources: bool = True


class SourceItem(BaseModel):
    title: str
    url: str
    chunk_index: int
    relevance_score: float
    excerpt: str


class AskResponse(BaseModel):
    question: str
    answer: str
    show_sources: bool
    sources_used: list[SourceItem]
    confidence_score: float
    hallucination_risk: str


class CompareRequest(BaseModel):
    question: str
    top_k: int = 8


class CompareResponse(BaseModel):
    question: str
    rag_answer: str
    general_answer: str
    rag_sources: list[dict]
    rag_confidence: float
    rag_risk: str


class ReindexResponse(BaseModel):
    status: str
    docs_indexed: int = 0
    chunks_indexed: int = 0
    detail: str = ""


def _score_risk(rows: list) -> tuple[float, str]:
    if not rows:
        return 0.0, "high"
    scores = [r[6] for r in rows]
    avg = round(sum(scores) / len(scores), 4)
    top = scores[0]
    if top >= 0.70 and len(rows) >= 3:
        risk = "low"
    elif top >= 0.55:
        risk = "medium"
    else:
        risk = "high"
    return avg, risk


def _count_indexed_rows() -> tuple[int, int]:
    with psycopg.connect(_resolve_database_url()) as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM source_docs")
        docs = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM doc_chunks")
        chunks = int(cur.fetchone()[0])
    return docs, chunks


@router.get("/health")
def health():
    try:
        with psycopg.connect(_resolve_database_url()) as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM doc_chunks")
            chunks = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM source_docs")
            docs = cur.fetchone()[0]
        return {"status": "ok", "docs_indexed": docs, "chunks_indexed": chunks}
    except Exception as e:
        return {"status": "db_error", "detail": str(e)}


@router.post("/reindex", response_model=ReindexResponse)
def reindex_rag():
    try:
        fetch_sources_main = importlib.import_module("chirrp.rag_pipeline.fetch_sources").main
        parse_extract_main = importlib.import_module("chirrp.rag_pipeline.ingest.parse_extract").main
        load_pg_main = importlib.import_module("chirrp.rag_pipeline.ingest.load_pg").main

        fetch_sources_main()
        parse_extract_main()
        load_pg_main()

        docs, chunks = _count_indexed_rows()
        return ReindexResponse(
            status="ok",
            docs_indexed=docs,
            chunks_indexed=chunks,
            detail="RAG index rebuilt successfully.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sources")
def list_sources():
    try:
        with psycopg.connect(_resolve_database_url()) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT doc_id, title, final_url, content_type FROM source_docs ORDER BY title"
            )
            rows = cur.fetchall()
        return [
            {"doc_id": r[0], "title": r[1], "url": r[2], "type": r[3]}
            for r in rows
        ]
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    embedder = get_embedder()
    dsn = _resolve_database_url()

    rows_all = retrieve_chunks_multi_query(
        dsn=dsn,
        embedder=embedder,
        question=req.question,
        top_k_per_query=req.top_k,
        min_score=req.min_score,
    )

    if not rows_all or evidence_is_weak(rows_all):
        return AskResponse(
            question=req.question,
            answer=(
                "The retrieved sources do not contain enough information to answer "
                "this question confidently. Try rephrasing or ask something more "
                "specific about Nebraska groundwater law."
            ),
            show_sources=req.show_sources,
            sources_used=[],
            confidence_score=0.0,
            hallucination_risk="high",
        )

    rows = pick_top_docs(rows_all, docs_k=3, chunks_per_doc=2)
    prompt = build_prompt(req.question, rows)
    answer = summarize_with_gemini(prompt)

    avg_score, risk = _score_risk(rows)

    sources = []
    if req.show_sources:
        sources = [
            SourceItem(
                title=r[2],
                url=r[3],
                chunk_index=r[4],
                relevance_score=round(float(r[6]), 4),
                excerpt=r[5][:350] + "..." if len(r[5]) > 350 else r[5],
            )
            for r in rows
        ]

    return AskResponse(
        question=req.question,
        answer=answer,
        show_sources=req.show_sources,
        sources_used=sources,
        confidence_score=avg_score,
        hallucination_risk=risk,
    )


class ScenarioContextRequest(BaseModel):
    percent_change: float
    q_rate: Optional[float] = None
    max_drawdown: Optional[float] = None
    instruction: Optional[str] = None


class ScenarioContextResponse(BaseModel):
    available: bool
    query: str = ""
    note: str
    sources: list[SourceItem] = []


@router.post("/scenario-context", response_model=ScenarioContextResponse)
def scenario_context(req: ScenarioContextRequest):
    """Regulations relevant to a PINN scenario — the RAG↔PINN decision link.

    Retrieval-only (no LLM synthesis) so it never consumes generate-content
    quota, and best-effort: a missing corpus, offline DB, or embedding-quota
    error degrades to 'unavailable' rather than failing the scenario.
    """
    direction = "increasing" if req.percent_change > 0 else "reducing"
    query = (f"Nebraska groundwater regulations relevant to {direction} pumping "
             f"by {abs(req.percent_change):.0f}%")
    if req.max_drawdown is not None:
        query += f" with a predicted drawdown of about {req.max_drawdown:.1f} m"
    query += (": permits, allocation limits, well spacing, Natural Resources "
              "District rules, and required approvals.")

    try:
        embedder = get_embedder()
        dsn = _resolve_database_url()
        rows_all = retrieve_chunks_multi_query(
            dsn=dsn, embedder=embedder, question=query,
            top_k_per_query=6, min_score=0.4)

        if not rows_all or evidence_is_weak(rows_all):
            return ScenarioContextResponse(
                available=False, query=query,
                note=("No closely matching regulations were found for this "
                      "scenario in the indexed policy corpus."))

        rows = pick_top_docs(rows_all, docs_k=3, chunks_per_doc=1)
        sources = [
            SourceItem(
                title=r[2], url=r[3], chunk_index=r[4],
                relevance_score=round(float(r[6]), 4),
                excerpt=r[5][:300] + "..." if len(r[5]) > 300 else r[5],
            )
            for r in rows
        ]
        return ScenarioContextResponse(
            available=True, query=query,
            note=("Relevant Nebraska groundwater rules retrieved for this "
                  "scenario — planning context, not a legal determination."),
            sources=sources)
    except Exception:
        return ScenarioContextResponse(
            available=False, query=query,
            note=("Regulatory context is unavailable — the policy corpus is "
                  "not loaded or the lookup service is offline."))


@router.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest):
    embedder = get_embedder()
    dsn = _resolve_database_url()

    rows_all = retrieve_chunks_multi_query(
        dsn=dsn,
        embedder=embedder,
        question=req.question,
        top_k_per_query=req.top_k,
        min_score=0.40,
    )

    rows = pick_top_docs(rows_all, docs_k=3, chunks_per_doc=2) if rows_all else []
    rag_prompt = build_prompt(req.question, rows) if rows else f"Answer this: {req.question}"
    rag_ans = summarize_with_gemini(rag_prompt)
    general_ans = _call_gemini_plain(req.question)

    avg_score, risk = _score_risk(rows)

    return CompareResponse(
        question=req.question,
        rag_answer=rag_ans,
        general_answer=general_ans,
        rag_sources=[
            {"title": r[2], "url": r[3], "score": round(float(r[6]), 4)}
            for r in rows
        ],
        rag_confidence=avg_score,
        rag_risk=risk,
    )
