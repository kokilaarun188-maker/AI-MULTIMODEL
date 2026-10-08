from typing import List

from fastapi import APIRouter, HTTPException

from config import CHAT_MODEL, EMBEDDING_MODEL, MATCH_COUNT, cohere_client, supabase
from schemas import ChatMessageOut, ChatRequest, ChatResponse, NewSessionResponse, SourceChunk

router = APIRouter()

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided "
    "document excerpts. If the excerpts don't contain the answer, say you don't "
    "know instead of making something up. Give thorough, well-explained answers: "
    "define key terms, explain the reasoning or context from the excerpts, and "
    "use examples from the document where relevant."
)

@router.post("/sessions", response_model=NewSessionResponse)
def create_session():
    created = supabase.table("chat_sessions").insert({}).execute()
    return NewSessionResponse(session_id=created.data[0]["id"])


@router.get("/sessions/{session_id}/history", response_model=List[ChatMessageOut])
def get_history(session_id: str):
    if not session_id or not session_id.strip():
        raise HTTPException(400, "session_id is required")

    result = (
        supabase.table("chat_messages")
        .select("role, content, sources, created_at")
        .eq("session_id", session_id)
        .order("created_at")
        .execute()
    )
    return result.data


def _embed_query(text: str) -> List[float]:
    response = cohere_client.embed(
        model=EMBEDDING_MODEL,
        texts=[text],
        input_type="search_query",
        embedding_types=["float"],
    )
    return response.embeddings.float_[0]


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(400, "message cannot be empty")

    # Guard against a missing/blank session_id BEFORE it hits Postgres as an
    # invalid UUID (which previously caused an unhandled 500).
    if not request.session_id or not request.session_id.strip():
        raise HTTPException(
            400,
            "session_id is required. Upload a PDF first (POST /upload) or "
            "create a session (POST /sessions), then send the returned "
            "session_id with your chat message.",
        )

    session_check = (
        supabase.table("chat_sessions")
        .select("id")
        .eq("id", request.session_id)
        .execute()
    )
    if not session_check.data:
        raise HTTPException(404, "session_id not found. Create a session first or upload a PDF.")

    try:
        query_embedding = _embed_query(request.message)
    except Exception as exc:
        raise HTTPException(500, f"Embedding failed: {exc}")

    try:
        matches = supabase.rpc(
            "match_document_chunks",
            {
                "query_embedding": query_embedding,
                "match_session_id": request.session_id,
                "match_count": MATCH_COUNT,
            },
        ).execute()
    except Exception as exc:
        raise HTTPException(500, f"Retrieval failed: {exc}")

    chunks = matches.data or []
    sources = [
        SourceChunk(
            document_id=c["document_id"],
            content=c["content"],
            similarity=c["similarity"],
        )
        for c in chunks
    ]

    if chunks:
        context = "\n\n---\n\n".join(c["content"] for c in chunks)
        user_content = (
            f"Document excerpts:\n{context}\n\nQuestion: {request.message}"
        )
    else:
        user_content = (
            f"No document excerpts were found for this session. "
            f"Question: {request.message}"
        )

    try:
        response = cohere_client.chat(
            model=CHAT_MODEL,
            max_tokens=800,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        answer = "".join(
            block.text for block in (response.message.content or []) if hasattr(block, "text")
        ).strip()
    except Exception as exc:
        raise HTTPException(500, f"Chat generation failed: {exc}")

    if not answer:
        answer = "I wasn't able to generate a response. Please try again."

    # Persist the exchange
    supabase.table("chat_messages").insert(
        {"session_id": request.session_id, "role": "user", "content": request.message}
    ).execute()
    supabase.table("chat_messages").insert(
        {
            "session_id": request.session_id,
            "role": "assistant",
            "content": answer,
            "sources": [s.model_dump() for s in sources],
        }
    ).execute()

    return ChatResponse(session_id=request.session_id, answer=answer, sources=sources)