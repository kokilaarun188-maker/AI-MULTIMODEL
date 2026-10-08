import uuid
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from config import (
    COHERE_API_KEY,
    EMBEDDING_MODEL,
    STORAGE_BUCKET,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    cohere_client,
    supabase,
)
from schemas import UploadResponse
from utils import batched, chunk_text, extract_text_from_pdf

router = APIRouter()

EMBED_BATCH_SIZE = 90  # Cohere embed endpoint accepts up to 96 texts per call


def _get_or_create_session(session_id: Optional[str]) -> str:
    if session_id:
        existing = (
            supabase.table("chat_sessions")
            .select("id")
            .eq("id", session_id)
            .execute()
        )
        if existing.data:
            return session_id
        # session_id was provided but doesn't exist yet -- create it with that id
        supabase.table("chat_sessions").insert({"id": session_id}).execute()
        return session_id

    created = supabase.table("chat_sessions").insert({}).execute()
    return created.data[0]["id"]


def _embed_chunks(chunks: List[str]) -> List[List[float]]:
    embeddings: List[List[float]] = []
    for batch in batched(chunks, EMBED_BATCH_SIZE):
        response = cohere_client.embed(
            model=EMBEDDING_MODEL,
            texts=batch,
            input_type="search_document",
            embedding_types=["float"],
        )
        embeddings.extend(response.embeddings.float_)
    return embeddings


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
):
    if not COHERE_API_KEY:
        raise HTTPException(500, "Server is missing COHERE_API_KEY")

    if file.content_type != "application/pdf" and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(400, "Uploaded file is empty")

    text = extract_text_from_pdf(file_bytes)
    if not text:
        raise HTTPException(
            422,
            "Couldn't extract any text from this PDF. It may be a scanned/image-only "
            "PDF, which isn't supported yet.",
        )

    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    if not chunks:
        raise HTTPException(422, "No usable text chunks were produced from this PDF")

    session_id = _get_or_create_session(session_id)

    # Store the original PDF in Supabase Storage
    storage_path = f"{session_id}/{uuid.uuid4()}_{file.filename}"
    try:
        supabase.storage.from_(STORAGE_BUCKET).upload(
            storage_path,
            file_bytes,
            {"content-type": "application/pdf"},
        )
    except Exception as exc:
        raise HTTPException(500, f"Failed to upload file to storage: {exc}")

    # Create the document record
    doc_result = (
        supabase.table("documents")
        .insert(
            {
                "session_id": session_id,
                "filename": file.filename,
                "storage_path": storage_path,
            }
        )
        .execute()
    )
    document_id = doc_result.data[0]["id"]

    # Embed and store chunks
    try:
        embeddings = _embed_chunks(chunks)
    except Exception as exc:
        raise HTTPException(500, f"Embedding failed: {exc}")

    rows = [
        {
            "document_id": document_id,
            "session_id": session_id,
            "content": chunk,
            "embedding": embedding,
            "chunk_index": idx,
        }
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    for batch in batched(rows, 200):
        supabase.table("document_chunks").insert(batch).execute()

    return UploadResponse(
        session_id=session_id,
        document_id=document_id,
        filename=file.filename,
        chunks_created=len(chunks),
    )
