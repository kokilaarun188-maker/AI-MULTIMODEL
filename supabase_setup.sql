-- ============================================================
-- Run this in Supabase: Project -> SQL Editor -> New query -> Run
-- ============================================================

create extension if not exists vector;

create table if not exists chat_sessions (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz default now()
);

create table if not exists documents (
  id uuid primary key default gen_random_uuid(),
  session_id uuid not null references chat_sessions(id) on delete cascade,
  filename text not null,
  storage_path text not null,
  uploaded_at timestamptz default now()
);

-- NOTE: embed-english-v3.0 (Cohere) outputs 1024-dim vectors.
-- If you switch embedding models, update this dimension to match.
create table if not exists document_chunks (
  id uuid primary key default gen_random_uuid(),
  document_id uuid references documents(id) on delete cascade,
  session_id uuid not null references chat_sessions(id) on delete cascade,
  content text not null,
  embedding vector(1024),
  chunk_index int,
  created_at timestamptz default now()
);

create table if not exists chat_messages (
  id uuid primary key default gen_random_uuid(),
  session_id uuid references chat_sessions(id) on delete cascade,
  role text check (role in ('user','assistant')),
  content text not null,
  sources jsonb,
  created_at timestamptz default now()
);

create or replace function match_document_chunks (
  query_embedding vector(1024),
  match_session_id uuid,
  match_count int default 4
)
returns table (
  id uuid,
  document_id uuid,
  content text,
  similarity float
)
language sql stable
as $$
  select
    document_chunks.id,
    document_chunks.document_id,
    document_chunks.content,
    1 - (document_chunks.embedding <=> query_embedding) as similarity
  from document_chunks
  where document_chunks.session_id = match_session_id
  order by document_chunks.embedding <=> query_embedding
  limit match_count;
$$;

create index if not exists document_chunks_embedding_idx
  on document_chunks using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);
