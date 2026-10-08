import os
from dotenv import load_dotenv
from supabase import create_client, Client
import cohere

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
STORAGE_BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET", "pdfs")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in .env")

if not COHERE_API_KEY:
    raise RuntimeError("Missing COHERE_API_KEY in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

cohere_client = cohere.ClientV2(api_key=COHERE_API_KEY)

# embed-english-v3.0 produces 1024-dimensional vectors.
# This must match the `vector(...)` dimension used in supabase_setup.sql.
EMBEDDING_MODEL = "embed-english-v3.0"
EMBEDDING_DIMENSIONS = 1024

CHAT_MODEL = "command-r-plus-08-2024"

# How the PDF text is split before embedding.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# How many chunks to retrieve per chat query.
MATCH_COUNT = 6

