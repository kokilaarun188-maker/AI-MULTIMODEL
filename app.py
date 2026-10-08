from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import upload, chat

app = FastAPI(title="AI PDF Chatbot Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, tags=["upload"])
app.include_router(chat.router, tags=["chat"])


@app.get("/")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
