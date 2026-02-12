from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Optional
from app.settings import settings
from app.core import build_chain

# Build retriever
rag = None
if settings.RAG_ENABLED:
    from app.rag.retriever import build_retriever
    rag = build_retriever(
        "data",
        settings.VECTOR_DIR,
        rebuild=True
    )

app = FastAPI(title="Local RAG Chatbot")

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str

# Initialize the chain with the RAG retriever (if enabled)
invoke = build_chain(settings.SYSTEM_PROMPT, rag)

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        resp = invoke({"input": req.message}, session_id=req.session_id)
        
        # Log the raw response to your terminal to see what's happening
        print(f"DEBUG: Raw LLM Response: {resp}")

        # Extract text based on different possible LangChain return types
        if isinstance(resp, dict) and "output" in resp:
            reply_text = resp["output"]
        elif hasattr(resp, 'content'):
            reply_text = resp.content
        else:
            reply_text = str(resp)

        return ChatResponse(reply=reply_text or "I couldn't generate a response.")
    except Exception as e:
        print(f"!!! CHAT ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Static UI
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def home():
    return RedirectResponse(url="/static/index.html")