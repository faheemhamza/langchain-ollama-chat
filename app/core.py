from typing import Dict
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import SystemMessage
from app.settings import settings
from app.prompts import BASE_SYSTEM_PROMPT

_sessions = {}

def get_history(session_id: str):
    if session_id not in _sessions:
        _sessions[session_id] = ChatMessageHistory()
    return _sessions[session_id]

def build_chain(system_prompt, retriever=None):
    llm = ChatOllama(
        model=settings.OLLAMA_MODEL,
        temperature=settings.TEMPERATURE
    )

    prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful assistant. Use the following pieces of retrieved context "
        "to answer the user's question. If you don't know the answer, just say "
        "that you don't know.\n\n"
        "Context:\n{context}"
    )),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

    def build_context(payload: Dict):
        if retriever:
            docs = retriever.invoke(payload["input"])
            parts = []
            for d in docs:
                src = d.metadata.get("source", "unknown")
                page = d.metadata.get("page")
                tag = f"{src}" + (f" (page {page})" if page else "")
                parts.append(f"[{tag}]\n{d.page_content}")
            payload["context"] = "\n\n".join(parts)
        else:
            payload["context"] = ""
        return payload

    chain = prompt | llm

    def history_loader(id: str):
        return get_history(id)

    runnable = RunnableWithMessageHistory(
        chain,
        history_loader,
        input_messages_key="input",
        history_messages_key="history"
    )

    def invoke(payload: Dict, session_id: str):
        # 1. Fetch context from RAG
        payload = build_context(payload)
        
        # 2. Ensure we are passing exactly what the prompt needs: "input" and "context"
        # The "history" key is managed automatically by the wrapper
        return runnable.invoke(
            {
                "input": payload["input"],
                "context": payload["context"]
            }, 
            config={"configurable": {"session_id": session_id}}
        )

    # Correct Indentation: This returns the 'invoke' function 
    # to the 'build_chain' caller.
    return invoke