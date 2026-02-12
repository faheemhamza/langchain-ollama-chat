from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from loaders.txt_loader import load_txt
from loaders.pdf_loader import load_pdf
from loaders.docx_loader import load_docx
from chunking.semantic_chunker import build_semantic_chunker, chunk_documents

EXTS = {".txt", ".pdf", ".docx"}

def find_files(dir):
    base = Path(dir)
    return [p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in EXTS]

def load_all(dir):
    paths = find_files(dir)
    txt = load_txt([p for p in paths if p.suffix == ".txt"])
    pdf = load_pdf([p for p in paths if p.suffix == ".pdf"])
    docx = load_docx([p for p in paths if p.suffix == ".docx"])
    return txt + pdf + docx

def build_or_load(chunks, persist, model):
    p = Path(persist)
    p.mkdir(exist_ok=True, parents=True)
    emb = OllamaEmbeddings(model=model)
    
    # Check if the vector store already exists
    if any(p.iterdir()):
        return Chroma(embedding_function=emb, persist_directory=str(p))
    
    return Chroma.from_documents(chunks, emb, persist_directory=str(p))

def build_retriever(
    data_dir,
    persist_dir,
    model="nomic-embed-text",
    rebuild=False,
    threshold=95
):
    p = Path(persist_dir)
    if rebuild and p.exists():
        import shutil
        shutil.rmtree(p)
    
    p.mkdir(exist_ok=True, parents=True)
    
    docs = load_all(data_dir)
    splitter = build_semantic_chunker(model=model, threshold_amount=threshold)
    chunks = chunk_documents(docs, splitter)
    
    vectordb = build_or_load(chunks, persist_dir, model)
    return vectordb.as_retriever(search_kwargs={"k": 4})