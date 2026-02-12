from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

def build_semantic_chunker(
    model="nomic-embed-text",
    threshold_type="percentile",
    threshold_amount=95
):
    try:
        emb = OllamaEmbeddings(model=model)
        return SemanticChunker(
            emb,
            breakpoint_threshold_type=threshold_type,
            breakpoint_threshold_amount=threshold_amount
        )
    except:
        # Fallback to character-based splitting if Ollama/Embeddings fail
        return RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)

def chunk_documents(documents, splitter):
    return splitter.split_documents(documents)