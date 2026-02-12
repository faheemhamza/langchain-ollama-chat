from pathlib import Path
from typing import List
from langchain_core.documents import Document

def load_txt(files: List[Path]) -> List[Document]:
    docs = []
    for p in files:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            docs.append(Document(page_content=text, metadata={"source": str(p)}))
        except Exception as e:
            print(f"[TXT] Error: {p} : {e}")
            
    return docs  # <--- MUST be indented at the same level as 'docs = []'