from pathlib import Path
from typing import List
from langchain_core.documents import Document

def try_unstructured(path: Path):
    try:
        from unstructured.partition.pdf import partition_pdf
        els = partition_pdf(filename=str(path))
        txt = "\n".join(e.text for e in els if hasattr(e, "text"))
        if txt.strip():
            return [Document(page_content=txt, metadata={"source": str(path)})]
    except:
        pass
    return []

def try_pypdf(path: Path):
    try:
        from pypdf import PdfReader
        r = PdfReader(str(path))
        docs = []
        for i, p in enumerate(r.pages):
            content = p.extract_text() or ""
            if content.strip():
                docs.append(Document(
                    page_content=content,
                    metadata={"source": str(path), "page": i + 1}
                ))
        return docs
    except:
        return []

def load_pdf(files: List[Path]) -> List[Document]:
    out = []
    for f in files:
        # Tries unstructured first; if it returns an empty list, tries pypdf
        d = try_unstructured(f) or try_pypdf(f)
        out.extend(d)
    return out