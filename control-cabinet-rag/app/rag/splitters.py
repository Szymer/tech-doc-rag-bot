from __future__ import annotations

import re
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_documents(docs: List[Document]) -> List[Document]:
    cleaned_docs: List[Document] = []

    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
        cleaned_docs.append(doc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
    )

    chunks = splitter.split_documents(cleaned_docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks


