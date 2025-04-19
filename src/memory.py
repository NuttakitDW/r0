# src/memory.py
"""
Long‑term vector memory for R0
------------------------------
Uses Pinecone’s serverless index + OpenAI embeddings.

ENV VARS required (already in .env):
    OPENAI_API_KEY     = sk‑...
    PINECONE_API_KEY   = pc‑...
    PINECONE_ENV       = us-east-1        # ← region where index lives
    PINECONE_INDEX     = r0-memory        # ← exact index name
"""

from __future__ import annotations

import os, uuid
from typing import List
from dotenv import load_dotenv

# ---- 1. load .env early ----------------------------------------------------
load_dotenv(".env", override=True)

# ---- 2. third‑party libs ---------------------------------------------------
from pinecone import Pinecone                    # v3 client
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# ---- 3. config -------------------------------------------------------------
API_KEY = os.environ["PINECONE_API_KEY"]
ENV     = os.environ["PINECONE_ENV"]
INDEX   = os.environ["PINECONE_INDEX"]

# ---- 4. connect to data‑plane ----------------------------------------------
pc   = Pinecone(api_key=API_KEY)
idx  = pc.Index(INDEX, environment=ENV)          # raises if index missing

# ---- 5. langchain vector store ---------------------------------------------
emb  = OpenAIEmbeddings(model="text-embedding-3-small")
vs   = PineconeVectorStore(index=idx, embedding=emb)  # v3‑native wrapper

# ---- 6. tiny helpers -------------------------------------------------------
def save_memory(text: str, meta: dict | None = None) -> None:
    """Store a piece of text with optional metadata."""
    vs.add_texts([text], metadatas=[meta or {}], ids=[str(uuid.uuid4())])

def retrieve_memory(query: str, k: int = 4) -> List[str]:
    """Return up to *k* semantically similar memory snippets."""
    docs = vs.similarity_search(query, k=k)
    return [d.page_content for d in docs]
