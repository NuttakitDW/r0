# backend/server.py
from __future__ import annotations
import asyncio, json
from typing import List, Optional, AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.agent_graph import app as agent_app


# ── 1. Pydantic schemas -------------------------------------------------
class ChatRequest(BaseModel):
    session: Optional[str] = None     # conversation ID (optional)
    message: str                      # the user prompt
    stream: bool = True               # default = SSE streaming

class ChatResponse(BaseModel):
    result: str
    recalled: List[str] = []          # may be empty


# ── 2. FastAPI app ------------------------------------------------------
api = FastAPI(title="R0‑Agent API", version="0.1.0")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


# helper: run LangGraph synchronously in a threadpool
async def run_agent(prompt: str):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, agent_app.invoke, {"text": prompt})


# ── 3. /chat endpoint ---------------------------------------------------
@api.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest):              # ← body is ChatRequest
    state = await run_agent(payload.message)
    result   = str(state.get("result", ""))
    recalled = state.get("recalled", [])

    if payload.stream:
        async def token_gen() -> AsyncIterator[str]:
            for ch in result:                      # naive char‑stream
                yield f"data: {ch}\n\n"
                await asyncio.sleep(0)
            yield "event: done\ndata: [DONE]\n\n"

        return StreamingResponse(token_gen(),
                                 media_type="text/event-stream")

    return ChatResponse(result=result, recalled=recalled)


# ── 4. Local dev runner -------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:api", host="0.0.0.0", port=8000, reload=True)
