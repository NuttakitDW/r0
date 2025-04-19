# R0 – Roostoo Trading Agent

R0 is an **autonomous trading assistant** built with the LangGraph orchestration framework.  
It reasons with GPT‑4‑o mini (or GPT‑4.1), executes signed HTTP calls to the [Roostoo mock exchange](https://mock-api.roostoo.com), and stores long‑term memories in a **Pinecone** serverless vector index.

---
## ✨ Key Features

| Capability | Implementation | Reference |
|-------------|----------------|-----------|
| Single‑turn reasoning loop | `think → act → memory → END` graph | LangGraph docs[^1] |
| Secure trading tools | Typed wrappers in `src/wrappers.py`; exposed via `src/tools.py` | API spec[^2] |
| LLM brain | `langchain_openai.ChatOpenAI` with tool‑binding | OpenAI model list[^3] |
| Long‑term memory | Pinecone serverless (AWS us‑east‑1) | Pinecone quick‑start[^4] |

---
## 🗂 Project Layout

```
├─ src/
│  ├─ wrappers.py       # low‑level HTTP helpers (HMAC, retries)
│  ├─ tools.py          # LangChain Tool objects + dispatcher
│  ├─ agent_state.py    # TypedDict schema for graph state
│  ├─ nodes.py          # think_node · act_node · memory_node
│  ├─ agent_graph.py    # LangGraph wiring (one‑pass)
│  └─ memory.py         # Pinecone helper (save / retrieve)
└─ README.md            # you are here
```

---
## ⚙ Installation

```bash
python3 -m venv agent_env && source agent_env/bin/activate
pip install -r requirements.txt  # langgraph, langchain-openai, pinecone-docs …
```

Create **.env**:
```
OPENAI_API_KEY=sk‑...
ROOSTOO_KEY=...
ROOSTOO_SECRET=...
PINECONE_API_KEY=...
PINECONE_ENV=us-east-1 
PINECONE_INDEX=r0-memory
OPENAI_MODEL=gpt-4o-mini    # or gpt-4.1
```

---
## 🚀 Running

```bash
python -m src.cli  # example CLI wrapper (or import app from src.agent_graph)
```

Example interaction:
```
> Give me the server time
1745066524030
> What did you just tell me about time?
I just told you the server time was 1745066524030.
```

---
## 🧠 How Memory Works

1. **Save** – `memory_node` stores the assistant’s final answer as a plain string in Pinecone.
2. **Retrieve** – On the next turn, top‑`k` similar snippets are fetched with cosine similarity and injected into the LLM context as **assistant messages**, allowing GPT‑4 to quote them naturally.
3. The `State` schema includes `recalled: List[str]` so LangGraph keeps the memories in the final state.

---
## 🗺 TODO / Roadmap

- [ ] **Short‑term RAM window** – add `ConversationBufferWindowMemory` (k=4) so clarifications don’t hit Pinecone every turn.
- [ ] **Guardrails** – budget limiter node to cap daily order volume & API spend.
- [ ] **Strategy executor** – multi‑tool loop (`getBalance → calc qty → placeOrder`).  Requires a bounded counter to avoid recursion.
- [ ] **Web dashboard** – React front‑end to visualize positions and chat.
- [ ] **CI/CD** – GitHub Actions: lint, pytest, run sample conversation, deploy docs.
- [ ] **Model upgrade switch** – env flag to swap `gpt-4o-mini` ↔ `gpt-4.1` when available.
- [ ] **Vector pruning** – periodic job to collapse redundant memories (embedding‑clustering).

---
## 📚 References

[^1]: LangGraph documentation – state graphs, conditional edges.  <https://python.langchain.com/v0.1/docs/langgraph/>  ([python.langchain.com](https://python.langchain.com/v0.1/docs/langgraph/?utm_source=chatgpt.com))
[^2]: Roostoo Public API spec – signed endpoints.  <https://mock-api.roostoo.com>  ([platform.openai.com](https://platform.openai.com/docs/models/gpt-4o-mini?utm_source=chatgpt.com))
[^3]: OpenAI model IDs & context windows – GPT‑4o, GPT‑4.1.  <https://platform.openai.com/docs/models>  ([docs.pinecone.io](https://docs.pinecone.io/guides/indexes/create-an-index?utm_source=chatgpt.com))
[^4]: Pinecone serverless index quick‑start (AWS us‑east‑1).  <https://docs.pinecone.io/guides/indexes/create-an-index>  ([docs.pinecone.io](https://docs.pinecone.io/guides/indexes/create-an-index?utm_source=chatgpt.com))
[^5]: LangChain testing guide – mocking LLMs.  <https://python.langchain.com/docs/how_to/>  ([python.langchain.com](https://python.langchain.com/docs/how_to/?utm_source=chatgpt.com))

