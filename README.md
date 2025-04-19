
# R0 − LangGraph Medium‑Article Agent

![LangGraph](https://img.shields.io/badge/LangGraph-v0.1-blueviolet) ![License](https://img.shields.io/badge/license-MIT-green)

**R0** is a minimalist example of an *AI agent* built with **[LangGraph](https://github.com/langchain-ai/langgraph)**.
It reads an article, then in three autonomous steps:

1. **Classifies** the text (Blog / News / Research / Other)  
2. **Extracts entities** (Person, Organisation, Location)  
3. **Summarises** the article in one short sentence  

No external databases, vector stores, or toolchains—just pure LangGraph
nodes wired into a deterministic flow.

---

## 📂 Project structure

```text
ai_agent_project/
├─ .env.example            # template for secrets
├─ README.md               # you are here
├─ test_setup.py           # hello‑LLM sanity check
└─src/
   ├─ agent_state.py      # Memory: TypedDict schema for the running state
   ├─ nodes.py            # Brain: three capability functions
   └─ agent_graph.py      # Scopes: LangGraph wiring + `analyze()` helper

```

---

## 🚀 Quick‑start

```bash
# 1. clone & enter
git clone https://github.com/NuttakitDW/ai-agent-poc.git && cd ai-agent-poc.git

# 2. python env
python3 -m venv agent_env
source agent_env/bin/activate  # Windows: agent_env\Scripts\activate

# 3. install deps
pip install -r requirements.txt

# 4. add your OpenAI key
echo "OPENAI_API_KEY=sk-..." > .env

# 5. run a single analysis
python - <<'PY'
from src.agent_graph import analyze
print(analyze("LangGraph makes agent flows easy."))
PY
```

---

## 🛠  Developer commands

| Action | Command |
|--------|---------|
| Run full agent loop on custom text | `python -m src.agent_graph` |
| Offline unit tests                | `PYTHONPATH=$(pwd) pytest -q` |
| Pre‑commit cache clean            | `git rm -r --cached **/__pycache__/` |

---

## 🔑 Environment variables

| Name | Required | Description |
|------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | Secret key for GPT‑4o‑mini or other Chat Completions model |

Copy `.env.example` → `.env` and fill in your key.

---

## 🧪 Tests

`tests/` uses **pytest**.  
The default fixture swaps the real LLM for a local `EchoLLM`, so tests run
offline and free:

```bash
pytest -q
```

Set `CI_REAL_KEYS=true` to hit the real model in CI.

---

## 📝 License

MIT — see `LICENSE`.

---

## 🙏 Credits

Based on *“The Complete Guide to Building Your First AI Agent with
LangGraph”* by Paolo Perrone (Mar 2025, Data Science Collective).
https://medium.com/data-science-collective/the-complete-guide-to-building-your-first-ai-agent-its-easier-than-you-think-c87f376c84b2
