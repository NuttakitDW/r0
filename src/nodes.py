"""
LangGraph node functions for our Medium‑article agent.
Each node:
  • accepts the running `state` dict
  • returns a *partial* dict to merge into state
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from typing import Dict, List

load_dotenv() 

# Shared deterministic brain
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── 1. Classification ────────────────────────────────────────────────
def classification_node(state: Dict) -> Dict:
    prompt = PromptTemplate(
        input_variables=["text"],
        template=(
            "Classify the following text as one of:"
            "  News, Blog, Research, Other.\n\nText:\n{text}\n\nCategory:"
        ),
    )
    msg = HumanMessage(content=prompt.format(text=state["text"]))
    category = llm.invoke([msg]).content.strip()
    return {"classification": category}


# ── 2. Entity extraction ─────────────────────────────────────────────
def entity_extraction_node(state: Dict) -> Dict:
    prompt = PromptTemplate(
        input_variables=["text"],
        template=(
            "Extract all Person, Organization, and Location entities "
            "from the text. Return as a comma‑separated list.\n\n{text}\n\nEntities:"
        ),
    )
    msg = HumanMessage(content=prompt.format(text=state["text"]))
    entities: List[str] = llm.invoke([msg]).content.strip().split(", ")
    return {"entities": entities}


# ── 3. Summarisation ────────────────────────────────────────────────
def summarize_node(state: Dict) -> Dict:
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text in one short sentence:\n\n{text}\n\nSummary:",
    )
    msg = HumanMessage(content=prompt.format(text=state["text"]))
    summary = llm.invoke([msg]).content.strip()
    return {"summary": summary}
