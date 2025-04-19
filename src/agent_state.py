from typing import TypedDict, List

class State(TypedDict):
    """
    The working memory that flows through the LangGraph.

    text            – the raw article the user feeds in
    classification  – e.g. 'News', 'Blog', 'Research', or 'Other'
    entities        – list of extracted named entities
    summary         – one‑sentence synopsis
    """
    text: str
    classification: str
    entities: List[str]
    summary: str
