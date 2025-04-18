# src/config.py
import json, os
from pathlib import Path
from dotenv import load_dotenv # type: ignore

load_dotenv()  # loads .env into os.environ

ROOT = Path(__file__).resolve().parents[1]

# --- Env keys --------------------------------------------------------------
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
ROOSTOO_KEY      = os.getenv("ROOSTOO_KEY")
ROOSTOO_SECRET   = os.getenv("ROOSTOO_SECRET")

if not all([OPENAI_API_KEY, ROOSTOO_KEY, ROOSTOO_SECRET]):
    raise RuntimeError("Missing one or more env vars in .env")

# --- Rules loader ----------------------------------------------------------
def load_rules(path: str | Path = ROOT / "rules.json") -> dict:
    """Load user‑config trading rules from JSON."""
    with open(path, "r", encoding="utf‑8") as f:
        return json.load(f)
