import pandas as pd

# helper used by analyzeMarket
def _last_cross(series_fast: pd.Series, series_slow: pd.Series):
    """Return 'bull', 'bear', or None for latest SMA crossover."""
    prev_fast, prev_slow = series_fast.iloc[-2], series_slow.iloc[-2]
    curr_fast, curr_slow = series_fast.iloc[-1], series_slow.iloc[-1]
    if prev_fast < prev_slow and curr_fast > curr_slow:
        return "bull"
    if prev_fast > prev_slow and curr_fast < curr_slow:
        return "bear"
    return None
