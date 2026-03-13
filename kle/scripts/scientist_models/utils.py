import numpy as np
import pandas as pd
from .constants import TOTAL


def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    num_cols = [c for c in df.columns if c.startswith('红球')]
    issues = df['期数'].astype(str).tolist()
    draws = df[num_cols].to_numpy(dtype=int)
    return issues, draws


def presence_matrix(hist, window=None):
    h = hist[:window] if window else hist
    P = np.zeros((len(h), TOTAL), dtype=int)
    for t, row in enumerate(h):
        for num in row:
            if 1 <= num <= TOTAL:
                P[t, num - 1] = 1
    return P


def norm(x: np.ndarray) -> np.ndarray:
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn) if mx > mn else np.zeros_like(x)
