"""Data repository: load/save CSV."""
import os
import pandas as pd


class DataRepository:
    ISSUE_COL = "期数"
    NUMBER_PREFIX = "红球_"

    def load(self, path: str, sort_desc: bool = True) -> pd.DataFrame:
        df = pd.read_csv(path, encoding="utf-8")
        if self.ISSUE_COL not in df.columns and "期号" in df.columns:
            df = df.rename(columns={"期号": self.ISSUE_COL})
        if sort_desc and self.ISSUE_COL in df.columns:
            df = df.sort_values(self.ISSUE_COL, ascending=False).reset_index(drop=True)
        return df

    def save(self, df: pd.DataFrame, path: str, sort_desc: bool = True) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        out = df.copy()
        if sort_desc and self.ISSUE_COL in out.columns:
            out = out.sort_values(self.ISSUE_COL, ascending=False).reset_index(drop=True)
        out.to_csv(path, index=False, encoding="utf-8")
