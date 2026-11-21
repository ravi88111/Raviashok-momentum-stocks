import pandas as pd
from pathlib import Path

MASTER_COLS = ["symbol","company","exchange","isin","series","segment","status"]
PRICE_COLS = ["date","symbol","exchange","open","high","low","close","volume"]

def _read(upload, path):
    if upload is not None:
        return pd.read_csv(upload)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

def load_security_master(upload, path):
    df = _read(upload, path)
    if df.empty:
        return df
    missing = set(MASTER_COLS) - set(df.columns)
    if missing:
        raise ValueError("Security master missing columns: " + ", ".join(sorted(missing)))
    for c in ["symbol","exchange","segment","status"]:
        df[c] = df[c].astype(str).str.strip().str.upper()
    df["company"] = df["company"].fillna(df["symbol"])
    df = df[df["exchange"].isin(["NSE","BSE"])]
    df = df[df["status"].eq("ACTIVE")]
    return df

def load_market_data(upload, path):
    df = _read(upload, path)
    if df.empty:
        return df
    missing = set(PRICE_COLS) - set(df.columns)
    if missing:
        raise ValueError("Daily data missing columns: " + ", ".join(sorted(missing)))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["symbol","exchange"]:
        df[c] = df[c].astype(str).str.strip().str.upper()
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=PRICE_COLS)
    df = df[df["exchange"].isin(["NSE","BSE"])]
    df = df[(df[["open","high","low","close"]] > 0).all(axis=1) & (df["volume"] >= 0)]
    df = df.sort_values(["exchange","symbol","date"]).drop_duplicates(["exchange","symbol","date"], keep="last")
    return df
