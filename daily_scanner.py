import numpy as np
import pandas as pd

def _one_symbol(g):
    g = g.sort_values("date").copy()
    g["ema20"] = g.close.ewm(span=20, adjust=False).mean()
    g["ema50"] = g.close.ewm(span=50, adjust=False).mean()
    g["ema200"] = g.close.ewm(span=200, adjust=False).mean()
    delta = g.close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    g["rsi14"] = 100 - (100/(1+rs))
    g["avg_volume20"] = g.volume.rolling(20).mean()
    g["relative_volume"] = g.volume / g.avg_volume20.replace(0, np.nan)
    g["avg_traded_value20"] = (g.close*g.volume).rolling(20).mean()
    return g.iloc[-1]

def build_daily_scan(df):
    if df.empty:
        return pd.DataFrame()
    rows=[]
    for (exchange,symbol), g in df.groupby(["exchange","symbol"], sort=False):
        if len(g) < 2:
            continue
        rows.append(_one_symbol(g))
    if not rows:
        return pd.DataFrame()
    out=pd.DataFrame(rows)
    out["company"] = out.get("company", out.symbol)
    out["segment"] = out.get("segment", "UNCLASSIFIED")
    out["momentum_score"] = (
        (out.close > out.ema20).astype(int)*20 +
        (out.ema20 > out.ema50).astype(int)*25 +
        (out.ema50 > out.ema200).astype(int)*25 +
        (out.rsi14.between(50,70)).astype(int)*15 +
        (out.relative_volume.fillna(0) > 1.2).astype(int)*15
    )
    out["signal"] = np.select(
        [out.momentum_score >= 70, out.momentum_score <= 30],
        ["Bullish","Bearish"], default="Neutral"
    )
    out["entry_reference"] = out.close
    out["stop_loss_reference"] = out.close * 0.98
    out["target_1_reference"] = out.close * 1.04
    out["target_2_reference"] = out.close * 1.06
    out["risk_reward_target_1"] = 2.0
    cols=["date","symbol","company","exchange","segment","close","ema20","ema50","ema200","rsi14",
          "relative_volume","avg_traded_value20","momentum_score","signal","entry_reference",
          "stop_loss_reference","target_1_reference","target_2_reference"]
    return out[[c for c in cols if c in out.columns]].sort_values("momentum_score",ascending=False)
