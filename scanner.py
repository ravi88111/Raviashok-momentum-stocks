import numpy as np
import pandas as pd

def _ema(s, n): return s.ewm(span=n, adjust=False).mean()
def _rsi(s, n=14):
    d=s.diff(); up=d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn=(-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs=up/dn.replace(0,np.nan)
    return 100-(100/(1+rs))

def build_signals(candles, quote_row, interval):
    df=candles.copy()
    for c in ["open","high","low","close","volume"]: df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.dropna(subset=["close","high","low","volume"])
    close=df.close; vol=df.volume
    e20=_ema(close,20); e50=_ema(close,50)
    rsi=_rsi(close).iloc[-1]
    avgvol=vol.iloc[-21:-1].mean() if len(vol)>21 else vol.iloc[:-1].mean()
    rvol=float(vol.iloc[-1]/avgvol) if avgvol and avgvol>0 else 0
    typical=(df.high+df.low+df.close)/3
    # Intraday VWAP resets at the current trading session; daily swing bars use lookback VWAP.
    if interval != "day" and "date" in df.columns:
        session_dates = pd.to_datetime(df["date"]).dt.date
        current_session = session_dates.iloc[-1]
        session = df.loc[session_dates == current_session]
        session_typical = (session.high + session.low + session.close) / 3
        vwap = float((session_typical * session.volume).sum() / session.volume.sum()) if session.volume.sum() else float(close.iloc[-1])
    else:
        vwap=float((typical*vol).sum()/vol.sum()) if vol.sum() else float(close.iloc[-1])
    prev_close = df.close.shift(1)
    true_range = pd.concat([(df.high-df.low), (df.high-prev_close).abs(), (df.low-prev_close).abs()], axis=1).max(axis=1)
    atr = float(true_range.rolling(14).mean().iloc[-1]) if len(df) >= 14 else float(true_range.mean())
    last=float(close.iloc[-1]); prev=float(close.iloc[-2]) if len(close)>1 else last
    change=(last/prev-1)*100 if prev else 0
    bullish=last>e20.iloc[-1]>e50.iloc[-1] and last>vwap and rsi>52
    bearish=last<e20.iloc[-1]<e50.iloc[-1] and last<vwap and rsi<48
    direction="BULLISH" if bullish else ("BEARISH" if bearish else "NEUTRAL")
    trend=25 if bullish or bearish else 8
    volume_score=min(25,max(0,(rvol-1)*12.5))
    momentum_score=min(25,abs(change)*4)
    rsi_score=15 if ((bullish and 55<=rsi<=72) or (bearish and 28<=rsi<=45)) else 5
    vwap_score=10 if ((bullish and last>vwap) or (bearish and last<vwap)) else 0
    score=int(min(100,trend+volume_score+momentum_score+rsi_score+vwap_score))
    turnover=float(quote_row.get("turnover",0) or 0)/1e7
    spread=quote_row.get("spread_pct")
    return {"symbol":quote_row["symbol"],"instrument_token":quote_row["instrument_token"],
            "last_price":last,"change_pct":change,"direction":direction,"score":score,
            "relative_volume":rvol,"turnover_cr":turnover,"spread_pct":spread,
            "vwap":vwap,"atr":atr,"rsi":float(rsi) if pd.notna(rsi) else 50,
            "setup":"EMA trend + VWAP + volume" if direction!="NEUTRAL" else "No aligned setup"}

def trade_plan(signal, capital, risk_pct):
    if signal["direction"]=="NEUTRAL": return None
    price=float(signal["last_price"])
    # ATR-based risk using recent price volatility proxy; conservative minimum 0.6%.
    risk=max(price*0.006, float(signal.get("atr", price*0.01))*1.2)
    side=1 if signal["direction"]=="BULLISH" else -1
    entry=price
    stop=entry-side*risk
    t1=entry+side*risk*1.5
    t2=entry+side*risk*2.5
    risk_budget=capital*risk_pct/100
    qty=max(0,int(risk_budget/risk))
    return {"entry":entry,"stop":stop,"target1":t1,"target2":t2,"quantity":qty,
            "risk_amount":qty*risk,"rr1":1.5,"rr2":2.5}

def order_block_zones(candles, lookback=50):
    df=candles.tail(lookback).reset_index(drop=True)
    out=[]
    if len(df)<5: return out
    for i in range(2,len(df)-1):
        cur=df.iloc[i]; nxt=df.iloc[i+1]
        rng=max(cur["high"]-cur["low"],1e-9)
        body=abs(nxt["close"]-nxt["open"])
        # Displacement candle after opposite candle; zone is the last opposing candle body.
        if body/rng>0.6 and nxt["close"]>nxt["open"] and cur["close"]<cur["open"]:
            out.append({"type":"Bullish OB","low":float(cur["low"]),"high":float(cur["open"]),"date":str(cur["date"])})
        elif body/rng>0.6 and nxt["close"]<nxt["open"] and cur["close"]>cur["open"]:
            out.append({"type":"Bearish OB","low":float(cur["open"]),"high":float(cur["high"]),"date":str(cur["date"])})
    return out[-5:]

def liquidity_rank(frame):
    return frame.sort_values("turnover",ascending=False)
