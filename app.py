from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scanner import build_signals, trade_plan, order_block_zones

ROOT = Path(__file__).parent
DATA_FILE = ROOT / "data" / "market_data.csv"
REQUIRED = {"date", "symbol", "open", "high", "low", "close", "volume"}

st.set_page_config(page_title="India NSE Momentum Dashboard", page_icon="📈", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.1rem;max-width:1600px}
[data-testid="stMetric"]{background:rgba(100,116,139,.09);border:1px solid rgba(148,163,184,.25);padding:10px;border-radius:10px}
.small-muted{color:#94a3b8;font-size:12px}
</style>
""", unsafe_allow_html=True)
st.title("🇮🇳 India NSE Momentum Dashboard")
st.caption("Public research dashboard · CSV/EOD data mode · No broker login · No order execution")
st.info("Data mode: offline historical OHLCV. No Angel One or Zerodha credentials are used. Live/delayed feeds are disabled until a provider grants the required public-display and redistribution rights.")

@st.cache_data(show_spinner=False)
def read_csv_bytes(raw: bytes):
    import io
    return pd.read_csv(io.BytesIO(raw))

@st.cache_data(show_spinner=False)
def read_project_file(path: str, mtime: float):
    return pd.read_csv(path)

def normalize_data(frame):
    df = frame.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "symbol", "open", "high", "low", "close", "volume"])
    df = df[(df["close"] > 0) & (df["volume"] >= 0)]
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)

with st.sidebar:
    st.header("Dashboard controls")
    st.subheader("Market data")
    upload = st.file_uploader("Upload historical OHLCV CSV", type=["csv"], help="Required columns: date, symbol, open, high, low, close, volume. Use at least 55 daily bars per stock for more reliable indicators.")
    st.caption("For a shared public dashboard, maintain data/market_data.csv in the project repository only where your data-source terms permit it. A browser upload is private to that visitor's session.")
    st.divider()
    horizon = st.radio("Analysis timeframe", ["Daily / Swing", "Intraday CSV"], index=0)
    min_turnover_cr = st.number_input("Minimum 20-day average turnover (₹ crore)", min_value=0.0, max_value=5000.0, value=10.0, step=5.0)
    min_rvol = st.slider("Minimum relative volume", 0.5, 5.0, 1.2, 0.1)
    st.divider()
    st.subheader("Risk assumptions")
    capital = st.number_input("Illustrative capital (₹)", min_value=10000, value=500000, step=25000)
    risk_pct = st.slider("Risk budget per trade (%)", 0.25, 2.0, 0.75, 0.05)

try:
    if upload is not None:
        raw_df = read_csv_bytes(upload.getvalue())
        data_source_label = f"Visitor upload: {upload.name} (session only)"
    elif DATA_FILE.exists():
        raw_df = read_project_file(str(DATA_FILE), DATA_FILE.stat().st_mtime)
        data_source_label = "Project data file: data/market_data.csv"
    else:
        st.warning("No market-data file is installed yet. Upload a CSV from the sidebar or add data/market_data.csv to the project repository.")
        st.markdown("**CSV template:** `data/market_data_template.csv`")
        st.stop()
    df = normalize_data(raw_df)
except Exception as exc:
    st.error(f"Could not load market data: {exc}")
    st.stop()

if df.empty:
    st.error("The CSV contains no valid OHLCV rows.")
    st.stop()

# Compute 20-session average traded value and momentum indicators per symbol.
results, latest_rows = [], []
for symbol, hist in df.groupby("symbol", sort=False):
    hist = hist.sort_values("date").tail(250).copy()
    if len(hist) < 25:
        continue
    hist["turnover"] = hist["close"] * hist["volume"]
    avg_turnover = float(hist["turnover"].tail(20).mean())
    last = hist.iloc[-1]
    quote = {"symbol": symbol, "instrument_token": 0, "turnover": float(last["turnover"]), "spread_pct": None}
    interval = "day" if horizon == "Daily / Swing" else "5minute"
    sig = build_signals(hist, quote, interval)
    sig["avg_turnover_cr"] = avg_turnover / 1e7
    sig["last_date"] = last["date"].strftime("%Y-%m-%d")
    sig["last_price"] = float(last["close"])
    sig["volume"] = int(last["volume"])
    if avg_turnover / 1e7 >= min_turnover_cr and sig["relative_volume"] >= min_rvol:
        results.append(sig)
    latest_rows.append({"symbol": symbol, "date": last["date"], "close": last["close"], "volume": last["volume"], "avg_turnover_cr": avg_turnover/1e7})

signals = pd.DataFrame(results)
if signals.empty:
    st.warning("No stocks meet the selected liquidity and relative-volume filters. Lower the filters or upload a larger history.")
    st.stop()
signals = signals.sort_values(["score", "avg_turnover_cr"], ascending=False)

last_date = df["date"].max().strftime("%d %b %Y")
a,b,c,d = st.columns(4)
a.metric("Stocks analysed", f"{df.symbol.nunique():,}")
b.metric("Bullish setups", int((signals.direction == "BULLISH").sum()))
c.metric("Bearish setups", int((signals.direction == "BEARISH").sum()))
d.metric("Data through", last_date)
st.caption(f"Source: {data_source_label} · {len(df):,} OHLCV rows · Data is as supplied; verify timestamps and completeness before relying on it.")

scanner_tab, liquidity_tab, analyser_tab, risk_tab = st.tabs(["Momentum scanner", "Liquidity ranking", "Stock analyser", "Risk management"])
with scanner_tab:
    bullish, bearish = st.columns(2)
    with bullish:
        st.subheader("Bullish leaders")
        st.dataframe(signals[signals.direction == "BULLISH"][["symbol","last_price","change_pct","score","relative_volume","avg_turnover_cr","rsi"]].head(30), use_container_width=True, hide_index=True)
    with bearish:
        st.subheader("Bearish leaders")
        st.dataframe(signals[signals.direction == "BEARISH"][["symbol","last_price","change_pct","score","relative_volume","avg_turnover_cr","rsi"]].head(30), use_container_width=True, hide_index=True)
    st.subheader("All qualified setups")
    st.dataframe(signals[["symbol","direction","last_price","change_pct","score","relative_volume","avg_turnover_cr","vwap","rsi","setup"]], use_container_width=True, hide_index=True)

with liquidity_tab:
    liq = pd.DataFrame(latest_rows).sort_values("avg_turnover_cr", ascending=False)
    st.subheader("20-session average traded value")
    st.dataframe(liq[["symbol","close","volume","avg_turnover_cr","date"]].rename(columns={"close":"latest close","date":"latest session"}), use_container_width=True, hide_index=True)
    fig = go.Figure(go.Bar(x=liq.head(25).symbol, y=liq.head(25).avg_turnover_cr))
    fig.update_layout(height=400, xaxis_title="NSE symbol", yaxis_title="Average turnover (₹ crore)")
    st.plotly_chart(fig, use_container_width=True)

with analyser_tab:
    options = sorted(df.symbol.unique().tolist())
    selected = st.selectbox("Select stock", options)
    hist = df[df.symbol == selected].sort_values("date").tail(150).copy()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=hist.date, open=hist.open, high=hist.high, low=hist.low, close=hist.close, name=selected))
    fig.update_layout(height=500, xaxis_rangeslider_visible=False, title=f"{selected} · OHLC")
    st.plotly_chart(fig, use_container_width=True)
    zones = order_block_zones(hist)
    if zones:
        st.subheader("Heuristic order-block zones")
        st.dataframe(pd.DataFrame(zones), use_container_width=True, hide_index=True)
    else:
        st.caption("No heuristic order-block zone detected in the latest lookback window.")
    row = signals[signals.symbol == selected]
    if not row.empty:
        sig = row.iloc[0].to_dict()
        plan = trade_plan(sig, capital, risk_pct)
        if plan:
            st.subheader("Illustrative trade plan")
            x1,x2,x3,x4 = st.columns(4)
            x1.metric("Entry reference", f"₹{plan['entry']:.2f}")
            x2.metric("Stop reference", f"₹{plan['stop']:.2f}")
            x3.metric("Target 1", f"₹{plan['target1']:.2f}")
            x4.metric("Target 2", f"₹{plan['target2']:.2f}")
            st.write(f"Indicative quantity: **{plan['quantity']:,} shares** · Modelled risk: **₹{plan['risk_amount']:,.0f}**")
        else:
            st.info("No directional setup for this stock under the current filters.")
    else:
        st.info("This stock is not in the qualified scanner list. It may still be viewed above.")

with risk_tab:
    risk_budget = capital * risk_pct / 100
    st.metric("Maximum planned risk per trade", f"₹{risk_budget:,.0f}")
    st.write("The quantity shown in Stock analyser is based on the indicator-derived stop distance and this risk budget. It does not account for slippage, brokerage, taxes, liquidity gaps or overnight gaps.")
    st.warning("Research and education only. Signals, order-block zones, entry/stop/target levels and position sizes are model outputs—not investment advice or guaranteed outcomes. Do not place trades solely on this dashboard.")

st.divider()
st.caption("No broker account is connected. No live feed, market orders, user portfolio, or automated execution is enabled. For a public live-data product, obtain written data-display/redistribution permissions from the exchange or an authorised vendor before enabling a feed.")
