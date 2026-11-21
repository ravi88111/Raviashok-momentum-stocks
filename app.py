import streamlit as st
import pandas as pd
from pathlib import Path
from scanners.daily_scanner import build_daily_scan
from utils.data_loader import load_market_data, load_security_master

st.set_page_config(page_title="MarketPulse India", page_icon="📈", layout="wide")
st.title("📈 MarketPulse India")
st.caption("NSE + BSE | Active listed equities including SME | Daily momentum research")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

with st.sidebar:
    st.header("Data")
    st.caption("Load authorised market data. This app does not scrape exchange websites or redistribute broker feeds.")
    uploaded_prices = st.file_uploader("Upload daily OHLCV CSV", type=["csv"])
    uploaded_master = st.file_uploader("Upload NSE/BSE security master CSV", type=["csv"])
    st.markdown("---")
    st.caption("Intraday analysis is enabled only when a permitted intraday CSV is supplied.")
    uploaded_intraday = st.file_uploader("Upload authorised intraday candles (optional)", type=["csv"])

try:
    master = load_security_master(uploaded_master, DATA_DIR / "security_master.csv")
except Exception as e:
    master = pd.DataFrame()
    st.error(f"Security master could not be loaded: {e}")

try:
    prices = load_market_data(uploaded_prices, DATA_DIR / "daily_market_data.csv")
except Exception as e:
    prices = pd.DataFrame()
    st.error(f"Daily market data could not be loaded: {e}")

if master.empty:
    st.info("No security master installed. Add data/security_master.csv or upload a CSV in the sidebar.")
    st.code("symbol,company,exchange,isin,series,segment,status\nRELIANCE,Reliance Industries,NSE,INE002A01018,EQ,MAINBOARD,ACTIVE", language="csv")
if prices.empty:
    st.info("No daily market data installed. Add data/daily_market_data.csv or upload an authorised CSV in the sidebar.")
    st.code("date,symbol,exchange,open,high,low,close,volume\n2026-11-20,RELIANCE,NSE,1450,1470,1445,1465,1200000", language="csv")
    st.stop()

if not master.empty:
    prices = prices.merge(master[["symbol","exchange","company","segment","status"]].drop_duplicates(["symbol","exchange"]),
                          on=["symbol","exchange"], how="left")
else:
    prices["company"] = prices["symbol"]
    prices["segment"] = "UNCLASSIFIED"
    prices["status"] = "UNVERIFIED"

st.sidebar.header("Scanner filters")
exchanges = sorted(prices["exchange"].dropna().unique().tolist())
selected_exchanges = st.sidebar.multiselect("Exchange", exchanges, default=exchanges)
segments = sorted(prices["segment"].fillna("UNCLASSIFIED").unique().tolist())
selected_segments = st.sidebar.multiselect("Segment", segments, default=segments)
filtered = prices[prices["exchange"].isin(selected_exchanges) & prices["segment"].fillna("UNCLASSIFIED").isin(selected_segments)]

scan = build_daily_scan(filtered)
if scan.empty:
    st.warning("Not enough valid history to calculate indicators. Provide more daily candles per symbol.")
    st.stop()

latest_date = pd.to_datetime(scan["date"]).max().date()
a,b,c,d = st.columns(4)
a.metric("Symbols analysed", f"{scan['symbol'].nunique():,}")
b.metric("Exchanges", ", ".join(sorted(scan["exchange"].unique())))
c.metric("Latest data date", str(latest_date))
d.metric("Bullish setups", int((scan["signal"]=="Bullish").sum()))

tab1,tab2,tab3,tab4 = st.tabs(["All Stocks Scanner","Bullish Leaders","Bearish Leaders","Data Quality"])
with tab1:
    query = st.text_input("Search symbol or company")
    view = scan.copy()
    if query:
        view = view[view["symbol"].str.contains(query, case=False, na=False) | view["company"].str.contains(query, case=False, na=False)]
    st.dataframe(view.sort_values("momentum_score", ascending=False), use_container_width=True, hide_index=True)
with tab2:
    st.dataframe(scan[scan.signal=="Bullish"].sort_values("momentum_score",ascending=False).head(100), use_container_width=True, hide_index=True)
with tab3:
    st.dataframe(scan[scan.signal=="Bearish"].sort_values("momentum_score").head(100), use_container_width=True, hide_index=True)
with tab4:
    st.write("Daily history coverage")
    coverage = filtered.groupby(["exchange","symbol"]).agg(first_date=("date","min"),last_date=("date","max"),rows=("date","count")).reset_index()
    st.dataframe(coverage, use_container_width=True, hide_index=True)
    if uploaded_intraday is not None:
        intra = pd.read_csv(uploaded_intraday)
        st.write("Intraday file received (not yet connected to a live provider):", len(intra), "rows")
    else:
        st.info("No authorised intraday feed supplied. Intraday signals are intentionally disabled.")

st.caption("Research and educational tool only. Signals are rule-based estimates, not investment advice. Validate exchange status, data rights, liquidity and execution risks independently.")
