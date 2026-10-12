# POWELL AI TRADER V2 — research prototype

This is a **historical simulation**, NOT a production or live-trading bot. The Alpaca connector is read-only and only checks a PAPER account; no orders are sent. No historical data are bundled, and no profitability is claimed.

## Install
Requires Python 3.11+:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run dashboard.py
```

Choose QQQ or SPY, simulated capital and risk settings, then click Run. Internet required to fetch Yahoo Finance data; Yahoo 5-minute history is limited and may be unavailable. Alternatively upload your own 5-minute CSV containing UTC `datetime,open,high,low,close,volume` columns. Timestamp should be the start of each bar. Data must be adjusted consistently and include regular NYSE sessions.

## Strategy and ML
* Session uses America/New_York (DST-aware); 09:30–09:45 ET opening range, signal window 09:45–11:00 ET.
* Daily features use only the first three 5-minute bars (including volume_ratio, which currently equals 1 and therefore contributes no information). To avoid accidental use of V2 as a validated model, treat all reported results as exploratory.
* Walk-forward HistGradientBoostingClassifier retrains on prior sessions; no random train/test split. Label is whether the subsequent close is above the opening midpoint. A confidence threshold gates trades.
* Enters after opening-range breakout; stop at opposite opening range, target at 2R, exits by 11:00 ET. When stop and target hit in same bar, stop wins. Fees approximated in bps; slippage, borrow fees, latency, spread and partial fills not modelled.
* One trade per day, long or short; position sizing limits nominal exposure to account equity. **No implemented daily loss kill switch, broker execution, or continuously running scheduler.**

## Optional read-only paper connection
Set `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` in your shell (use paper keys, never paste credentials into chat), then run `python paper.py`. This prints paper account balances; it does not execute orders.

## Important
Backtests are not guarantees. Yahoo's short intraday window gives far too few sessions for robust ML. Model training should be evaluated with a much longer out-of-sample dataset and trading costs before considering any execution. Do not trade funds needed for living expenses.
