# SKILLS DEMONSTRATED — Candlestick Pattern Detection & Out-of-Sample Signals

## Languages, libraries, quantitative methods and tools

Python; Pandas/NumPy; OHLC validation; 13 deterministic candlestick pattern flags; shift/lag logic; current/previous-candle-only rules; intraday open-to-close backtesting; transaction fees/slippage; train/holdout split; descriptive hit rates; scipy t-tests; Benjamini-Hochberg false-discovery-rate adjustments; Matplotlib; pytest; CSV SHA256 fingerprints.

## Verified implementation vs unverified claims

13 rule-based flags on synthetic OHLC sample; 12 directional hypotheses with exploratory FDR q-values; validation and next-session lagged execution; 8 local tests. No actual exchange data, no demonstrated significant alpha, no robust block-bootstrap/HAC inference or margin accounting.

## Interview questions you should be able to answer

Define hammer versus engulfing versus harami; explain nonstandard star pattern gap assumption, overlap of signals, why enter next open not current close, why multiple comparisons matter and why t-tests are exploratory.

## How to verify

`python -m pip install -r requirements.txt` then `python -m pytest -q` and `python -m src.demo`. Offline fixtures are **synthetic** and intentionally labeled.
