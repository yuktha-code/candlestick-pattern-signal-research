# OHLC Candlestick Recognition & Next-Session Backtest

**Independent Python pattern-recognition research code.** Creates per-candle labels for **13 candlestick patterns**: doji, bullish/bearish marubozu, hammer, shooting star, bullish/bearish engulfing, bullish/bearish harami, morning/evening star, three white soldiers, three black crows. Each row includes the complete list of matching flags and a consensus directional signal.

## Research controls

- Validates OHLC consistency, finite prices, dates and duplicates before detection.
- Uses only **current and preceding** candles to label patterns; the signal is formed at bar `t` close.
- Positions enter at **next session open** and exit at next session close; no assumption that the completed candle was tradable retroactively.
- Deducts round-trip illustrative commissions + slippage (default 12bp per active session).
- Fixed 65%/35% chronological train/test split; reports costs, active sessions, conditional average net return for each pattern and equity/drawdown separately. The split is for descriptive holdout analysis; there is no supervised learning or statistical significance test.
- Does not optimize parameters to make the demo profitable.

## Run

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.demo
```

Outputs: labeled data `results/per_bar_pattern_labels_and_signals.csv`, `results/per_pattern_next_session.csv`, `results/holdout_backtest_summary.csv`, and `figures/synthetic_candlestick_backtest.png`.

## Scope and limits

**The input OHLC dataset is entirely simulated.** Synthetic strategies are not evidence of real investment returns or live predictive power. Pattern definitions differ across sources and charting tools. Some patterns (morning/evening star) use liquid-market adaptations without traditional gaps. No trend-regime filter, multiple-comparison correction, short-borrow availability, session calendars, exchange margin, contract rollover, order-book fills, market data feed, live execution, or capital/margin-based portfolio return normalization is provided. The forced stop at each next session close may not suit every trading market. Strategy-level costs are illustrative, not exchange quotations.

## Attribution

All logic in this repository is independently authored and tested; no third-party GitHub sources copied. Studied ideas from `z-waterking/candlestick-pattern-detector` and `zhirodadkhah/CandleKit`; consult their licenses before copying any code. This implementation uses newly defined rules with documented trade-offs.


## Research upgrade (October 2026)

The accompanying `SKILLS_DEMONSTRATED.md` and `REAL_DATA_WORKFLOW.md` explain tested capabilities and how to supply genuine data. See added optional import/analysis modules in `src/`; synthetic runs are not real-market out-of-sample results.
