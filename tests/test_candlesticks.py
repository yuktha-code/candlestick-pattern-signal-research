import numpy as np
import pandas as pd
import pytest
from src.patterns import validate_ohlc,recognize
from src.backtest import evaluate,conditional_pattern_study,summarize

def bars():
    o=[100,100,101,99,100,100.5,101,101.2,101.4]
    c=[100.001,101,99,100.5,100.2,101,101.3,101.6,102.0]
    return pd.DataFrame({'date':pd.bdate_range('2026-01-01',periods=len(o)),
      'open':o,'high':[max(x,y)+.04 for x,y in zip(o,c)],
      'low':[min(x,y)-.04 for x,y in zip(o,c)],'close':c})

def test_doji_and_marubozu():
    a=recognize(bars())
    assert a.doji.iloc[0]
    assert a.bullish_marubozu.iloc[1]

def test_engulfing():
    d=pd.DataFrame({'date':pd.bdate_range('2026-01-01',periods=3),
      'open':[100,100,98.9],'high':[101,101,101.4],
      'low':[99,98.9,98.5], 'close':[100.5,99,101.2]})
    a=recognize(d)
    assert a.bullish_engulfing.iloc[2]

def test_validation_rejects_impossible_candles():
    d=bars();d.loc[2,'high']=50
    with pytest.raises(ValueError):validate_ohlc(d)

def test_next_open_no_lookahead():
    orig=pd.concat([bars() for _ in range(5)],ignore_index=True)
    orig['date']=pd.bdate_range('2025-01-01',periods=len(orig))
    a=evaluate(orig)
    assert a.next_open_position.iloc[0]==0
    assert (a.next_open_position.iloc[1:].to_numpy()==a.signal_at_close.iloc[:-1].to_numpy()).all()
    edit=orig.copy();edit.loc[30,'close']+=.01;edit.loc[30,'high']+=.01
    b=evaluate(edit)
    assert (a.next_open_position.iloc[:31].to_numpy()==b.next_open_position.iloc[:31].to_numpy()).all()

def test_costs_and_summary():
    orig=pd.concat([bars() for _ in range(6)],ignore_index=True)
    orig['date']=pd.bdate_range('2025-01-01',periods=len(orig))
    a=evaluate(orig)
    assert (a.cost_return>=0).all()
    assert (a.net_return<=a.gross_return).all()
    assert len(summarize(a))==2
    assert len(conditional_pattern_study(a))==24
