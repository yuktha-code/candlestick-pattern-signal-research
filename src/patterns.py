"""Deterministic and explainable OHLC pattern rules; patterns use no future bars."""
from __future__ import annotations
import numpy as np
import pandas as pd


def validate_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    req=['date','open','high','low','close']
    if any(x not in df for x in req):raise ValueError(f'requires {req}')
    a=df.copy()
    a['date']=pd.to_datetime(a['date'])
    if not a.date.is_monotonic_increasing or a.date.duplicated().any():
        raise ValueError('dates must be ascending and unique')
    for x in req[1:]:a[x]=pd.to_numeric(a[x],errors='raise')
    if not np.isfinite(a[req[1:]].to_numpy()).all() or (a[req[1:]]<=0).any().any():
        raise ValueError('OHLC must be finite and strictly positive')
    if (a.high < a[['open','close']].max(axis=1)).any() or (a.low > a[['open','close']].min(axis=1)).any() or (a.high<a.low).any():
        raise ValueError('OHLC high/low consistency error')
    return a.reset_index(drop=True)


def recognize(raw:pd.DataFrame) -> pd.DataFrame:
    """Create boolean pattern flags for every row; no rolling labels from future rows.

    Futures/liquid-market adaptations: no strict overnight gap necessary for
    three-candle star patterns. Rules are documented and adjustable in source.
    """
    df=validate_ohlc(raw)
    o,h,l,c=[df[x] for x in ('open','high','low','close')]
    rng=(h-l).clip(lower=1e-10)
    body=(c-o).abs();upper=h-pd.concat([o,c],axis=1).max(axis=1)
    lower=pd.concat([o,c],axis=1).min(axis=1)-l
    bull=c>o;bear=c<o
    real_body=body/rng
    small=real_body<=.25
    long=real_body>=.60
    # Pattern masks use only current and previous bars.
    flags={
      'doji':(real_body<=.10),
      'bullish_marubozu':bull & (real_body>=.85) & (upper/rng<=.075)&(lower/rng<=.075),
      'bearish_marubozu':bear & (real_body>=.85) & (upper/rng<=.075)&(lower/rng<=.075),
      'hammer':(lower>=2*body)&(upper<=.30*body)&(body/rng>=.10),
      'shooting_star':(upper>=2*body)&(lower<=.30*body)&(body/rng>=.10),
      'bullish_engulfing':bull & bear.shift(1,fill_value=False)&(o<=c.shift(1))&(c>=o.shift(1))&(body>body.shift(1)),
      'bearish_engulfing':bear & bull.shift(1,fill_value=False)&(o>=c.shift(1))&(c<=o.shift(1))&(body>body.shift(1)),
      'bullish_harami':bull & bear.shift(1,fill_value=False)&(body<=.65*body.shift(1))&(o>=c.shift(1))&(c<=o.shift(1)),
      'bearish_harami':bear & bull.shift(1,fill_value=False)&(body<=.65*body.shift(1))&(o<=c.shift(1))&(c>=o.shift(1)),
      'morning_star':bear.shift(2,fill_value=False)&long.shift(2,fill_value=False)&small.shift(1,fill_value=False)&bull&long&(c>=(o.shift(2)+c.shift(2))/2),
      'evening_star':bull.shift(2,fill_value=False)&long.shift(2,fill_value=False)&small.shift(1,fill_value=False)&bear&long&(c<=(o.shift(2)+c.shift(2))/2),
      'three_white_soldiers':bull&bull.shift(1,fill_value=False)&bull.shift(2,fill_value=False)
          &(c>c.shift(1))&(c.shift(1)>c.shift(2))&(o>=o.shift(1))&(o<=c.shift(1))
          &(o.shift(1)>=o.shift(2))&(o.shift(1)<=c.shift(2)),
      'three_black_crows':bear&bear.shift(1,fill_value=False)&bear.shift(2,fill_value=False)
          &(c<c.shift(1))&(c.shift(1)<c.shift(2))&(o<=o.shift(1))&(o>=c.shift(1))
          &(o.shift(1)<=o.shift(2))&(o.shift(1)>=c.shift(2)),
    }
    out=df.copy()
    for key,v in flags.items():out[key]=v.fillna(False).astype(bool)
    bullish=['bullish_marubozu','hammer','bullish_engulfing','bullish_harami','morning_star','three_white_soldiers']
    bearish=['bearish_marubozu','shooting_star','bearish_engulfing','bearish_harami','evening_star','three_black_crows']
    b=out[bullish].sum(axis=1);s=out[bearish].sum(axis=1)
    out['signal_at_close']=np.sign(b-s).astype(int)
    out['patterns_detected']=out[list(flags)].apply(lambda z: '|'.join(z.index[z.to_numpy(dtype=bool)]),axis=1)
    return out
