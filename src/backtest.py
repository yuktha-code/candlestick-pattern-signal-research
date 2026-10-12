"""End-of-day OHLC detector and next-session open-to-close signal validation."""
import numpy as np
import pandas as pd
from .patterns import recognize


def evaluate(raw:pd.DataFrame,train_fraction=.65, commission_bps=4.,slippage_bps=2.):
    """A pattern confirmed on bar t may first execute at open of bar t+1.

    Each position is intraday t+1 open->close, not a continuous contract hedge.
    To avoid market-open slippage being treated as alpha, subtract round-trip
    commission and slippage for each active session. No parameter calibration.
    """
    if not 0<train_fraction<1 or commission_bps<0 or slippage_bps<0:
        raise ValueError('invalid backtest parameters')
    a=recognize(raw)
    n=len(a)
    if n<20:raise ValueError('need at least 20 bars')
    cutoff=int(n*train_fraction)
    a['next_open_position']=a.signal_at_close.shift(1,fill_value=0)
    a['intraday_return']=a['close']/a['open']-1
    a['gross_return']=a.next_open_position*a.intraday_return
    a['cost_return']=(a.next_open_position!=0).astype(float)*2*(commission_bps+slippage_bps)/10000
    a['net_return']=a.gross_return-a.cost_return
    a['split']=['train' if i<cutoff else 'test' for i in range(n)]
    a['equity_index']=(1+a.net_return).cumprod()
    return a


def summarize(a:pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for part in ('train','test'):
        x=a[a.split==part]
        active=x[x.next_open_position!=0]
        equity=(1+x.net_return).cumprod()
        max_dd=float((equity/equity.cummax()-1).min())
        rows.append({'split':part,'bars':len(x),'active_sessions':len(active),
           'positive_active_pct':round(100*float((active.net_return>0).mean()),2) if len(active) else np.nan,
           'mean_net_return_active_bp':round(float(active.net_return.mean())*10000,3) if len(active) else np.nan,
           'strategy_total_return_pct':round(100*float(equity.iloc[-1]-1),3),
           'max_drawdown_pct':round(100*max_dd,3),
           'always_long_open_close_pct':round(100*float((1+x.intraday_return).prod()-1),3)})
    return pd.DataFrame(rows)


def conditional_pattern_study(a:pd.DataFrame) -> pd.DataFrame:
    """Report next-session net return for each isolated direction *without* selecting winners.

    Every row in each mask is a bar where that named pattern completed; return
    is open-to-close on following day with illustrative roundtrip costs.
    This is descriptive, NOT inference of statistical significance.
    """
    directions={'bullish_marubozu':1,'bearish_marubozu':-1,'hammer':1,'shooting_star':-1,
        'bullish_engulfing':1,'bearish_engulfing':-1,'bullish_harami':1,'bearish_harami':-1,
        'morning_star':1,'evening_star':-1,'three_white_soldiers':1,'three_black_crows':-1}
    # Dataframe 'a' contains next-open positions; fee per round-trip is inferred
    # from strategy rows with a nonzero signal (0 when no trades).
    cost=float(a.loc[a.next_open_position!=0,'cost_return'].iloc[0]) if (a.next_open_position!=0).any() else 0.
    rows=[]
    for key,dirn in directions.items():
        for part in ('train','test'):
            eligible=(a[key].shift(1,fill_value=False))&(a['split']==part)
            ret=dirn*a.loc[eligible,'intraday_return']-cost
            rows.append({'pattern':key,'direction':dirn,'split':part,'signals':int(eligible.sum()),
                         'mean_next_bar_net_bp':round(float(ret.mean())*1e4,3) if len(ret) else np.nan})
    return pd.DataFrame(rows)
