"""Confidence intervals for descriptive signals, controlling many hypothesis tests.

Statistical caveat: overlapping candle patterns create cross-sectional and serial
correlation; naive iid p-values are exploratory only. Use block-bootstrap or
HAC inference for serious tradeable-return claims.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from .backtest import conditional_pattern_study

def fdr_benjamini_hochberg(pvalues:list[float])->list[float]:
    a=np.asarray(pvalues,dtype=float)
    if len(a)==0:return []
    if not np.isfinite(a).all() or ((a<0)|(a>1)).any():raise ValueError('bad p-values')
    order=np.argsort(a)
    adjusted=np.minimum.accumulate((a[order]*len(a)/np.arange(1,len(a)+1))[::-1])[::-1]
    out=np.empty(len(a));out[order]=np.minimum(1,adjusted)
    return out.tolist()

def exploratory_pattern_tests(evaluated:pd.DataFrame,holdout_only=True,cost_bp=12)->pd.DataFrame:
    """Two-sided ONE-SAMPLE t-test on realized next-day simple signal returns.

    Hypothesis means zero return. The FDR correction addresses many patterns
    under independence/positive-dependence assumptions; serial dependence and
    pattern clustering are NOT resolved, so interpret as descriptive screening.
    """
    flags=[x for x in evaluated.columns if x in ('bullish_marubozu','bearish_marubozu','hammer','shooting_star',
         'bullish_engulfing','bearish_engulfing','bullish_harami','bearish_harami','morning_star','evening_star',
         'three_white_soldiers','three_black_crows')]
    bullish={'bullish_marubozu','hammer','bullish_engulfing','bullish_harami','morning_star','three_white_soldiers'}
    rows=[]
    for f in flags:
        selector=evaluated[f].shift(1,fill_value=False)
        if holdout_only:selector &= evaluated['split'].eq('test')
        direction=1 if f in bullish else -1
        r=(direction*evaluated.loc[selector,'intraday_return']-cost_bp/10000).to_numpy(dtype=float)
        p=float(stats.ttest_1samp(r,0).pvalue) if len(r)>=3 and r.std(ddof=1)>0 else 1.0
        rows.append({'pattern':f,'observations':len(r),'mean_net_bp':round(float(r.mean()*1e4),3) if len(r) else np.nan,'exploratory_iid_p':p})
    out=pd.DataFrame(rows)
    out['bh_fdr_q']=fdr_benjamini_hochberg(out.exploratory_iid_p.tolist())
    return out
