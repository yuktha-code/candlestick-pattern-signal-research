from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .backtest import evaluate,summarize,conditional_pattern_study
ROOT=Path(__file__).resolve().parents[1]

def main():
    a=evaluate(pd.read_csv(ROOT/'data'/'synthetic_ohlc.csv'))
    (ROOT/'results').mkdir(exist_ok=True);(ROOT/'figures').mkdir(exist_ok=True)
    a.to_csv(ROOT/'results'/'per_bar_pattern_labels_and_signals.csv',index=False)
    s=summarize(a);s.to_csv(ROOT/'results'/'holdout_backtest_summary.csv',index=False)
    conditional_pattern_study(a).to_csv(ROOT/'results'/'per_pattern_next_session.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,4));ax.plot(a.date,a.equity_index)
    ax.axvline(a.loc[a.split.eq('test').idxmax(),'date'],linestyle='--',color='grey',label='Holdout starts')
    ax.set_title('Synthetic OHLC patterns — cost-adjusted illustrative equity index')
    ax.set_ylabel('Equity index (start 1.0)');ax.legend();fig.tight_layout()
    fig.savefig(ROOT/'figures'/'synthetic_candlestick_backtest.png',dpi=170);plt.close(fig)
    print(s.to_string(index=False));print('Pattern hits:',{c:int(a[c].sum()) for c in ['doji','hammer','bullish_engulfing','evening_star','three_white_soldiers']})
if __name__=='__main__':main()
