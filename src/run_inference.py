from pathlib import Path
import pandas as pd
from .backtest import evaluate
from .inference import exploratory_pattern_tests
ROOT=Path(__file__).resolve().parents[1]
def main():
    data=pd.read_csv(ROOT/'data'/'synthetic_ohlc.csv')
    scored=evaluate(data)
    results=exploratory_pattern_tests(scored)
    results.to_csv(ROOT/'results'/'exploratory_pattern_tests_SYNTHETIC.csv',index=False)
    print(results[['pattern','observations','mean_net_bp','bh_fdr_q']].to_string(index=False))
if __name__=='__main__':main()
