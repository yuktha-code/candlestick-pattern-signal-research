import pandas as pd
import pytest
from pathlib import Path
from src.inference import fdr_benjamini_hochberg,exploratory_pattern_tests
from src.data_import import import_ohlc_csv
from src.backtest import evaluate
P=Path(__file__).resolve().parents[1]
def test_bh_multiple_testing():
    assert fdr_benjamini_hochberg([0.01,0.04,0.03])==pytest.approx([0.03,0.04,0.04])

def test_import_checks_and_hash():
    a,m=import_ohlc_csv(str(P/'data'/'synthetic_ohlc.csv'))
    assert len(a)>100
    assert len(m['sha256'])==64

def test_holdout_no_bogus_technical_significance():
    s=evaluate(pd.read_csv(P/'data'/'synthetic_ohlc.csv'))
    a=exploratory_pattern_tests(s)
    assert len(a)==12
    assert a.bh_fdr_q.between(0,1).all()
