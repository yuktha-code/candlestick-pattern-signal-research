"""Contract-independent OHLC CSV ingestion with transparent provenance metadata."""
import hashlib
from pathlib import Path
import pandas as pd
from .patterns import validate_ohlc

def import_ohlc_csv(path:str, date_col:str='date')->tuple[pd.DataFrame,dict]:
    p=Path(path)
    if not p.exists():raise FileNotFoundError(path)
    raw=pd.read_csv(p)
    names={c.lower():c for c in raw.columns}
    if date_col.lower() not in names:
        raise ValueError('Missing date column')
    ren={names[k]:k for k in ('open','high','low','close') if k in names}
    ren[names[date_col.lower()]]='date'
    standardized=raw.rename(columns=ren)
    df=validate_ohlc(standardized)
    return df, {'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rows':len(df),
                'origin':'user-supplied CSV; consult vendor/license for source authenticity'}
