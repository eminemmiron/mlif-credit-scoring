"""Память-эффективная загрузка витрины фич в pandas.

Каталог parquet читается по одному part-файлу: каждая часть кастуется в Arrow
(float64->float32, int64->int32, кроме id/flag) ещё до перевода в pandas, поэтому
пик по памяти держится низким. Итог - один float32 DataFrame ~2 ГБ на 3M×180.
"""
import glob

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

KEEP_INT64 = {"id", "flag"}


def _downcast_table(t: pa.Table) -> pa.Table:
    cols = []
    for field in t.schema:
        col = t[field.name]
        if pa.types.is_floating(field.type):
            col = col.cast(pa.float32())
        elif pa.types.is_integer(field.type) and field.name not in KEEP_INT64:
            col = col.cast(pa.int32())
        cols.append(col)
    return pa.table(cols, names=t.schema.names)


def load_features(path: str) -> pd.DataFrame:
    """path - локальный каталог parquet (например /opt/app/data/train_features)."""
    files = sorted(glob.glob(f"{path}/*.parquet"))
    if not files:
        raise FileNotFoundError(f"нет parquet-частей в {path}")
    frames = [_downcast_table(pq.read_table(f)).to_pandas() for f in files]
    return pd.concat(frames, ignore_index=True)
