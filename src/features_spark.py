"""Feature engineering: агрегация истории кредитных продуктов rn -> id на Spark SQL.

Витрина строится ОДНИМ SQL-запросом. Две части:

1. Базовые агрегаты по каждому из 59 закодированных столбцов - mean / max / last
   (last = значение при максимальном rn через max_by) + длина истории rn_cnt.

2. Поведенческие признаки через ОКОННЫЕ ФУНКЦИИ (демонстрация продвинутого SQL):
   * тренд - средний шаг изменения по истории, `LAG(...) OVER (PARTITION BY id ORDER BY rn)`;
   * свежесть - среднее по последним 3 продуктам, окно `MAX(rn) OVER (PARTITION BY id)`.
   Порядок продуктов (rn) несёт сигнал, который простые агрегаты теряют.

Витрина сохраняется:
  * в HDFS (канонический слой data/features/{train,test})
  * локально в /opt/app/data/{train,test}_features (parquet-каталог для бустинга в pandas)

Запуск:
    spark-submit --master spark://spark-master:7077 src/features_spark.py
"""
from __future__ import annotations

from pyspark.sql import functions as F

import config as C

ID_COLS = {C.KEY, C.SEQ}

# столбцы, по которым считаем поведенческие оконные признаки (топ по важности модели)
TREND_COLS = ["pre_util", "pre_loans_outstanding"]
RECENT_COLS = ["pre_util", "pre_loans_outstanding", "enc_loans_credit_status"]
RECENT_WINDOW = 3  # «последние N продуктов»


def build_sql(view: str, feat_cols: list[str]) -> str:
    """Собрать SQL-запрос витрины для temp-view с сырой историей."""
    base = ",\n        ".join(
        f"AVG({c}) AS {c}_mean, MAX({c}) AS {c}_max, MAX_BY({c}, {C.SEQ}) AS {c}_last"
        for c in feat_cols
    )
    # оконные вычисления на уровне записи (CTE), затем свёртка на уровне клиента
    lag_cols = ",\n        ".join(
        f"{c} - LAG({c}) OVER (PARTITION BY {C.KEY} ORDER BY {C.SEQ}) AS _d_{c}"
        for c in TREND_COLS
    )
    trend_aggs = ",\n        ".join(
        f"COALESCE(AVG(_d_{c}), 0) AS {c}_trend" for c in TREND_COLS
    )
    recent_aggs = ",\n        ".join(
        f"AVG(CASE WHEN {C.SEQ} > _max_rn - {RECENT_WINDOW} "
        f"THEN CAST({c} AS DOUBLE) END) AS {c}_recent{RECENT_WINDOW}"
        for c in RECENT_COLS
    )
    return f"""
    WITH seq AS (
        SELECT *,
        MAX({C.SEQ}) OVER (PARTITION BY {C.KEY}) AS _max_rn,
        {lag_cols}
        FROM {view}
    )
    SELECT
        {C.KEY},
        {base},
        COUNT(*) AS rn_cnt,
        {trend_aggs},
        {recent_aggs}
    FROM seq
    GROUP BY {C.KEY}
    """


def build_features(spark, data):
    feat_cols = [c for c in data.columns if c not in ID_COLS]
    data.createOrReplaceTempView("history")
    client = spark.sql(build_sql("history", feat_cols))
    return client, len(feat_cols)


def write_features(spark, df, hdfs_path, local_path, n_parts):
    """Материализуем витрину ОДИН раз.

    DataFrame ленивый: каждое действие (write/count) пересчитывает весь план.
    Считаем один раз в HDFS, а локальную копию и счётчик строк берём из готового parquet.
    """
    df.repartition(n_parts).write.mode("overwrite").parquet(hdfs_path)
    out = spark.read.parquet(hdfs_path)
    out.write.mode("overwrite").parquet(f"file://{local_path}")
    return out


def main():
    spark = C.get_spark("features")
    spark.sparkContext.setLogLevel("WARN")

    #  train 
    train = spark.read.parquet(C.RAW_TRAIN_DATA)
    tgt = spark.read.parquet(C.RAW_TRAIN_TARGET)
    train_feat, n_feats = build_features(spark, train)
    # broadcast: таргет мал (3 млн × 2 колонки ~24 МБ) -> рассылаем его на воркеры.
    train_feat = train_feat.join(F.broadcast(tgt), on=C.KEY, how="inner")
    out = write_features(spark, train_feat, f"{C.FEATURES}/train",
                         f"{C.LOCAL}/data/train_features", n_parts=8)
    print(f">>> train features: rows={out.count():,} cols={len(out.columns)} "
          f"(из {n_feats} сырых столбцов)")

    #  test (без метки) 
    test = spark.read.parquet(C.RAW_TEST_DATA)
    test_feat, _ = build_features(spark, test)
    out = write_features(spark, test_feat, f"{C.FEATURES}/test",
                         f"{C.LOCAL}/data/test_features", n_parts=4)
    print(f">>> test features:  rows={out.count():,} cols={len(out.columns)}")

    print(">>> FEATURES DONE")
    spark.stop()


if __name__ == "__main__":
    main()
