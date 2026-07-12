"""Ingest: сырьё (parquet-история + csv-таргеты) с локального маунта в HDFS.

Запуск (внутри контейнера spark-master):
    spark-submit --master spark://spark-master:7077 src/ingest_to_hdfs.py
"""
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StructType, StructField

import config as C


def ingest_data(spark, src_glob: str, dst: str, n_parts: int) -> int:
    df = spark.read.parquet(src_glob)
    (df.repartition(n_parts).write.mode("overwrite").parquet(dst))
    return df.count()


def ingest_target(spark, src_csv: str, dst: str, with_flag: bool) -> int:
    fields = [StructField(C.KEY, IntegerType(), False)]
    if with_flag:
        fields.append(StructField(C.TARGET, IntegerType(), False))
    schema = StructType(fields)
    df = spark.read.csv(src_csv, header=True, schema=schema)
    df.write.mode("overwrite").parquet(dst)
    return df.count()


def main():
    spark = C.get_spark("ingest_to_hdfs")
    spark.sparkContext.setLogLevel("WARN")

    # После рестарта Docker namenode стартует в safe mode -> overwrite упадёт.
    C.wait_safemode_off()

    print(">>> train_data ...")
    n = ingest_data(spark, f"{C.SRC_TRAIN_DATA}/*.pq", C.RAW_TRAIN_DATA, n_parts=24)
    print(f"    train_data rows = {n:,}")

    print(">>> test_data ...")
    n = ingest_data(spark, f"{C.SRC_TEST_DATA}/*.pq", C.RAW_TEST_DATA, n_parts=8)
    print(f"    test_data rows = {n:,}")

    print(">>> train_target ...")
    n = ingest_target(spark, C.SRC_TRAIN_TARGET, C.RAW_TRAIN_TARGET, with_flag=True)
    print(f"    train_target rows = {n:,}")

    print(">>> test_target ...")
    n = ingest_target(spark, C.SRC_TEST_TARGET, C.RAW_TEST_TARGET, with_flag=False)
    print(f"    test_target rows = {n:,}")

    # sanity: баланс таргета
    tgt = spark.read.parquet(C.RAW_TRAIN_TARGET)
    dist = tgt.groupBy(C.TARGET).count().orderBy(C.TARGET).collect()
    total = sum(r["count"] for r in dist)
    for r in dist:
        print(f"    flag={r[C.TARGET]}: {r['count']:,} ({100*r['count']/total:.2f}%)")

    print(">>> INGEST DONE")
    spark.stop()


if __name__ == "__main__":
    main()
