"""Общие пути и хелпер Spark-сессии для всех этапов пайплайна.

Проектный корень смонтирован в /opt/app во всех Spark-контейнерах,
поэтому сырьё читается из локального маунта, а хранится/обрабатывается в HDFS.
"""
#HDFS
HDFS = "hdfs://namenode:9000"
NAMENODE_JMX = ("http://namenode:9870/jmx"
                "?qry=Hadoop:service=NameNode,name=NameNodeInfo")
RAW = f"{HDFS}/data/raw"            # сырьё после ingest
FEATURES = f"{HDFS}/data/features"  # клиентская витрина фич
PREDICTIONS = f"{HDFS}/data/predictions"

RAW_TRAIN_DATA = f"{RAW}/train_data"
RAW_TEST_DATA = f"{RAW}/test_data"
RAW_TRAIN_TARGET = f"{RAW}/train_target"
RAW_TEST_TARGET = f"{RAW}/test_target"

#  локальный маунт с исходниками 
LOCAL = "/opt/app"
SRC_TRAIN_DATA = f"{LOCAL}/train_data"
SRC_TEST_DATA = f"{LOCAL}/test_data"
SRC_TRAIN_TARGET = f"{LOCAL}/train_target.csv"
SRC_TEST_TARGET = f"{LOCAL}/test_target.csv"

# каталог отчётов (на маунте, виден с хоста)
REPORTS = f"{LOCAL}/reports"

TARGET = "flag"
KEY = "id"
SEQ = "rn"


def get_spark(app_name: str):
    from pyspark.sql import SparkSession
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate()
    )


def wait_safemode_off(timeout: int = 300, interval: int = 5) -> None:
    """Дождаться выхода NameNode из safe mode.

    После рестарта Docker namenode стартует в safe mode (ждёт block report от
    datanode). Любая запись с mode=overwrite в этот момент падает с
    SafeModeException, роняя пайплайн. Статус берём из JMX: атрибут Safemode
    пустой, когда режим выключен.
    """
    import json
    import time
    import urllib.request

    deadline = time.time() + timeout
    while True:
        try:
            with urllib.request.urlopen(NAMENODE_JMX, timeout=10) as resp:
                beans = json.load(resp).get("beans") or [{}]
            status = beans[0].get("Safemode", "")
        except Exception as e:  # namenode ещё поднимается
            status = f"namenode недоступен: {e}"

        if not status:
            return
        if time.time() >= deadline:
            raise RuntimeError(f"NameNode не вышел из safe mode за {timeout}s: {status}")
        print(f"    ждём выхода из safe mode: {status}")
        time.sleep(interval)
