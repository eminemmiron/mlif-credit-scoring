"""(baseline). Логистическая регрессия Spark MLlib на клиентской витрине из HDFS.

Сравнение «чистый Spark» против бустинга. Дисбаланс - через балансирующие веса
(weightCol). Метрики (ROC-AUC/PR-AUC/Gini) на отложенной части пишутся в
reports/model/baseline_metrics.json.

Запуск:
    spark-submit --master spark://spark-master:7077 src/baseline_mllib.py
"""
import json
import os

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import functions as F

import config as C

OUT = f"{C.REPORTS}/model"


def main():
    spark = C.get_spark("baseline_mllib")
    spark.sparkContext.setLogLevel("WARN")
    os.makedirs(OUT, exist_ok=True)

    df = spark.read.parquet(f"{C.FEATURES}/train")
    feat_cols = [c for c in df.columns if c not in (C.KEY, C.TARGET)]

    # балансирующие веса классов
    n = df.count()
    pos = df.filter(F.col(C.TARGET) == 1).count()
    neg = n - pos
    df = df.withColumn(
        "w",
        F.when(F.col(C.TARGET) == 1, n / (2.0 * pos)).otherwise(n / (2.0 * neg)),
    )

    train, valid = df.randomSplit([0.8, 0.2], seed=42)

    pipe = Pipeline(stages=[
        VectorAssembler(inputCols=feat_cols, outputCol="features_raw", handleInvalid="keep"),
        StandardScaler(inputCol="features_raw", outputCol="features", withMean=False),
        LogisticRegression(featuresCol="features", labelCol=C.TARGET,
                           weightCol="w", maxIter=50, regParam=0.01),
    ])
    model = pipe.fit(train)
    pred = model.transform(valid)

    roc = BinaryClassificationEvaluator(labelCol=C.TARGET, rawPredictionCol="rawPrediction",
                                        metricName="areaUnderROC").evaluate(pred)
    pr = BinaryClassificationEvaluator(labelCol=C.TARGET, rawPredictionCol="rawPrediction",
                                       metricName="areaUnderPR").evaluate(pred)

    metrics = {
        "model": "spark_mllib_logreg",
        "roc_auc": round(roc, 5),
        "gini": round(2 * roc - 1, 5),
        "pr_auc": round(pr, 5),
        "valid_rows": valid.count(),
        "train_pos": pos,
    }
    with open(f"{OUT}/baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(">>> BASELINE MLlib:", metrics)
    spark.stop()


if __name__ == "__main__":
    main()
