"""Airflow DAG: оркестрация пайплайна PD-скоринга.

Каждый этап запускается через docker SDK (exec_run) в уже поднятом контейнере:
  * Spark-этапы  -> spark-master (spark-submit к standalone-мастеру)
  * бустинг      -> trainer (обычный python с lightgbm)

Граф (линейный - одна тяжёлая стадия за раз, т.к. один Spark-воркер и лимит RAM):
    ingest -> eda -> features -> baseline -> train -> evaluate -> predict

Требуется: подняты профили default (HDFS+Spark) и ml (trainer); сокет docker
проброшен в контейнер планировщика.
"""
from __future__ import annotations

import pendulum
from airflow.decorators import task
from airflow.models.dag import DAG

WORKDIR = "/opt/app"
SPARK_MASTER = "spark://spark-master:7077"

SPARK_CONF = ["--conf", "spark.executor.memory=2500m", "--conf", "spark.executor.cores=4"]


def _run(container: str, cmd: list[str]) -> None:
    """Запустить команду в работающем контейнере, пробросить логи, упасть при ec!=0."""
    import docker

    client = docker.from_env()
    ctr = client.containers.get(container)
    ec, out = ctr.exec_run(cmd, workdir=WORKDIR, demux=False)
    text = out.decode("utf-8", "replace") if out else ""
    print(text[-6000:])
    if ec != 0:
        raise RuntimeError(f"[{container}] {' '.join(cmd)} -> exit {ec}")


def _spark(script: str) -> None:
    _run("spark-master", ["/opt/spark/bin/spark-submit", "--master", SPARK_MASTER,
                          *SPARK_CONF, f"src/{script}"])


def _trainer(script: str) -> None:
    _run("trainer", ["python3", f"src/{script}"])


with DAG(
    dag_id="credit_scoring",
    description="PD-скоринг: ingest -> EDA -> features -> модели -> оценка -> submission",
    schedule=None,
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    tags=["credit-scoring", "spark", "lightgbm"],
) as dag:

    @task
    def ingest():
        _spark("ingest_to_hdfs.py")

    @task
    def eda():
        _spark("eda_spark.py")

    @task
    def features():
        _spark("features_spark.py")

    @task
    def baseline():
        _spark("baseline_mllib.py")

    @task
    def train():
        _trainer("train.py")

    @task
    def train_catboost():
        """Вторая boosting-модель для сравнения (та же витрина и сплит)."""
        _trainer("train_catboost.py")

    @task
    def evaluate():
        _trainer("evaluate.py")

    @task
    def predict():
        _trainer("predict.py")

    @task
    def validate():
        """Ворота качества: проверяет submission перед сдачей и роняет DAG,
        если формат/полнота/скоры не в порядке."""
        _trainer("validate_submission.py")

    # Граф линейный - и это осознанно, а не по лени.
    #
    # Пробовали разнести eda/baseline в параллельную ветку с train (Spark и trainer -
    # разные контейнеры, за executor'ы не конкурируют). Замерили: wall-clock 1223 c
    # против 1252 c у линейного, то есть выигрыш 2% - в пределах шума.
    # Причина: у машины 4 ядра, и параллельные задачи просто делят те же ядра.
    # Общий объём вычислений не меняется => wall-clock ~ (вся работа / число ядер).
    # При этом train замедлился на 53%, а baseline втрое.
    #
    # Параллелить имеет смысл на настоящем кластере, где задачи уедут на разные узлы:
    #     ingest >> features
    #     features >> train >> evaluate >> predict
    #     features >> eda >> baseline >> evaluate   # evaluate ждёт baseline:
    #                                               # он читает baseline_metrics.json
    (ingest() >> features() >> eda() >> baseline()
     >> train() >> train_catboost() >> evaluate() >> predict() >> validate())
