"""EDA через Spark SQL: баланс, длина истории, сигналы просрочки, срез фич по классам.

Тяжёлые агрегации считаются в Spark (SQL), маленькие результаты собираются на драйвер
и рисуются matplotlib в reports/eda/. Итоговая сводка - reports/eda/eda_summary.json.

Запуск:
    spark-submit --master spark://spark-master:7077 src/eda_spark.py
"""
import json
import os

from pyspark.sql import functions as F

import config as C
import viz
from viz import plt, PALETTE

EDA_DIR = f"{C.REPORTS}/eda"

# информативные (по смыслу имён) признаки для среза по классам
FEATURE_COLS = [
    "pre_loans530", "pre_loans3060", "pre_loans6090", "pre_loans90",
    "pre_loans_total_overdue", "pre_loans_outstanding",
    "pre_util", "pre_over2limit", "pre_loans_credit_limit",
    "enc_paym_24", "pclose_flag", "fclose_flag",
]


def main():
    spark = C.get_spark("eda")
    spark.sparkContext.setLogLevel("WARN")
    os.makedirs(EDA_DIR, exist_ok=True)
    summary = {}

    data = spark.read.parquet(C.RAW_TRAIN_DATA)
    tgt = spark.read.parquet(C.RAW_TRAIN_TARGET)
    data.createOrReplaceTempView("train_data")
    tgt.createOrReplaceTempView("train_target")

    # 1) Баланс таргета
    bal = spark.sql("""
        SELECT flag, COUNT(*) AS n
        FROM train_target GROUP BY flag ORDER BY flag
    """).collect()
    total = sum(r["n"] for r in bal)
    summary["target_balance"] = {int(r["flag"]): r["n"] for r in bal}
    summary["default_rate"] = round(summary["target_balance"].get(1, 0) / total, 4)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["не дефолт (0)", "дефолт (1)"], [r["n"] for r in bal],
           color=[PALETTE["neg"], PALETTE["pos"]])
    for i, r in enumerate(bal):
        ax.text(i, r["n"], f"{r['n']:,}\n{100*r['n']/total:.2f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_title("Баланс классов (train_target)")
    ax.set_ylabel("клиентов")
    ax.margins(y=0.15)
    viz.save(fig, f"{EDA_DIR}/01_target_balance.png")

    # 2) Длина кредитной истории на клиента
    hist = spark.sql("""
        SELECT id, COUNT(*) AS rn_cnt
        FROM train_data GROUP BY id
    """)
    hist.createOrReplaceTempView("hist")
    stats = hist.selectExpr(
        "min(rn_cnt) mn", "max(rn_cnt) mx", "avg(rn_cnt) avg",
        "percentile_approx(rn_cnt, 0.5) p50", "percentile_approx(rn_cnt, 0.9) p90",
        "percentile_approx(rn_cnt, 0.99) p99",
    ).collect()[0]
    summary["history_len"] = {k: (round(v, 2) if isinstance(v, float) else v)
                              for k, v in stats.asDict().items()}

    hbins = spark.sql("""
        SELECT LEAST(rn_cnt, 40) AS b, COUNT(*) AS n
        FROM hist GROUP BY LEAST(rn_cnt, 40) ORDER BY b
    """).collect()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([r["b"] for r in hbins], [r["n"] for r in hbins], color=PALETTE["accent"])
    ax.set_title("Распределение длины кредитной истории (число продуктов на клиента, обрез. на 40)")
    ax.set_xlabel("продуктов у клиента (rn)")
    ax.set_ylabel("клиентов")
    viz.save(fig, f"{EDA_DIR}/02_history_length.png")

    # 3) Доля дефолта по длине истории
    dr = spark.sql("""
        SELECT LEAST(h.rn_cnt, 20) AS b,
               AVG(t.flag) AS dr, COUNT(*) AS n
        FROM hist h JOIN train_target t ON h.id = t.id
        GROUP BY LEAST(h.rn_cnt, 20) ORDER BY b
    """).collect()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([r["b"] for r in dr], [100 * r["dr"] for r in dr],
            marker="o", color=PALETTE["pos"])
    ax.axhline(100 * summary["default_rate"], ls="--", color=PALETTE["neg"],
               label=f"средний {100*summary['default_rate']:.2f}%")
    ax.set_title("Доля дефолта в зависимости от длины истории")
    ax.set_xlabel("продуктов у клиента (обрез. на 20)")
    ax.set_ylabel("доля дефолта, %")
    ax.legend()
    viz.save(fig, f"{EDA_DIR}/03_default_rate_by_history.png")

    # 4) Средние значения фич по классам (record-level)
    joined = data.join(tgt, on="id", how="inner")
    aggs = []
    for c in FEATURE_COLS:
        aggs += [F.avg(F.when(F.col("flag") == 0, F.col(c))).alias(f"{c}__0"),
                 F.avg(F.when(F.col("flag") == 1, F.col(c))).alias(f"{c}__1")]
    row = joined.agg(*aggs).collect()[0].asDict()
    feat_means = {c: {"neg": row[f"{c}__0"], "pos": row[f"{c}__1"]} for c in FEATURE_COLS}
    summary["feature_means_by_class"] = {
        c: {"neg": round(v["neg"], 4), "pos": round(v["pos"], 4)}
        for c, v in feat_means.items()
    }

    import numpy as np
    x = np.arange(len(FEATURE_COLS))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 0.2, [feat_means[c]["neg"] for c in FEATURE_COLS], width=0.4,
           label="не дефолт (0)", color=PALETTE["neg"])
    ax.bar(x + 0.2, [feat_means[c]["pos"] for c in FEATURE_COLS], width=0.4,
           label="дефолт (1)", color=PALETTE["pos"])
    ax.set_xticks(x)
    ax.set_xticklabels(FEATURE_COLS, rotation=40, ha="right")
    ax.set_title("Среднее значение признака по классам (уровень записи)")
    ax.legend()
    viz.save(fig, f"{EDA_DIR}/04_feature_means_by_class.png")

    # 5) Распределение кодов признака по классам
    # Признаки - анонимные порядковые КОДЫ, поэтому смотрим не среднее, а сдвиг
    # распределения кодов между классами (P(code | class)).
    def code_dist(col):
        rows = (joined.groupBy("flag", col).count()
                .where(F.col(col).isNotNull()).collect())
        tot0 = sum(r["count"] for r in rows if r["flag"] == 0)
        tot1 = sum(r["count"] for r in rows if r["flag"] == 1)
        d = {}
        for r in rows:
            d.setdefault(r[col], [0.0, 0.0])
            d[r[col]][int(r["flag"])] = r["count"]
        codes = sorted(d)
        neg = [d[c][0] / tot0 for c in codes]
        pos = [d[c][1] / tot1 for c in codes]
        return codes, neg, pos

    dist_cols = ["enc_loans_credit_status", "pre_util"]
    summary["code_distribution"] = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col in zip(axes, dist_cols):
        codes, neg, pos = code_dist(col)
        x = np.arange(len(codes))
        ax.bar(x - 0.2, [100 * v for v in neg], width=0.4, label="не дефолт (0)", color=PALETTE["neg"])
        ax.bar(x + 0.2, [100 * v for v in pos], width=0.4, label="дефолт (1)", color=PALETTE["pos"])
        ax.set_xticks(x); ax.set_xticklabels([str(c) for c in codes], fontsize=8)
        ax.set_title(f"Распределение кодов: {col}")
        ax.set_xlabel("код"); ax.set_ylabel("% записей внутри класса")
        ax.legend()
        summary["code_distribution"][col] = {
            "codes": [int(c) for c in codes],
            "neg_pct": [round(100 * v, 3) for v in neg],
            "pos_pct": [round(100 * v, 3) for v in pos],
        }
    viz.save(fig, f"{EDA_DIR}/05_code_distribution_by_class.png")

    # 6) Пропуски по колонкам
    n_rows = data.count()
    null_counts = data.select([
        F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in data.columns
    ]).collect()[0].asDict()
    nulls = {c: v for c, v in null_counts.items() if v > 0}
    summary["rows_train_data"] = n_rows
    summary["columns"] = len(data.columns)
    summary["nulls"] = nulls or "нет пропусков"
    summary["note"] = (
        "Признаки анонимны и порядково закодированы (коды, не реальные величины). "
        "Record-level средние по классам почти совпадают - предиктивный сигнал в "
        "распределении/частоте кодов по истории клиента, поэтому основа модели - агрегация rn→id (Э3)."
    )

    with open(f"{EDA_DIR}/eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("    saved", f"{EDA_DIR}/eda_summary.json")
    print(">>> EDA DONE. default_rate =", summary["default_rate"],
          "| history p50/p90 =", summary["history_len"]["p50"], "/", summary["history_len"]["p90"])
    spark.stop()


if __name__ == "__main__":
    main()
