"""Оценка бустинга на отложенной выборке: метрики + графики.

Читает valid-предсказания и модель, считает ROC-AUC/Gini/PR-AUC/KS, строит
ROC/PR-кривые, распределение скоров, KS, калибровку, важность признаков,
матрицу ошибок при KS-пороге и сравнение с baseline. Итог - reports/model/.

Запуск (в контейнере trainer):
    python src/evaluate.py
"""
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score, confusion_matrix, precision_recall_curve,
    roc_auc_score, roc_curve,
)

import config as C
import viz
from viz import plt, PALETTE

OUT = f"{C.REPORTS}/model"
MODELS = f"{C.LOCAL}/models"


def main():
    os.makedirs(OUT, exist_ok=True)
    vp = pd.read_parquet(f"{C.LOCAL}/data/valid_pred.parquet")
    y, s = vp["y_true"].to_numpy(), vp["y_score"].to_numpy()

    fpr, tpr, thr = roc_curve(y, s)
    auc = roc_auc_score(y, s)
    pr_auc = average_precision_score(y, s)
    ks_arr = tpr - fpr
    ks_i = int(np.argmax(ks_arr))
    ks = float(ks_arr[ks_i])
    ks_thr = float(thr[ks_i])

    metrics = {
        "model": "lightgbm",
        "roc_auc": round(float(auc), 5),
        "gini": round(2 * float(auc) - 1, 5),
        "pr_auc": round(float(pr_auc), 5),
        "ks": round(ks, 5),
        "ks_threshold": round(ks_thr, 5),
        "valid_rows": int(len(y)),
        "valid_pos": int(y.sum()),
        "base_rate": round(float(y.mean()), 5),
    }

    # 1) ROC
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color=PALETTE["pos"], lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC-кривая (LightGBM, valid)")
    ax.legend(loc="lower right")
    viz.save(fig, f"{OUT}/01_roc.png")

    # 2) PR
    prec, rec, _ = precision_recall_curve(y, s)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(rec, prec, color=PALETTE["accent"], lw=2, label=f"PR-AUC = {pr_auc:.4f}")
    ax.axhline(y.mean(), ls="--", color="grey", lw=1, label=f"baseline = {y.mean():.4f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall (дисбаланс 3.55%)")
    ax.legend()
    viz.save(fig, f"{OUT}/02_pr.png")

    # 3) Распределение скоров по классам
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, s.max(), 60)
    ax.hist(s[y == 0], bins=bins, density=True, alpha=0.6, color=PALETTE["neg"], label="не дефолт (0)")
    ax.hist(s[y == 1], bins=bins, density=True, alpha=0.6, color=PALETTE["pos"], label="дефолт (1)")
    ax.axvline(ks_thr, ls="--", color="black", lw=1, label=f"KS-порог = {ks_thr:.3f}")
    ax.set_xlabel("предсказанный score"); ax.set_ylabel("плотность")
    ax.set_title("Распределение скоров по классам")
    ax.legend()
    viz.save(fig, f"{OUT}/03_score_distribution.png")

    # 4) KS
    order = np.argsort(s)
    s_sorted, y_sorted = s[order], y[order]
    cum_pos = np.cumsum(y_sorted) / y_sorted.sum()
    cum_neg = np.cumsum(1 - y_sorted) / (1 - y_sorted).sum()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(s_sorted, cum_pos, color=PALETTE["pos"], label="CDF дефолт (1)")
    ax.plot(s_sorted, cum_neg, color=PALETTE["neg"], label="CDF не дефолт (0)")
    ax.set_title(f"KS-статистика = {ks:.4f}")
    ax.set_xlabel("score"); ax.set_ylabel("накопленная доля")
    ax.legend()
    viz.save(fig, f"{OUT}/04_ks.png")

    # 5) Калибровка
    frac_pos, mean_pred = calibration_curve(y, s, n_bins=15, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(mean_pred, frac_pos, marker="o", color=PALETTE["accent"], label="LightGBM")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", label="идеал")
    ax.set_xlabel("средний предсказанный score"); ax.set_ylabel("фактическая доля дефолта")
    ax.set_title("Калибровочная кривая")
    ax.legend()
    viz.save(fig, f"{OUT}/05_calibration.png")

    # 6) Важность признаков (gain, top-30)
    booster = lgb.Booster(model_file=f"{MODELS}/lgbm.txt")
    imp = pd.DataFrame({
        "feature": booster.feature_name(),
        "gain": booster.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False).head(30).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(imp["feature"], imp["gain"], color=PALETTE["accent"])
    ax.set_title("Важность признаков (gain, top-30)")
    ax.set_xlabel("gain")
    viz.save(fig, f"{OUT}/06_feature_importance.png")
    metrics["top10_features"] = imp["feature"].iloc[::-1].head(10).tolist()

    # 7) Матрица ошибок при KS-пороге
    y_pred = (s >= ks_thr).astype(int)
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred 0", "pred 1"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["true 0", "true 1"])
    ax.set_title(f"Матрица ошибок (порог KS={ks_thr:.3f})")
    viz.save(fig, f"{OUT}/07_confusion.png")

    # 8) Сравнение моделей
    # Порядок слева-направо: от слабой к сильной. LightGBM - основная (в submission).
    models_cmp = []

    base_path = f"{OUT}/baseline_metrics.json"
    if os.path.exists(base_path):
        b = json.load(open(base_path, encoding="utf-8"))
        models_cmp.append(("Spark MLlib LogReg", PALETTE["neg"],
                           [b["roc_auc"], b["gini"], b["pr_auc"]]))
        metrics["baseline"] = {k: b[k] for k in ("roc_auc", "gini", "pr_auc")}

    cb_pred = f"{C.LOCAL}/data/valid_pred_catboost.parquet"
    if os.path.exists(cb_pred):
        cb = pd.read_parquet(cb_pred)
        cb_auc = float(roc_auc_score(cb["y_true"], cb["y_score"]))
        cb_pr = float(average_precision_score(cb["y_true"], cb["y_score"]))
        cb_row = [round(cb_auc, 5), round(2 * cb_auc - 1, 5), round(cb_pr, 5)]
        models_cmp.append(("CatBoost", PALETTE["accent"], cb_row))
        metrics["catboost"] = {"roc_auc": cb_row[0], "gini": cb_row[1], "pr_auc": cb_row[2]}

    models_cmp.append(("LightGBM", PALETTE["pos"],
                       [metrics["roc_auc"], metrics["gini"], metrics["pr_auc"]]))

    names = ["ROC-AUC", "Gini", "PR-AUC"]
    x = np.arange(len(names))
    n = len(models_cmp)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, (label, color, vals) in enumerate(models_cmp):
        off = (i - (n - 1) / 2) * width
        ax.bar(x + off, vals, width, label=label, color=color)
        for xi, v in zip(x, vals):
            ax.text(xi + off, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_title("Сравнение моделей (valid)")
    ax.legend(); ax.margins(y=0.15)
    viz.save(fig, f"{OUT}/08_model_comparison.png")

    with open(f"{OUT}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(">>> EVAL:", json.dumps(metrics, ensure_ascii=False))
    print(">>> EVALUATE DONE")


if __name__ == "__main__":
    main()
