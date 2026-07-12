"""Обучение CatBoost - третья модель для сравнения.

Обучается на ТОЙ ЖЕ витрине и ТОМ ЖЕ hold-out сплите, что и LightGBM (сплит по
хешу id импортируется из train.py), поэтому сравнение честное - на одной валидации.

Все признаки поданы как числовые (как и в LightGBM) - так модели сравниваются на
идентичной матрице. Отдельный резерв CatBoost (нативные категории для `_last`/`_max`
кодовых столбцов) намеренно не задействован, чтобы сравнение осталось «модель против модели».

Запуск (в контейнере trainer):
    python src/train_catboost.py
"""
import json
import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

import config as C
from data_io import load_features
from train import SPLIT_SEED, VALID_FRACTION, hash_unit_interval

MODELS = f"{C.LOCAL}/models"
OUT = f"{C.REPORTS}/model"
VALID_PRED = f"{C.LOCAL}/data/valid_pred_catboost.parquet"

PARAMS = dict(
    iterations=1500,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    l2_leaf_reg=6.0,
    border_count=128,        # меньше корзин -> легче по памяти/быстрее
    random_seed=42,
    thread_count=-1,
    verbose=200,
)


def main():
    os.makedirs(MODELS, exist_ok=True)
    df = load_features(f"{C.LOCAL}/data/train_features")
    feat_cols = [c for c in df.columns if c not in (C.KEY, C.TARGET)]
    print(f"rows={len(df):,} features={len(feat_cols)}")

    # Память: удаляем df до создания копий-сплитов (тот же приём, что в train.py).
    y = df[C.TARGET].to_numpy()
    ids = df[C.KEY].to_numpy()
    Xdf = df[feat_cols]
    del df

    va = hash_unit_interval(ids, SPLIT_SEED) < VALID_FRACTION
    X_tr = Xdf[~va]
    X_va = Xdf[va].copy()
    del Xdf
    y_tr, y_va, id_va = y[~va], y[va], ids[va]
    print(f"train={len(y_tr):,} valid={len(y_va):,} | "
          f"доля дефолта: train={y_tr.mean():.4f} valid={y_va.mean():.4f}")

    model = CatBoostClassifier(**PARAMS)
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va),
              early_stopping_rounds=100, use_best_model=True)
    del X_tr

    score = model.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, score)
    print(f">>> CatBoost valid ROC-AUC = {auc:.5f} | Gini = {2*auc-1:.5f} | "
          f"best_iter = {model.get_best_iteration()}")

    model.save_model(f"{MODELS}/catboost.cbm")
    pd.DataFrame({C.KEY: id_va, "y_true": y_va, "y_score": score}) \
        .to_parquet(VALID_PRED, index=False)
    with open(f"{OUT}/catboost_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"model": "catboost",
                   "roc_auc": round(float(auc), 5),
                   "gini": round(2 * float(auc) - 1, 5)},
                  f, ensure_ascii=False, indent=2)
    print(">>> CATBOOST DONE")


if __name__ == "__main__":
    main()
