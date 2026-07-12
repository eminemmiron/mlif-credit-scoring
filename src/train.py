"""(основная модель). Обучение LightGBM на клиентской витрине.

Метрика - ROC-AUC (ранговая, устойчива к дисбалансу 3.55%). Сохраняем модель,
список фич и валид-предсказания (id, y_true, y_score) для этапа оценки.

Запуск (внутри контейнера trainer):
    python src/train.py
"""
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config as C
from data_io import load_features

MODELS = f"{C.LOCAL}/models"
VALID_PRED = f"{C.LOCAL}/data/valid_pred.parquet"

SPLIT_SEED = 42
VALID_FRACTION = 0.2


def hash_unit_interval(ids: np.ndarray, seed: int) -> np.ndarray:
    """Детерминированное число в [0, 1) для каждого id (финализатор splitmix64).

    Даёт воспроизводимый train/valid-сплит: клиент всегда попадает в ту же
    часть, независимо от порядка строк в витрине.
    """
    u64 = np.uint64
    x = ids.astype(u64) + u64((seed * 0x9E3779B97F4A7C15) % 2**64)
    x ^= x >> u64(30)
    x *= u64(0xBF58476D1CE4E5B9)
    x ^= x >> u64(27)
    x *= u64(0x94D049BB133111EB)
    x ^= x >> u64(31)
    return (x >> u64(11)).astype(np.float64) / float(1 << 53)

PARAMS = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.03,
    num_leaves=128,
    min_child_samples=300,
    feature_fraction=0.6,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=5.0,
    max_depth=-1,
    num_threads=0,
    verbosity=-1,
)


def main():
    os.makedirs(MODELS, exist_ok=True)
    df = load_features(f"{C.LOCAL}/data/train_features")
    feat_cols = [c for c in df.columns if c not in (C.KEY, C.TARGET)]
    print(f"rows={len(df):,} features={len(feat_cols)}")

    # Память: удаляем df до создания копий-сплитов, работаем по индексам.
    y = df[C.TARGET].to_numpy()
    ids = df[C.KEY].to_numpy()
    Xdf = df[feat_cols]
    del df

    # Сплит по хешу id, а не по позиции строки: Spark раскидывает строки по
    # партициям со случайным сдвигом, поэтому порядок строк в витрине меняется
    # от прогона к прогону. Сплит по индексам из-за этого «плавал» вместе с
    # метриками (AUC гулял на ±0.003). Хеш привязан к самому клиенту => одна и
    # та же выборка при любом порядке строк.
    frac = hash_unit_interval(ids, seed=SPLIT_SEED)
    va_mask = frac < VALID_FRACTION
    va_idx = np.flatnonzero(va_mask)
    tr_idx = np.flatnonzero(~va_mask)

    X_va = Xdf.iloc[va_idx].copy()          # держим valid для предсказаний
    y_va, id_va = y[va_idx], ids[va_idx]
    # Хеш независим от метки, поэтому доля дефолтов в обеих частях сохраняется
    # сама собой - отдельная стратификация не нужна (проверяем печатью).
    print(f"train={len(tr_idx):,} valid={len(va_idx):,} | "
          f"доля дефолта: train={y[tr_idx].mean():.4f} valid={y_va.mean():.4f}")

    dtr = lgb.Dataset(Xdf.iloc[tr_idx], label=y[tr_idx], free_raw_data=True)
    del Xdf
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr, free_raw_data=False)

    booster = lgb.train(
        PARAMS, dtr,
        num_boost_round=3000,
        valid_sets=[dtr, dva],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(150),
            lgb.log_evaluation(100),
        ],
    )

    y_score = booster.predict(X_va, num_iteration=booster.best_iteration)
    auc = roc_auc_score(y_va, y_score)
    print(f">>> LightGBM valid ROC-AUC = {auc:.5f} | Gini = {2*auc-1:.5f} | "
          f"best_iter = {booster.best_iteration}")

    booster.save_model(f"{MODELS}/lgbm.txt", num_iteration=booster.best_iteration)
    with open(f"{MODELS}/feature_cols.json", "w", encoding="utf-8") as f:
        json.dump(feat_cols, f)
    pd.DataFrame({C.KEY: id_va, "y_true": y_va, "y_score": y_score}) \
        .to_parquet(VALID_PRED, index=False)
    print(">>> saved model, feature list, valid predictions")
    print(">>> TRAIN DONE")


if __name__ == "__main__":
    main()
