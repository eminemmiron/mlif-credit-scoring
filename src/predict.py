"""Инференс: скоринг test-клиентов обученной моделью -> submission.csv (id, score).

Запуск (в контейнере trainer):
    python src/predict.py
"""
import json

import lightgbm as lgb
import pandas as pd

import config as C
from data_io import load_features

MODELS = f"{C.LOCAL}/models"
SUBMISSION = f"{C.LOCAL}/submission.csv"


def main():
    with open(f"{MODELS}/feature_cols.json", encoding="utf-8") as f:
        feat_cols = json.load(f)
    booster = lgb.Booster(model_file=f"{MODELS}/lgbm.txt")

    df = load_features(f"{C.LOCAL}/data/test_features")
    ids = df[C.KEY].to_numpy()
    X = df[feat_cols]                      # порядок колонок как при обучении
    del df

    scores = booster.predict(X)
    sub = pd.DataFrame({"id": ids, "score": scores}).sort_values("id")
    sub.to_csv(SUBMISSION, index=False)
    print(f">>> submission rows={len(sub):,} "
          f"score[min/mean/max]={scores.min():.4f}/{scores.mean():.4f}/{scores.max():.4f}")
    print(f">>> saved {SUBMISSION}")
    print(">>> PREDICT DONE")


if __name__ == "__main__":
    main()
