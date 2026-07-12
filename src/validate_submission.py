"""Проверка submission.csv перед сдачей.

У test нет меток, поэтому качество на нём не посчитать. Что реально можно проверить:
  * формат и полнота (все клиенты на месте, id совпадают, дублей нет);
  * корректность скоров (нет NaN, лежат в [0, 1], модель не выродилась);
  * отсутствие дрейфа - распределение скоров на test должно совпадать с валидацией,
    иначе модель на test ведёт себя иначе, чем мы измерили.

Жёсткие нарушения роняют задачу (её нельзя пропустить молча), мягкие - печатаются
как предупреждения. Итог пишется в reports/model/submission_check.json.

Запуск (в контейнере trainer):
    python src/validate_submission.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import config as C

OUT = f"{C.REPORTS}/model"
SUBMISSION = f"{C.LOCAL}/submission.csv"
VALID_PRED = f"{C.LOCAL}/data/valid_pred.parquet"

# порог расхождения среднего скора между test и валидацией (относительный)
DRIFT_TOLERANCE = 0.20
# минимальная доля уникальных скоров - защита от вырожденной модели
MIN_UNIQUE_RATIO = 0.50


def main():
    os.makedirs(OUT, exist_ok=True)
    errors, warnings, report = [], [], {}

    sub = pd.read_csv(SUBMISSION)
    tgt = pd.read_csv(C.SRC_TEST_TARGET)

    # формат и полнота
    if list(sub.columns) != ["id", "score"]:
        errors.append(f"ожидались колонки ['id','score'], а есть {list(sub.columns)}")
    if len(sub) != len(tgt):
        errors.append(f"строк {len(sub):,}, а в test_target {len(tgt):,}")
    n_dup = int(sub["id"].duplicated().sum())
    if n_dup:
        errors.append(f"дубликаты id: {n_dup}")
    missing = set(tgt["id"]) - set(sub["id"])
    extra = set(sub["id"]) - set(tgt["id"])
    if missing:
        errors.append(f"нет скоров для {len(missing):,} клиентов из test_target")
    if extra:
        errors.append(f"лишние {len(extra):,} id, которых нет в test_target")

    report.update(rows=len(sub), expected_rows=len(tgt), duplicates=n_dup,
                  missing_ids=len(missing), extra_ids=len(extra))

    # корректность скоров
    s = sub["score"].to_numpy(dtype=float)
    n_nan = int(np.isnan(s).sum())
    n_out = int(((s < 0) | (s > 1)).sum())
    if n_nan:
        errors.append(f"NaN в score: {n_nan}")
    if n_out:
        errors.append(f"score вне [0,1]: {n_out}")

    uniq_ratio = len(np.unique(s)) / len(s) if len(s) else 0.0
    if uniq_ratio < MIN_UNIQUE_RATIO:
        errors.append(f"модель выродилась: уникальных скоров всего {uniq_ratio:.1%}")

    report.update(nan=n_nan, out_of_range=n_out, unique_ratio=round(uniq_ratio, 4),
                  score_min=round(float(s.min()), 5), score_mean=round(float(s.mean()), 5),
                  score_max=round(float(s.max()), 5))

    # дрейф относительно валидации
    if os.path.exists(VALID_PRED):
        v = pd.read_parquet(VALID_PRED)["y_score"].to_numpy()
        drift = abs(s.mean() - v.mean()) / v.mean()
        report["valid_score_mean"] = round(float(v.mean()), 5)
        report["mean_drift"] = round(float(drift), 4)
        report["percentiles"] = {
            "valid": [round(float(x), 5) for x in np.percentile(v, [50, 90, 99])],
            "test": [round(float(x), 5) for x in np.percentile(s, [50, 90, 99])],
        }
        if drift > DRIFT_TOLERANCE:
            warnings.append(
                f"средний скор на test ({s.mean():.4f}) расходится с валидацией "
                f"({v.mean():.4f}) на {drift:.1%} - модель на test ведёт себя иначе"
            )
    else:
        warnings.append("valid_pred.parquet не найден - проверку на дрейф пропускаю")

    # итог
    report["errors"] = errors
    report["warnings"] = warnings
    report["passed"] = not errors
    with open(f"{OUT}/submission_check.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    for w in warnings:
        print(f"    [warn] {w}")
    if errors:
        for e in errors:
            print(f"    [FAIL] {e}")
        print(">>> SUBMISSION НЕ ПРОШЁЛ ПРОВЕРКУ")
        sys.exit(1)

    print(f">>> submission OK: {len(sub):,} строк, скоры [{s.min():.4f}, {s.max():.4f}], "
          f"среднее {s.mean():.4f}, уникальных {uniq_ratio:.1%}")
    print(">>> VALIDATE DONE")


if __name__ == "__main__":
    main()
