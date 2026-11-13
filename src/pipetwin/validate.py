"""Метрики."""
from __future__ import annotations

import numpy as np


def pearson_r2(y_true, y_pred) -> float:
    """Квадрат коэффициента корреляции Пирсона — метрика организаторов. NaN для константы."""
    a, b = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"разная длина: {len(a)} и {len(b)}")
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    # Константа определяется по равенству значений: std от [0.1, 0.1, 0.1] в машинной арифметике не ноль.
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1] ** 2)


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def top_recall(y_true, score, frac: float = 0.10) -> float:
    """Доля реально самых тонких сегментов (нижние frac), попавших в наш список риска.

    score — чем меньше, тем опаснее (например, нижняя граница толщины).
    """
    y_true, score = np.asarray(y_true, dtype=float), np.asarray(score, dtype=float)
    k = max(1, int(round(len(y_true) * frac)))
    worst = set(np.argsort(y_true)[:k])
    ours = set(np.argsort(score)[:k])
    return len(worst & ours) / k
