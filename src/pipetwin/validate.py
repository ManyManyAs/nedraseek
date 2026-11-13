"""Метрика организаторов и бэктест по прошлым пропускам ВТД."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import baselines
from .segments import thickness_at_run


def pearson_r2(y_true, y_pred) -> float:
    """Квадрат коэффициента корреляции Пирсона. NaN, если один из рядов константа."""
    a, b = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1] ** 2)


def backtest_case1(grid, vtd_seg, pipe_id) -> pd.DataFrame:
    """Прогноз пропуска k+1 по пропускам <= k. Одна строка на (целевой пропуск, метод).

    Будущее не используется: каждый метод видит только пропуски до целевого.
    """
    dates = sorted(vtd_seg.loc[vtd_seg["id"] == pipe_id, "date"].unique())
    rows = []
    for k in range(1, len(dates)):
        target, last = dates[k], dates[k - 1]
        fact = thickness_at_run(grid, vtd_seg, pipe_id, target)["thk_min_mm"].to_numpy()
        horizon = (pd.Timestamp(target) - pd.Timestamp(last)).days / 365.25
        preds = {"persistence": baselines.persistence(grid, vtd_seg, pipe_id, last)}
        if k >= 2:
            preds["linear_growth"] = baselines.linear_growth(
                grid, vtd_seg, pipe_id, pd.Timestamp(dates[k - 2]), pd.Timestamp(last), pd.Timestamp(target))
        for name, p in preds.items():
            y = p["thickness_pred"].to_numpy()
            rows.append({"id": pipe_id, "target_run": pd.Timestamp(target).date(), "horizon_years": round(horizon, 2),
                         "method": name, "r2": pearson_r2(fact, y),
                         "mae_mm": float(np.mean(np.abs(fact - y)))})
    return pd.DataFrame(rows)
