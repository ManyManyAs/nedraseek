"""Базовые решения. С ними сравнивается любая итоговая модель.

Кейс 1: persistence (толщина по предыдущему ВТД) и линейная экстраполяция роста.
Кейс 2: физическая скорость CO2-коррозии, накопленная за срок службы, с поправкой на рельеф.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .segments import thickness_at_run


def persistence(grid, vtd_seg, pipe_id, last_date) -> pd.DataFrame:
    """Прогноз = минимальная толщина по последнему известному ВТД."""
    g = thickness_at_run(grid, vtd_seg, pipe_id, last_date)
    return g[["id", "segment_id"]].assign(thickness_pred=g["thk_min_mm"].to_numpy())


def linear_growth(grid, vtd_seg, pipe_id, prev_date, last_date, t_date, shrink: float = 0.5) -> pd.DataFrame:
    """Экстраполяция утонения между двумя последними ВТД на дату T.

    shrink < 1 гасит шум измерений; отрицательный «рост» обнуляется.
    """
    a = thickness_at_run(grid, vtd_seg, pipe_id, prev_date)
    b = thickness_at_run(grid, vtd_seg, pipe_id, last_date)
    years_ab = max((last_date - prev_date).days / 365.25, 1e-6)
    years_bt = max((t_date - last_date).days / 365.25, 0.0)
    rate = np.clip((a["thk_min_mm"].to_numpy() - b["thk_min_mm"].to_numpy()) / years_ab, 0, None)
    pred = b["thk_min_mm"].to_numpy() - shrink * rate * years_bt
    return b[["id", "segment_id"]].assign(thickness_pred=np.clip(pred, 0, None))


def dewaard_rate_mm_per_year(t_c, p_co2_bar):
    """Де Ваард–Мильямс (1991): lg V = 5.8 - 1710/(T+273) + 0.67*lg(pCO2)."""
    t_c = np.asarray(t_c, dtype=float)
    p = np.clip(np.asarray(p_co2_bar, dtype=float), 1e-4, None)
    return 10 ** (5.8 - 1710.0 / (t_c + 273.15) + 0.67 * np.log10(p))


def regime_exposure(regime: pd.DataFrame, inhibitor_eff: float) -> pd.DataFrame:
    """Накопленное утонение (мм) по гидравлическим сегментам за весь доступный период.

    ДОПУЩЕНИЯ (проверить на данных): co2 — мольная доля в %, p — в барах, t — в °C.
    Отсутствующие колонки заменяются нейтральными значениями.
    """
    r = regime.copy()
    t = r["t"] if "t" in r else 40.0
    p = r["p"] if "p" in r else 10.0
    co2 = r["co2"] if "co2" in r else 1.0
    rate = dewaard_rate_mm_per_year(t, np.asarray(p, dtype=float) * np.asarray(co2, dtype=float) / 100.0)
    if "watercut" in r:
        rate = rate * np.clip(r["watercut"].to_numpy(dtype=float) / 100.0, 0.05, 1.0)
    if "inhibitor" in r:
        rate = rate * np.where(r["inhibitor"].fillna(0).to_numpy(dtype=float) > 0, 1 - inhibitor_eff, 1.0)
    r["loss_mm"] = np.nan_to_num(rate) / 365.25
    return r.groupby(["id", "hseg"])["loss_mm"].sum().reset_index()


def physics_case2(grid, prof_feats, seg2hseg, regime, pipe_id, cfg) -> pd.DataFrame:
    """Прогноз от номинала: накопленная физическая потеря, усиленная в низинах."""
    c = cfg["case2"]
    g = grid[grid["id"] == pipe_id][["id", "segment_id", "wall_mm"]]
    g = g.merge(prof_feats, on=["id", "segment_id"], how="left").merge(seg2hseg, on=["id", "segment_id"], how="left")
    exp = regime_exposure(regime[regime["id"] == pipe_id], c["inhibitor_efficiency"])
    g = g.merge(exp, on=["id", "hseg"], how="left")
    loss = g["loss_mm"].fillna(g["loss_mm"].median()).fillna(0.0).to_numpy()
    low = g["depth_below_local_max_m"].fillna(0).to_numpy()
    low_n = low / low.max() if low.max() > 0 else low
    # Малый вклад уклона гарантирует ненулевую дисперсию прогноза (иначе r² не определён).
    slope = g["abs_slope"].fillna(0).to_numpy()
    slope_n = slope / slope.max() if slope.max() > 0 else slope
    factor = 1 + c["lowpoint_weight"] * low_n - 0.2 * slope_n
    scale = loss if loss.max() > 0 else np.ones_like(loss)
    pred = g["wall_mm"].to_numpy() - np.clip(scale * factor, 0, None)
    return g[["id", "segment_id"]].assign(thickness_pred=np.clip(pred, 0, None))
