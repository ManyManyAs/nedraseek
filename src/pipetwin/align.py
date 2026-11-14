"""Выравнивание дистанций между пропусками ВТД.

Одометры разных снарядов расходятся: одна и та же точка трубы получает разную дистанцию.
При сегменте 10 м расхождение в несколько метров переносит дефекты в соседний сегмент,
поэтому до сравнения пропусков дистанции приводятся к шкале целевого пропуска.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _profile(x: np.ndarray, bins: np.ndarray, smooth: int) -> np.ndarray:
    h = np.histogram(x, bins=bins)[0].astype(float)
    return np.convolve(h, np.ones(smooth) / smooth, mode="same")


def estimate_shift(xa, xb, lo: float, hi: float, k0: float = 0.0, span: float = 15.0,
                   res: float = 0.25, smooth_m: float = 1.0) -> tuple[float, float]:
    """Сдвиг k, при котором профиль плотности дефектов A, сдвинутый на k, лучше всего
    совпадает с профилем B на окне [lo, hi) шкалы B. Возвращает (k, корреляция)."""
    xa, xb = np.asarray(xa, dtype=float), np.asarray(xb, dtype=float)
    bins = np.arange(lo - 3, hi + 3 + res, res)
    sm = max(1, int(round(smooth_m / res)))
    hb = _profile(xb, bins, sm)
    best_k, best_c = k0, -1.0
    if hb.std() == 0:
        return best_k, float("nan")
    for k in np.arange(k0 - span, k0 + span + res / 2, res):
        ha = _profile(xa + k, bins, sm)
        if ha.std() == 0:
            continue
        c = float(np.corrcoef(ha, hb)[0, 1])
        if c > best_c:
            best_k, best_c = float(k), c
    return best_k, best_c


def window_shifts(xa, xb, length_m: float, window: float = 100.0, k0: float = 0.0,
                  span: float = 15.0, min_points: int = 15) -> pd.DataFrame:
    """Сдвиги по окнам вдоль участка: видно, растёт ли расхождение с дистанцией."""
    xa, xb = np.asarray(xa, dtype=float), np.asarray(xb, dtype=float)
    rows, k_prev = [], k0
    for lo in np.arange(0, length_m, window):
        hi = lo + window
        if ((xb >= lo) & (xb < hi)).sum() < min_points:
            continue
        k, c = estimate_shift(xa, xb, lo, hi, k_prev, span)
        rows.append({"lo": lo, "hi": hi, "mid": lo + window / 2, "shift": k, "corr": c})
        if c > 0.5:
            k_prev = k
    return pd.DataFrame(rows)


def fine_align(a: pd.DataFrame, b: pd.DataFrame, length_m: float, coarse: pd.DataFrame,
               window: float = 25.0) -> np.ndarray:
    """Дистанции дефектов A в шкале B: грубый сдвиг по окнам + уточнение в окнах по 25 м."""
    mid = coarse["mid"].to_numpy()
    k_coarse = lambda x: float(coarse["shift"].to_numpy()[np.abs(mid - x).argmin()])  # noqa: E731
    xa = a["x"].to_numpy()
    x2 = xa + np.array([k_coarse(x) for x in xa])
    for lo in np.arange(0, length_m, window):
        m = (xa >= lo) & (xa < lo + window)
        if m.sum() < 5:
            continue
        k0 = k_coarse(lo + window / 2)
        k, c = estimate_shift(xa, b["x"].to_numpy(), lo + k0, lo + window + k0, k0, span=1.5, res=0.05, smooth_m=0.3)
        if c > 0.3:
            x2[m] = xa[m] + k
    return x2
