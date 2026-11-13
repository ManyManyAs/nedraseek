"""Признаки сегментов: рельеф трассы и технологический режим."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import SEG_LEN, n_segments


def _roll(a: np.ndarray, w: int, how: str) -> np.ndarray:
    s = pd.Series(a).rolling(w, center=True, min_periods=1)
    return getattr(s, how)().to_numpy()


def profile_features(profile: pd.DataFrame, pipe_id: int, length_m: float) -> pd.DataFrame:
    """Рельеф по сегментам. Низины важны физически: в них скапливается вода и осадок."""
    n = n_segments(length_m)
    x = (np.arange(n) + 0.5) * SEG_LEN
    p = profile[profile["id"] == pipe_id]
    out = pd.DataFrame({"segment_id": np.arange(n), "p_dist": x, "p_frac": x / max(length_m, 1.0)})
    if len(p) < 2:
        for c in ("p_z", "p_slope", "p_abs_slope", "p_pit100", "p_pit300", "p_pit1000", "p_zrel300",
                  "p_curv", "p_is_min", "p_after_min", "p_dist_min"):
            out[c] = 0.0
        return out
    z_raw = pd.Series(p["z_m"].to_numpy()).rolling(3, center=True, min_periods=1).median().to_numpy()
    z = np.interp(x, p["dist_m"].to_numpy(), z_raw)
    slope = np.gradient(z, x) if n > 1 else np.zeros(n)
    out["p_z"] = z
    out["p_slope"] = slope                      # > 0: подъём по ходу потока
    out["p_abs_slope"] = np.abs(slope)
    for w_m in (100, 300, 1000):
        w = max(3, int(w_m / SEG_LEN) | 1)
        out[f"p_pit{w_m}"] = _roll(z, w, "max") - z   # глубина «ямы» относительно окрестности
    out["p_zrel300"] = z - _roll(z, 31, "mean")
    out["p_curv"] = np.gradient(slope, x) if n > 1 else 0.0
    w = 21
    is_min = (z <= _roll(z, w, "min") + 1e-9) & (out["p_pit300"].to_numpy() > 0.3)
    out["p_is_min"] = is_min.astype(float)
    idx = np.where(is_min)[0]
    if len(idx):
        d = np.abs(np.arange(n)[:, None] - idx[None, :]).min(axis=1) * SEG_LEN
        prev = np.array([idx[idx <= i].max() if (idx <= i).any() else -10**6 for i in range(n)])
        after = ((np.arange(n) - prev) * SEG_LEN <= 50) & (slope > 0)
    else:
        d, after = np.full(n, length_m), np.zeros(n, dtype=bool)
    out["p_dist_min"] = d
    out["p_after_min"] = after.astype(float)
    return out


REGIME_COLS = ["watercut", "v_mix", "v_liquid_true", "t_mix", "p_start", "q", "q_oil", "q_gas",
               "gas_content_rate", "pit_corrosion_intensity", "fill_coefficient"]


def regime_features(regime: pd.DataFrame, pipe_id: int, date_from, date_to) -> dict:
    """Агрегаты режима участка за интервал. Режим один на весь участок."""
    r = regime[(regime["id"] == pipe_id) & (regime["date"] > pd.Timestamp(date_from)) &
               (regime["date"] <= pd.Timestamp(date_to))]
    years = (pd.Timestamp(date_to) - pd.Timestamp(date_from)).days / 365.25
    out = {"r_years": years, "r_coverage": len(r) / max((pd.Timestamp(date_to) - pd.Timestamp(date_from)).days, 1)}
    for c in REGIME_COLS:
        v = r[c].astype(float) if c in r else pd.Series(dtype=float)
        if c == "p_start":
            v = v[v < 200]  # выбросы давления
        out[f"r_{c}"] = float(v.median()) if len(v) else np.nan
    out["r_lowv_share"] = float((r["v_mix"] < 0.3).mean()) if len(r) else np.nan
    out["r_laminar_share"] = float((r["flow_regime_real"] == "Ламинарный").mean()) if len(r) else np.nan
    return out
