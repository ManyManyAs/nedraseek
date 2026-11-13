"""Сетка 10-метровых сегментов и привязка к ней ВТД, УЗТ, профиля и режима."""
from __future__ import annotations

import numpy as np
import pandas as pd


def segment_id(dist_m, seg_len: float = 10.0):
    """Номер сегмента: целая часть дистанции (м), делённой на длину сегмента."""
    return np.floor(np.asarray(dist_m, dtype=float) / seg_len).astype(int)


def build_grid(passport: pd.DataFrame, seg_len: float, sample: pd.DataFrame | None) -> pd.DataFrame:
    """Одна строка на сегмент. Если есть sample_result.csv, список сегментов берётся из него."""
    if sample is not None:
        grid = sample[["id", "segment_id"]].drop_duplicates().copy()
    else:
        rows = []
        for r in passport.itertuples():
            n = int(np.ceil(r.length_m / seg_len))  # допущение, см. DATA_NOTES
            rows.append(pd.DataFrame({"id": r.id, "segment_id": np.arange(n)}))
        grid = pd.concat(rows, ignore_index=True)
    grid["dist_mid_m"] = (grid["segment_id"] + 0.5) * seg_len
    cols = [c for c in ["id", "wall_mm", "start_date", "length_m", "diameter_mm"] if c in passport.columns]
    return grid.merge(passport[cols], on="id", how="left")


def vtd_by_segment(vtd: pd.DataFrame, passport: pd.DataFrame, seg_len: float,
                   exclude_types: list | None = None) -> pd.DataFrame:
    """Агрегат дефектов: одна строка на (труба, пропуск, сегмент).

    Минимальная толщина = номинал * (1 - макс. глубина % / 100).
    Сегменты без дефектов сюда не попадают; для них берётся номинал.
    """
    if vtd.empty:
        return pd.DataFrame(columns=["id", "date", "segment_id", "max_depth_pct", "n_defects", "thk_min_mm"])
    d = vtd.copy()
    if exclude_types and "type" in d.columns:
        d = d[~d["type"].isin(exclude_types)]
    d = d.dropna(subset=["dist_m", "depth_pct", "date"])
    d = d[(d["depth_pct"] >= 0) & (d["depth_pct"] <= 100)]
    d["segment_id"] = segment_id(d["dist_m"], seg_len)
    agg = (d.groupby(["id", "date", "segment_id"])
            .agg(max_depth_pct=("depth_pct", "max"), n_defects=("depth_pct", "size"))
            .reset_index())
    agg = agg.merge(passport[["id", "wall_mm"]], on="id", how="left")
    agg["thk_min_mm"] = agg["wall_mm"] * (1 - agg["max_depth_pct"] / 100)
    return agg.drop(columns="wall_mm")


def thickness_at_run(grid: pd.DataFrame, vtd_seg: pd.DataFrame, pipe_id, date) -> pd.DataFrame:
    """Полная карта толщины по сегментам на дату пропуска (без дефектов -> номинал)."""
    g = grid[grid["id"] == pipe_id][["id", "segment_id", "wall_mm"]].copy()
    run = vtd_seg[(vtd_seg["id"] == pipe_id) & (vtd_seg["date"] == date)]
    g = g.merge(run[["segment_id", "thk_min_mm", "max_depth_pct", "n_defects"]], on="segment_id", how="left")
    g["thk_min_mm"] = g["thk_min_mm"].fillna(g["wall_mm"])
    g[["max_depth_pct", "n_defects"]] = g[["max_depth_pct", "n_defects"]].fillna(0)
    return g


def profile_features(grid: pd.DataFrame, profile: pd.DataFrame, seg_len: float, window_m: float) -> pd.DataFrame:
    """Рельеф: высота, уклон, глубина относительно окрестности (низины копят воду)."""
    out = []
    for pid, g in grid.groupby("id"):
        p = profile[profile["id"] == pid].dropna(subset=["dist_m", "z_m"]).sort_values("dist_m")
        p = p.drop_duplicates("dist_m")
        x = g["dist_mid_m"].to_numpy()
        if len(p) < 2:
            z = np.zeros_like(x)
        else:
            z = np.interp(x, p["dist_m"].to_numpy(), p["z_m"].to_numpy())
        slope = np.gradient(z, x) if len(x) > 1 else np.zeros_like(x)
        w = max(3, int(round(window_m / seg_len)) | 1)
        zs = pd.Series(z)
        local_max = zs.rolling(w, center=True, min_periods=1).max().to_numpy()
        local_mean = zs.rolling(w, center=True, min_periods=1).mean().to_numpy()
        out.append(pd.DataFrame({
            "id": pid, "segment_id": g["segment_id"].to_numpy(),
            "z_m": z, "slope": slope, "abs_slope": np.abs(slope),
            "depth_below_local_max_m": local_max - z,   # насколько сегмент «в яме»
            "z_minus_local_mean_m": z - local_mean,
        }))
    return pd.concat(out, ignore_index=True)


def map_regime_to_segments(grid: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    """Соответствие сегмент 10 м -> сегмент гидравлического расчёта.

    Если в режиме есть hseg_start_m/hseg_end_m, используем их. Иначе делим трубу
    на равные части по числу гидравлических сегментов (допущение, см. DATA_NOTES).
    """
    out = []
    for pid, g in grid.groupby("id"):
        r = regime[regime["id"] == pid]
        hsegs = sorted(r["hseg"].dropna().unique())
        x = g["dist_mid_m"].to_numpy()
        if not hsegs:
            h = np.full(len(g), np.nan)
        elif {"hseg_start_m", "hseg_end_m"} <= set(r.columns):
            b = r.groupby("hseg")[["hseg_start_m", "hseg_end_m"]].first().sort_values("hseg_start_m")
            idx = np.clip(np.searchsorted(b["hseg_start_m"].to_numpy(), x, side="right") - 1, 0, len(b) - 1)
            h = b.index.to_numpy()[idx]
        else:
            edges = np.linspace(0, g["length_m"].iloc[0], len(hsegs) + 1)
            idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(hsegs) - 1)
            h = np.asarray(hsegs)[idx]
        out.append(pd.DataFrame({"id": pid, "segment_id": g["segment_id"].to_numpy(), "hseg": h}))
    return pd.concat(out, ignore_index=True)
