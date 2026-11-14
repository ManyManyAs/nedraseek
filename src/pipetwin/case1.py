"""Кейс 1: прогноз между пропусками ВТД.

Идея. Минимальная толщина сегмента: это самый глубокий дефект, который увидит следующий снаряд.
Поэтому прогноз строится на уровне отдельных дефектов:

  1. дефекты стартового пропуска переводятся в шкалу дистанций целевого пропуска (align.py);
  2. на участках, где известны оба пропуска, дефекты сопоставляются попарно, и из пар
     оценивается закон изменения глубины: как глубина в новом пропуске зависит от глубины
     в старом, какая доля дефектов не подтверждается и сколько появляется новых;
  3. этот закон применяется к дефектам целевого участка методом Монте-Карло; минимальная
     толщина сегмента усредняется по розыгрышам, а разброс розыгрышей даёт интервал P10–P90.

Неопределённость положения (сдвиг одометра между опорными точками известен неточно)
разыгрывается вместе с глубиной, поэтому прогноз сглаживается ровно там, где мы хуже знаем дистанцию.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .data import n_segments, seg_of


def match_defects(a: pd.DataFrame, b: pd.DataFrame, d_mm: float, tol_m: float = 0.08):
    """Сопоставляет дефекты пропуска B с дефектами пропуска A (колонка x2: дистанция A в шкале B).

    Расстояние считается по развёртке трубы: вдоль оси и по окружности.
    Возвращает (пары, новые дефекты B без пары, маска подтверждённых дефектов A).
    """
    arc = np.pi * d_mm / 1000.0 / 360.0  # метров на градус
    # Угол периодичен, 359° и 1° соседние. Копии A со сдвигом ±360° дают кратчайшую дугу.
    xa, ang = a["x2"].to_numpy(), a["ang"].to_numpy() % 360.0
    n_a = len(a)
    tree = cKDTree(np.c_[np.tile(xa, 3), np.concatenate([ang - 360.0, ang, ang + 360.0]) * arc])
    dist, j = tree.query(np.c_[b["x"].to_numpy(), (b["ang"].to_numpy() % 360.0) * arc], distance_upper_bound=tol_m)
    ok = np.isfinite(dist)
    j = np.where(ok, j % max(n_a, 1), 0)  # индекс исходного дефекта A
    pairs = pd.DataFrame({
        "dp_a": a["dp"].to_numpy()[j[ok]], "dp_b": b["dp"].to_numpy()[ok],
        "x_b": b["x"].to_numpy()[ok],
    })
    confirmed = np.zeros(len(a), dtype=bool)
    confirmed[j[ok]] = True
    return pairs, b[~ok], confirmed


@dataclass
class GrowthModel:
    """Эмпирический закон перехода «глубина в старом пропуске → глубина в новом»."""
    pools: dict = field(default_factory=dict)       # целая глубина A -> выборка приращений
    p_confirm: dict = field(default_factory=dict)   # целая глубина A -> доля подтверждённых
    new_rate: float = 0.0                           # новых дефектов на один старый
    new_dp: np.ndarray = field(default_factory=lambda: np.array([15.0]))
    n_pairs: int = 0

    @classmethod
    def fit(cls, pairs: pd.DataFrame, new_dp, dp_a_all, confirmed) -> "GrowthModel":
        m = cls(n_pairs=len(pairs))
        g = (pairs["dp_b"] - pairs["dp_a"]).to_numpy()
        da = pairs["dp_a"].to_numpy()
        dp_a_all, confirmed = np.asarray(dp_a_all, dtype=float), np.asarray(confirmed, dtype=bool)
        for d in range(0, 101):
            for w in (1.0, 2.0, 4.0, 8.0, 100.0):  # расширяем окно, пока не наберётся выборка
                sel = np.abs(da - d) <= w
                if sel.sum() >= 40:
                    break
            m.pools[d] = g[sel]
            for w in (1.0, 2.0, 4.0, 8.0, 100.0):
                sel = np.abs(dp_a_all - d) <= w
                if sel.sum() >= 60:
                    break
            m.p_confirm[d] = float(confirmed[sel].mean())
        m.new_rate = len(new_dp) / max(len(dp_a_all), 1)
        m.new_dp = np.asarray(new_dp, dtype=float)
        return m

    def simulate(self, x, dp, wt, length_m: float, wall_mm: float, n_sim: int = 400,
                 seed: int = 42, pos_sigma: float = 0.0, pos_bump=None,
                 repair_mask=None, repair_prob: float = 0.0, repair_groups=None) -> np.ndarray:
        """Розыгрыши минимальной толщины сегментов. Возвращает массив (n_sim, n_seg).

        pos_sigma: СКО ошибки положения в метрах в точке, где pos_bump = 1.
        repair_mask: дефекты на трубах, которые отчёт стартового пропуска предписал вырезать.
        repair_groups: номер трубы для каждого дефекта. Каждая такая труба в розыгрыше заменена
        новой независимо от других с вероятностью repair_prob, и дефектов на ней нет.
        """
        rng = np.random.default_rng(seed)
        x, dp, wt = (np.asarray(v, dtype=float) for v in (x, dp, wt))
        n_seg = n_segments(length_m)
        bump = np.zeros_like(x) if pos_bump is None else np.asarray(pos_bump, dtype=float)
        d_int = np.clip(np.rint(dp).astype(int), 0, 100)
        groups = {d: np.where(d_int == d)[0] for d in np.unique(d_int)}
        p_conf = np.array([self.p_confirm[d] for d in d_int])
        seg0 = seg_of(x, length_m)
        wt_seg = pd.Series(wt).groupby(seg0).median().reindex(range(n_seg)).interpolate(limit_direction="both")
        wt_seg = wt_seg.fillna(wall_mm).to_numpy()
        rep = np.zeros(len(x), dtype=bool) if repair_mask is None else np.asarray(repair_mask, dtype=bool)
        grp = np.zeros(len(x), dtype=int) if repair_groups is None else np.asarray(repair_groups)
        rep_groups = np.unique(grp[rep])
        rep_idx = {g: np.where(rep & (grp == g))[0] for g in rep_groups}
        out = np.empty((n_sim, n_seg))
        for s in range(n_sim):
            seg = seg_of(x + rng.normal(0.0, pos_sigma) * bump, length_m) if pos_sigma > 0 else seg0
            dp_new = np.empty_like(dp)
            for d, idx in groups.items():
                dp_new[idx] = d + rng.choice(self.pools[d], size=len(idx))
            rw = wt * (1.0 - np.clip(dp_new, 0.0, 95.0) / 100.0)
            alive = rng.random(len(x)) < p_conf
            present = np.ones(len(x), dtype=bool)
            u = rng.random(max(len(rep_groups), 1))   # разыгрывается всегда: поток случайных чисел не зависит от repair_prob
            for g, ug in zip(rep_groups, u):
                if ug < repair_prob:
                    present[rep_idx[g]] = False          # труба заменена: старых дефектов на ней нет
            alive &= present
            mins = np.full(n_seg, np.inf)
            np.minimum.at(mins, seg[alive], rw[alive])
            # новые дефекты появляются там, где уже есть коррозия
            counts = rng.poisson(self.new_rate * np.bincount(seg[present], minlength=n_seg))
            if counts.sum():
                seg_new = np.repeat(np.arange(n_seg), counts)
                rw_new = wt_seg[seg_new] * (1.0 - rng.choice(self.new_dp, size=len(seg_new)) / 100.0)
                np.minimum.at(mins, seg_new, rw_new)
            out[s] = np.where(np.isfinite(mins), mins, wt_seg)
        return out


def summarize(sims: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "segment_id": np.arange(sims.shape[1]),
        "thickness_pred": sims.mean(axis=0),
        "p10": np.quantile(sims, 0.10, axis=0),
        "p90": np.quantile(sims, 0.90, axis=0),
    })
