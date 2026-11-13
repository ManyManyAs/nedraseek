"""Сборка решения: от исходных файлов до прогноза, валидации и вспомогательных таблиц."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import align, case1, case2, data
from .validate import mae, pearson_r2, top_recall


def _offsets(vtd: pd.DataFrame) -> dict:
    """Смещение начала участка по трубопроводу: x_raw - x (одинаково во всех пропусках)."""
    return (vtd["x_raw"] - vtd["x"]).groupby(vtd["id"]).median().round(1).to_dict()


def _aligned_pair(V, pid, length, d_mm, date_a, date_b, tol):
    """Дефекты двух пропусков одного участка; A переведён в шкалу B и сопоставлен с B."""
    a = V[(V["id"] == pid) & (V["date"] == date_a)].copy()
    b = V[(V["id"] == pid) & (V["date"] == date_b)].copy()
    k0, _ = align.estimate_shift(a["x"], b["x"], 0, min(length, 150.0), 0.0, span=25.0, res=0.5)
    coarse = align.window_shifts(a["x"], b["x"], length, window=min(100.0, length), k0=k0, span=6.0, min_points=10)
    if coarse.empty:
        coarse = pd.DataFrame([{"lo": 0.0, "hi": length, "mid": length / 2, "shift": k0, "corr": np.nan}])
    a["x2"] = align.fine_align(a, b, length, coarse)
    pairs, new, confirmed = case1.match_defects(a, b, d_mm, tol)
    return {"a": a, "b": b, "coarse": coarse, "pairs": pairs, "new": new, "confirmed": confirmed, "L": length}


def _fit(parts) -> case1.GrowthModel:
    return case1.GrowthModel.fit(
        pd.concat([p["pairs"] for p in parts], ignore_index=True),
        np.concatenate([p["new"]["dp"].to_numpy() for p in parts]),
        np.concatenate([p["a"]["dp"].to_numpy() for p in parts]),
        np.concatenate([p["confirmed"] for p in parts]),
    )


def _subset(s: dict, lo: float, hi: float, inside: bool) -> dict:
    """Часть данных участка по дистанции в шкале B: внутри [lo, hi) или снаружи."""
    def m(x):
        inb = (x >= lo) & (x < hi)
        return inb if inside else ~inb
    ma = m(s["a"]["x2"]).to_numpy()
    return {"a": s["a"][ma], "confirmed": s["confirmed"][ma],
            "pairs": s["pairs"][m(s["pairs"]["x_b"]).to_numpy()], "new": s["new"][m(s["new"]["x"]).to_numpy()]}


def run_case1(cfg: dict, P: pd.DataFrame, V: pd.DataFrame) -> dict:
    c = cfg["case1"]
    tid, A, B = int(c["target_id"]), pd.Timestamp(c["start_run"]), pd.Timestamp(c["target_run"])
    seed, tol = cfg["seed"], c["match_tol_m"]
    pp = P.set_index("id")
    pipe_ids = P.loc[P["pipeline"] == pp.loc[tid, "pipeline"], "id"].tolist()
    acc = _offsets(V)
    L_t, wall_t, d_mm = pp.loc[tid, "length_m"], pp.loc[tid, "wall_mm"], pp.loc[tid, "d_mm"]

    # --- 1. Опорные участки: известны оба пропуска. Выравнивание и пары дефектов.
    anchors = {}
    for pid in pipe_ids:
        if pid != tid and {A, B} <= set(data.run_dates(V, pid)):
            anchors[pid] = _aligned_pair(V, pid, pp.loc[pid, "length_m"], d_mm, A, B, tol)
    if not anchors:
        raise RuntimeError("нет участков с обоими пропусками — закон роста оценить не на чем")

    # --- 2. Модель сдвига дистанций вдоль трубопровода (в координатах пропуска A).
    rows = []
    for pid, s in anchors.items():
        co = s["coarse"]
        med = co["shift"].median()
        for r in co.itertuples():
            good = (np.isnan(r.corr) or r.corr >= 0.7) and abs(r.shift - med) <= 2.0
            rows.append({"id": pid, "window_from": r.lo, "window_to": r.hi, "shift_m": r.shift, "corr": r.corr,
                         "raw_a": acc[pid] + r.mid - r.shift, "used": good})
    align_tbl = pd.DataFrame(rows)
    used = align_tbl[align_tbl["used"]]
    drift = align.LinearDrift(used["raw_a"], used["shift_m"])

    # --- 3. Проверка на опорных участках: прогноз B по A, закон роста оценён без проверяемой части.
    val_rows, facts, preds = [], [], {k: [] for k in ("raw", "pers", "q05", "mc", "p10", "p90")}
    for pid, s in anchors.items():
        L, wall = s["L"], pp.loc[pid, "wall_mm"]
        fact = data.segment_state(s["b"], L, wall)
        n = len(fact)
        mc, p10, p90 = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
        folds = [(0.0, L / 2), (L / 2, 1e12)] if n >= 20 else [(0.0, 1e12)]
        for lo, hi in folds:
            train = [_subset(s, lo, hi, inside=False)] + [o for q, o in anchors.items() if q != pid]
            train = [t for t in train if len(t["pairs"])]
            gm = _fit(train)
            sm = case1.summarize(gm.simulate(s["a"]["x2"], s["a"]["dp"], s["a"]["wt"], L, wall, n_sim=400, seed=seed))
            seg = (fact["segment_id"] * data.SEG_LEN >= lo) & (fact["segment_id"] * data.SEG_LEN < hi)
            mc[seg], p10[seg], p90[seg] = sm["thickness_pred"][seg], sm["p10"][seg], sm["p90"][seg]
        st_al = data.segment_state(s["a"], L, wall, x_col="x2")
        facts.append(fact["thk_min"].to_numpy())
        preds["raw"].append(data.segment_state(s["a"], L, wall)["thk_min"].to_numpy())
        preds["pers"].append(st_al["thk_min"].to_numpy())
        preds["q05"].append(st_al["rw_q05"].to_numpy())
        preds["mc"].append(mc); preds["p10"].append(p10); preds["p90"].append(p90)
    y = np.concatenate(facts)
    pr = {k: np.concatenate(v) for k, v in preds.items()}
    horizon = round((B - A).days / 365.25, 2)
    for name, key, risk in (("Прошлый ВТД без выравнивания дистанций", "raw", "raw"),
                            ("Прошлый ВТД, дистанции выровнены", "pers", "pers"),
                            ("5-й перцентиль толщины по дефектам сегмента", "q05", "q05"),
                            ("Модель роста дефектов (итоговая)", "mc", "p10")):
        val_rows.append({"method": name, "horizon_years": horizon, "n_segments": len(y),
                         "r2": pearson_r2(y, pr[key]), "mae_mm": mae(y, pr[key]),
                         "top10_recall": top_recall(y, pr[risk])})
    coverage = float(np.mean((y >= pr["p10"]) & (y <= pr["p90"])))
    validation = pd.DataFrame(val_rows)

    # --- 4. Точность по горизонтам: все пары пропусков на участках трубопровода.
    hz = []
    for pid in pipe_ids:
        ds = data.run_dates(V, pid)
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                L, wall = pp.loc[pid, "length_m"], pp.loc[pid, "wall_mm"]
                if data.n_segments(L) < 20:
                    continue
                s = _aligned_pair(V, pid, L, d_mm, ds[i], ds[j], tol)
                f = data.segment_state(s["b"], L, wall)["thk_min"]
                hz.append({"id": pid, "from_run": pd.Timestamp(ds[i]).date(), "to_run": pd.Timestamp(ds[j]).date(),
                           "horizon_years": round((ds[j] - ds[i]).days / 365.25, 2), "n_segments": len(f),
                           "r2_prev_run_aligned": pearson_r2(f, data.segment_state(s["a"], L, wall, x_col="x2")["thk_min"]),
                           "r2_prev_run_raw": pearson_r2(f, data.segment_state(s["a"], L, wall)["thk_min"])})
    horizons = pd.DataFrame(hz).sort_values(["horizon_years", "id"])

    # --- 5. Прогноз целевого участка. Источник — дефекты пропуска A всего трубопровода:
    #        после перевода в шкалу B часть дефектов соседних участков попадает в целевой.
    src = V[(V["id"].isin(pipe_ids)) & (V["date"] == A)].copy()
    raw_b = src["x_raw"] + drift(src["x_raw"])
    src["xt"] = raw_b - acc[tid]
    src = src[(src["xt"] >= 0) & (src["xt"] < L_t)]
    bump = 0.25 + 0.75 * np.sin(np.pi * src["xt"].to_numpy() / L_t)  # у опорных концов сдвиг известен лучше
    gm = _fit(list(anchors.values()))
    sims = gm.simulate(src["xt"], src["dp"], src["wt"], L_t, wall_t, n_sim=c["n_sim"], seed=seed,
                       pos_sigma=c["pos_sigma_m"], pos_bump=bump)
    pred = case1.summarize(sims)
    pred.insert(0, "id", tid)
    prev = data.segment_state(src, L_t, wall_t, x_col="xt")
    pred["thk_start_run"] = prev["thk_min"].to_numpy()
    pred["n_defects_start_run"] = prev["n_def"].to_numpy()
    pred["wall_mm"] = prev["wt"].to_numpy()
    return {"pred": pred, "validation": validation, "coverage": coverage, "horizons": horizons,
            "alignment": align_tbl, "model": gm,
            "drift_ends": (float(drift(acc[tid])), float(drift(acc[tid] + L_t)))}


def run_case2(cfg: dict, P: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    c = cfg["case2"]
    row = P.set_index("id").loc[int(c["target_id"])]
    pred = case2.predict(profile, int(c["target_id"]), row["length_m"], row["wall_mm"], c["mean_loss_frac"], c["spread"])
    pred.insert(0, "id", int(c["target_id"]))
    pred["wall_mm"] = row["wall_mm"]
    return pred


def top_segments(pred: pd.DataFrame, frac: float = 0.10) -> pd.DataFrame:
    """Ранжированный список самых опасных сегментов: по нижней границе толщины относительно стенки."""
    t = pred.assign(risk=pred["p10"] / pred["wall_mm"]).sort_values("risk")
    t = t.head(max(1, int(round(len(pred) * frac)))).copy()
    t.insert(0, "rank", np.arange(1, len(t) + 1))
    t["dist_from_m"] = t["segment_id"] * data.SEG_LEN
    t["dist_to_m"] = t["dist_from_m"] + data.SEG_LEN
    return t[["rank", "id", "segment_id", "dist_from_m", "dist_to_m", "thickness_pred", "p10", "p90", "wall_mm"]]
