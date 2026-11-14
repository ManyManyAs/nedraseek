"""Сборка решения: от исходных файлов до прогноза, валидации и вспомогательных таблиц."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import align, case1, case2, data
from .validate import mae, pearson_r2, top_recall

CUT_OUT = "Врезка катушки"   # рекомендация отчёта ВТД: вырезать трубу с дефектом и врезать новую


# ---------------------------------------------------------------- геометрия

def _offsets(vtd: pd.DataFrame, date) -> dict:
    """Смещение начала участка по трубопроводу (x_raw - x) в пропуске date.

    Берётся только стартовый пропуск, чтобы более поздние выгрузки не меняли систему отсчёта.
    """
    v = vtd[vtd["date"] == pd.Timestamp(date)]
    return (v["x_raw"] - v["x"]).groupby(v["id"]).median().round(1).to_dict()


def joint_shifts(J: pd.DataFrame, date_a, date_b, ids) -> pd.DataFrame:
    """Сдвиг дистанций B − A по трубам, которые есть в обоих пропусках трубопровода."""
    a = J[(J["date"] == pd.Timestamp(date_a)) & J["id"].isin(ids)]
    b = J[(J["date"] == pd.Timestamp(date_b)) & J["id"].isin(ids)]
    m = a.merge(b, on="sec", suffixes=("_a", "_b"))
    m["shift"] = m["js_b"] - m["js_a"]
    return m.sort_values("js_a").reset_index(drop=True)


def _shift_fn(js: pd.DataFrame):
    """Сдвиг как функция дистанции пропуска A: линейная интерполяция между началами общих труб."""
    xa, sh = js["js_a"].to_numpy(dtype=float), js["shift"].to_numpy(dtype=float)
    return lambda x_raw: np.interp(np.asarray(x_raw, dtype=float), xa, sh)


def _anchor(V, js, pid, length, d_mm, acc, date_a, date_b, tol, exclude=None) -> dict:
    """Опорный участок: дефекты A в шкале B и их пары с дефектами B.

    exclude = (lo, hi) в метрах шкалы B внутри участка: трубы, начинающиеся в этом интервале,
    в оценке сдвига не участвуют. Так проверяемый блок не подсказывает собственное положение.
    """
    a = V[(V["id"] == pid) & (V["date"] == date_a)].copy()
    b = V[(V["id"] == pid) & (V["date"] == date_b)].copy()
    if exclude is not None:
        pos_b = js["js_b"] - acc
        js = js[~((pos_b >= exclude[0]) & (pos_b < exclude[1]))]
    a["x2"] = a["x_raw"].to_numpy() + _shift_fn(js)(a["x_raw"]) - acc
    pairs, new, confirmed = case1.match_defects(a, b, d_mm, tol)
    return {"a": a, "b": b, "pairs": pairs, "new": new, "confirmed": confirmed, "L": length}


def _xcorr_pair(V, pid, length, date_a, date_b) -> dict:
    """Выравнивание по плотности дефектов — для пар пропусков, где номера труб не совпадают (2018 год).

    Использует пропуск B, поэтому годится только для диагностики, не для оценки точности прогноза.
    """
    a = V[(V["id"] == pid) & (V["date"] == date_a)].copy()
    b = V[(V["id"] == pid) & (V["date"] == date_b)].copy()
    k0, _ = align.estimate_shift(a["x"], b["x"], 0, min(length, 150.0), 0.0, span=25.0, res=0.5)
    coarse = align.window_shifts(a["x"], b["x"], length, window=min(250.0, length), k0=k0, span=6.0, min_points=10)
    if coarse.empty:
        coarse = pd.DataFrame([{"lo": 0.0, "hi": length, "mid": length / 2, "shift": k0, "corr": np.nan}])
    a["x2"] = align.fine_align(a, b, length, coarse)
    return {"a": a, "b": b, "coarse": coarse, "L": length}


def drift_linearity(V, pid, length, date_a, date_b) -> pd.DataFrame:
    """Насколько сдвиг внутри участка отклоняется от прямой между его концами.

    Считается на паре пропусков, известной по всей длине участка. Разброс отклонений — оценка
    ошибки, которую даёт линейная интерполяция сдвига, когда целевой пропуск внутри участка скрыт.
    """
    co = _xcorr_pair(V, pid, length, date_a, date_b)["coarse"]
    co = co[co["corr"] >= 0.3].reset_index(drop=True)
    line = np.interp(co["mid"], [co["mid"].iloc[0], co["mid"].iloc[-1]], [co["shift"].iloc[0], co["shift"].iloc[-1]])
    return co.assign(linear=line, residual=co["shift"] - line)[["lo", "hi", "shift", "corr", "linear", "residual"]]


def _bump(x, length: float) -> np.ndarray:
    """Форма ошибки положения: у концов участка, где сдвиг измерен, меньше; в середине — больше.

    Ошибка одна на весь розыгрыш (общий уход одометра), а не независимая для каждого дефекта.
    """
    return 0.25 + 0.75 * np.sin(np.pi * np.clip(np.asarray(x, dtype=float), 0, length) / length)


# ---------------------------------------------------------------- закон роста

def _fit(parts) -> case1.GrowthModel:
    return case1.GrowthModel.fit(
        pd.concat([p["pairs"] for p in parts], ignore_index=True),
        np.concatenate([p["new"]["dp"].to_numpy() for p in parts]),
        np.concatenate([p["a"]["dp"].to_numpy() for p in parts]),
        np.concatenate([p["confirmed"] for p in parts]),
    )


def _outside(s: dict, lo: float, hi: float) -> dict:
    """Данные опорного участка вне интервала [lo, hi) шкалы B — для обучения закона роста."""
    out = lambda x: ~((x >= lo) & (x < hi))  # noqa: E731
    ma = out(s["a"]["x2"]).to_numpy()
    return {"a": s["a"][ma], "confirmed": s["confirmed"][ma],
            "pairs": s["pairs"][out(s["pairs"]["x_b"]).to_numpy()], "new": s["new"][out(s["new"]["x"]).to_numpy()]}


def _z(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / (v.std() if v.std() > 0 else 1.0)


def blend(mc: np.ndarray, q05: np.ndarray, w: float) -> np.ndarray:
    """Смесь модели роста и 5-го перцентиля в стандартизованных единицах, в шкале модели (мм)."""
    return mc.mean() + mc.std() * ((1 - w) * _z(mc) + w * _z(q05))


# ---------------------------------------------------------------- кейс 1

def run_case1(cfg: dict, P: pd.DataFrame, V: pd.DataFrame, J: pd.DataFrame | None = None) -> dict:
    c = cfg["case1"]
    tid, A, B = int(c["target_id"]), pd.Timestamp(c["start_run"]), pd.Timestamp(c["target_run"])
    seed, tol = cfg["seed"], c["match_tol_m"]
    J = data.load_joints(cfg["paths"]["raw"]) if J is None else J
    pp = P.set_index("id")
    pipe_ids = P.loc[P["pipeline"] == pp.loc[tid, "pipeline"], "id"].tolist()
    acc = _offsets(V, A)
    L_t, wall_t, d_mm = pp.loc[tid, "length_m"], pp.loc[tid, "wall_mm"], pp.loc[tid, "d_mm"]
    js = joint_shifts(J, A, B, pipe_ids)
    if len(js) < 2:
        raise RuntimeError("нет общих труб в двух пропусках — сдвиг дистанций оценить не на чем")

    # --- 1. Опорные участки: известны оба пропуска.
    anchor_ids = [p for p in pipe_ids if p != tid and {A, B} <= set(data.run_dates(V, p))]
    anchors = {p: _anchor(V, js, p, pp.loc[p, "length_m"], d_mm, acc[p], A, B, tol) for p in anchor_ids}
    if not anchors:
        raise RuntimeError("нет участков с обоими пропусками — закон роста оценить не на чем")

    # --- 2. Независимая проверка по блокам на самом длинном опорном участке.
    #     Блок сегментов скрыт целиком вместе с зазором: его пропуск B не участвует ни в оценке
    #     сдвига (трубы блока исключены, сдвиг интерполируется по соседним), ни в законе роста.
    vid = max(anchors, key=lambda p: data.n_segments(anchors[p]["L"]))
    Lv, wall_v = anchors[vid]["L"], pp.loc[vid, "wall_mm"]
    fact = data.segment_state(anchors[vid]["b"], Lv, wall_v)
    n = len(fact)
    gap = c["val_gap_m"]
    P_ = {k: np.full(n, np.nan) for k in ("raw", "pers", "q05", "mc", "p10", "p90", "blend")}
    block_of = np.zeros(n, dtype=int)
    for bi, blk in enumerate(np.array_split(np.arange(n), c["val_blocks"])):
        lo, hi = blk[0] * data.SEG_LEN, (blk[-1] + 1) * data.SEG_LEN
        s = _anchor(V, js, vid, Lv, d_mm, acc[vid], A, B, tol, exclude=(lo - gap, hi + gap))
        train = [_outside(s, lo - gap, hi + gap)] + [anchors[q] for q in anchors if q != vid]
        gm = _fit([t for t in train if len(t["pairs"])])
        sm = case1.summarize(gm.simulate(s["a"]["x2"], s["a"]["dp"], s["a"]["wt"], Lv, wall_v, n_sim=c["n_sim"],
                                         seed=seed, pos_sigma=c["val_pos_sigma_m"], pos_bump=np.ones(len(s["a"]))))
        st = data.segment_state(s["a"], Lv, wall_v, x_col="x2")
        P_["raw"][blk] = data.segment_state(s["a"], Lv, wall_v)["thk_min"].to_numpy()[blk]
        P_["pers"][blk], P_["q05"][blk] = st["thk_min"].to_numpy()[blk], st["rw_q05"].to_numpy()[blk]
        P_["mc"][blk], P_["p10"][blk], P_["p90"][blk] = sm["thickness_pred"][blk], sm["p10"][blk], sm["p90"][blk]
        # Смесь нормируется по прогнозам этого фолда на всём участке, как в итоговом прогнозе.
        # Метки скрытого блока в эти статистики не попадают.
        P_["blend"][blk] = blend(sm["thickness_pred"].to_numpy(), st["rw_q05"].to_numpy(), c["blend_w"])[blk]
        block_of[blk] = bi
    y = fact["thk_min"].to_numpy()
    horizon = round((B - A).days / 365.25, 2)
    methods = (("Прошлый ВТД без выравнивания дистанций", "raw", "raw"),
               ("Прошлый ВТД, дистанции выровнены", "pers", "pers"),
               ("5-й перцентиль толщины по дефектам сегмента", "q05", "q05"),
               ("Модель роста дефектов", "mc", "p10"),
               ("Модель роста + 5-й перцентиль (итоговая)", "blend", "blend"))
    validation = pd.DataFrame([{"method": name, "horizon_years": horizon, "n_segments": n,
                                "r2": pearson_r2(y, P_[k]), "mae_mm": mae(y, P_[k]),
                                "top10_recall": top_recall(y, P_[r])} for name, k, r in methods])
    by_block = pd.DataFrame([{"block": bi, "segments": f"{np.where(block_of == bi)[0][0]}–{np.where(block_of == bi)[0][-1]}",
                              **{k: pearson_r2(y[block_of == bi], P_[k][block_of == bi]) for k in ("pers", "q05", "mc", "blend")}}
                             for bi in range(c["val_blocks"])])
    coverage = float(np.mean((y >= P_["p10"]) & (y <= P_["p90"])))
    val_points = pd.DataFrame({"id": vid, "segment_id": np.arange(n), "block": block_of, "fact": y,
                               **{f"pred_{k}": P_[k] for k in ("raw", "pers", "q05", "mc", "blend", "p10", "p90")}})

    # --- 3. Точность «прошлого ВТД» по горизонтам (диагностика: выравнивание видит пропуск B).
    hz = []
    for pid in pipe_ids:
        ds = data.run_dates(V, pid)
        L, wall = pp.loc[pid, "length_m"], pp.loc[pid, "wall_mm"]
        if data.n_segments(L) < 20:
            continue
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                s = _xcorr_pair(V, pid, L, ds[i], ds[j])
                f = data.segment_state(s["b"], L, wall)["thk_min"]
                hz.append({"id": pid, "from_run": pd.Timestamp(ds[i]).date(), "to_run": pd.Timestamp(ds[j]).date(),
                           "horizon_years": round((ds[j] - ds[i]).days / 365.25, 2), "n_segments": len(f),
                           "r2_prev_run_aligned": pearson_r2(f, data.segment_state(s["a"], L, wall, x_col="x2")["thk_min"]),
                           "r2_prev_run_raw": pearson_r2(f, data.segment_state(s["a"], L, wall)["thk_min"])})
    horizons = pd.DataFrame(hz).sort_values(["horizon_years", "id"])

    # --- 4. Ошибка линейной интерполяции сдвига: по предыдущей паре пропусков на самом целевом участке.
    prev_runs = [d for d in data.run_dates(V, tid) if d < A]
    lin = drift_linearity(V, tid, L_t, prev_runs[-1], A) if prev_runs else pd.DataFrame()
    pos_sigma = float(np.sqrt(np.mean(lin["residual"] ** 2))) if len(lin) >= 4 else c["pos_sigma_m"]
    pos_sigma = max(pos_sigma, c["pos_sigma_m"])

    # --- 5. Прогноз целевого участка. Источник — дефекты пропуска A всего трубопровода:
    #        после перевода в шкалу B часть дефектов соседних участков попадает в целевой.
    src = V[(V["id"].isin(pipe_ids)) & (V["date"] == A)].copy()
    src["xt"] = src["x_raw"].to_numpy() + _shift_fn(js)(src["x_raw"]) - acc[tid]
    src = src[(src["xt"] >= 0) & (src["xt"] < L_t)]
    # Трубы, которые отчёт пропуска A предписал вырезать: с вероятностью repair_prob они заменены.
    cut_secs = sorted(src.loc[src["rec_repair"] == CUT_OUT, "sec"].unique())
    in_cut = src["sec"].isin(cut_secs).to_numpy()
    gm = _fit(list(anchors.values()))
    sims = gm.simulate(src["xt"], src["dp"], src["wt"], L_t, wall_t, n_sim=c["n_sim"], seed=seed,
                       pos_sigma=pos_sigma, pos_bump=_bump(src["xt"], L_t),
                       repair_mask=in_cut, repair_prob=c["repair_prob"], repair_groups=src["sec"].to_numpy())
    pred = case1.summarize(sims)
    prev = data.segment_state(src, L_t, wall_t, x_col="xt")
    # Чувствительность к сценарию ремонта: прогноз модели при «ремонта не было» и «заменены все три».
    for tag, prob in (("if_no_repair", 0.0), ("if_repaired", 1.0)):
        pred[tag] = case1.summarize(gm.simulate(src["xt"], src["dp"], src["wt"], L_t, wall_t, n_sim=max(200, c["n_sim"] // 5),
                                                seed=seed, pos_sigma=pos_sigma, pos_bump=_bump(src["xt"], L_t),
                                                repair_mask=in_cut, repair_prob=prob, repair_groups=src["sec"].to_numpy()))["thickness_pred"]
    pred["model_only"] = pred["thickness_pred"]
    pred["thickness_pred"] = blend(pred["model_only"].to_numpy(), prev["rw_q05"].to_numpy(), c["blend_w"])
    # На трубах под замену 5-й перцентиль описывает трубу, которой, скорее всего, уже нет: оставляем модель.
    cut_seg = np.unique(data.seg_of(src.loc[in_cut, "xt"], L_t)) if in_cut.any() else np.array([], dtype=int)
    touched = np.unique(np.concatenate([cut_seg - 1, cut_seg, cut_seg + 1])) if len(cut_seg) else cut_seg
    touched = touched[(touched >= 0) & (touched < len(pred))]
    pred.loc[touched, "thickness_pred"] = pred.loc[touched, "model_only"]
    pred["thickness_pred"] = pred["thickness_pred"].clip(lower=0.0)
    pred.insert(0, "id", tid)
    pred["thk_start_run"] = prev["thk_min"].to_numpy()
    pred["n_defects_start_run"] = prev["n_def"].to_numpy()
    pred["wall_mm"] = prev["wt"].to_numpy()
    pred["repair_flag"] = np.isin(np.arange(len(pred)), cut_seg)
    repairs = (src[in_cut].groupby("sec").agg(dist_from_m=("xt", "min"), dist_to_m=("xt", "max"), defects=("dp", "size"),
                                              max_depth_pct=("dp", "max"), min_thk_mm=("rw", "min")).reset_index())
    shift_ends = _shift_fn(js)([acc[tid], acc[tid] + L_t])
    return {"pred": pred, "validation": validation, "by_block": by_block, "val_points": val_points,
            "coverage": coverage, "horizons": horizons, "joint_shifts": js, "linearity": lin, "repairs": repairs,
            "model": gm, "pos_sigma": pos_sigma, "val_section": vid,
            "drift_ends": (float(shift_ends[0]), float(shift_ends[1]))}


# ---------------------------------------------------------------- кейс 2 и общее

def run_case2(cfg: dict, P: pd.DataFrame, profile: pd.DataFrame, V: pd.DataFrame) -> dict:
    c = cfg["case2"]
    tid = int(c["target_id"])
    row = P.set_index("id").loc[tid]
    ref, mean_loss, high_loss = case2.reference_table(P, V, profile, exclude_ids={tid}, window_m=c["smooth_window_m"])
    pred = case2.predict(profile, tid, row["length_m"], row["wall_mm"], mean_loss, high_loss, c["spread"], c["smooth_window_m"])
    pred.insert(0, "id", tid)
    pred["wall_mm"] = row["wall_mm"]
    by_pipe = (ref.groupby("pipeline").apply(lambda g: pd.Series({
        "sections": len(g), "n_segments": int(g["n_segments"].sum()),
        "r2_weighted": float(np.average(g["r2"], weights=g["n_segments"])),
        "low_is_thinner_sections": int(g["low_is_thinner"].sum())}), include_groups=False).reset_index())
    return {"pred": pred, "reference": ref, "by_pipeline": by_pipe, "mean_loss": mean_loss, "high_loss": high_loss}


def top_segments(pred: pd.DataFrame, length_m: float, frac: float = 0.10) -> pd.DataFrame:
    """Ранжированный список самых опасных сегментов: по нижней границе толщины относительно стенки."""
    t = pred.assign(risk=pred["p10"] / pred["wall_mm"]).sort_values(["risk", "segment_id"])
    t = t.head(max(1, int(round(len(pred) * frac)))).copy()
    t.insert(0, "rank", np.arange(1, len(t) + 1))
    t["dist_from_m"] = t["segment_id"] * data.SEG_LEN
    t["dist_to_m"] = np.minimum(t["dist_from_m"] + data.SEG_LEN, length_m)
    return t[["rank", "id", "segment_id", "dist_from_m", "dist_to_m", "thickness_pred", "p10", "p90", "wall_mm"]]
