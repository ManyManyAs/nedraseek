"""Разбор кейса 1 после результата на скрытых данных.

Запуск из корня:  python scripts/postmortem_case1.py
Пишет таблицы в outputs/postmortem/. Основной пайплайн и result.csv не трогает.

Что считает:
  1. На чём держится r² 0,76 проверки (два сегмента без дефектов, бутстреп).
  2. Перенос закона роста между опорными участками (смещение уровня и MAE).
  3. Проверка «сдвиг известен только на концах» на 1250011604.
  4. Сдвиг 2018 -> 2022 по швам вдоль целевого участка и та же проверка на этой паре.
  5. Двойник целевого участка: факт 2023 разыгрывается законом роста из дефектов 2022,
     сверху накладываются ошибка положения и замены труб. Показывает, какие сценарии
     дают r² и MAE, похожие на скрытый результат.
  6. Варианты правок на тех же сценариях.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipetwin import case1, data, pipeline as pl  # noqa: E402
from pipetwin.validate import mae, pearson_r2  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
OUT = ROOT / "outputs" / "postmortem"
OUT.mkdir(parents=True, exist_ok=True)

cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
c = cfg["case1"]
RAW = ROOT / cfg["paths"]["raw"]
P, V, J = data.load_passport(RAW), data.load_vtd(RAW), data.load_joints(RAW)
TID, A, B = int(c["target_id"]), pd.Timestamp(c["start_run"]), pd.Timestamp(c["target_run"])
D18 = pd.Timestamp("2018-09-23")
pp = P.set_index("id")
PIPE = P.loc[P["pipeline"] == pp.loc[TID, "pipeline"], "id"].tolist()
ACC = pl._offsets(V, A)
L_T, WALL_T, D_MM = pp.loc[TID, "length_m"], pp.loc[TID, "wall_mm"], pp.loc[TID, "d_mm"]
JS = pl.joint_shifts(J, A, B, PIPE)
VID, SMALL = 1250011604, 1250009490
LV, WALL_V = pp.loc[VID, "length_m"], pp.loc[VID, "wall_mm"]
ANC = {p: pl._anchor(V, JS, p, pp.loc[p, "length_m"], D_MM, ACC[p], A, B, c["match_tol_m"]) for p in PIPE if p != TID}
GM = pl._fit(list(ANC.values()))


def save(df: pd.DataFrame, name: str, index: bool = False) -> None:
    df.to_csv(OUT / name, index=index)
    print(f"\n--- {name}")
    print(df.round(3).to_string(index=index))


# ---------------------------------------------------------------- 1. рычаги проверки

def leverage() -> None:
    val = pd.read_csv(ROOT / "outputs" / "validation_case1_points.csv")
    y = val["fact"].to_numpy()
    clean = y >= WALL_V - 0.01
    rows = []
    for k in ("raw", "pers", "q05", "mc", "blend"):
        v = val[f"pred_{k}"].to_numpy()
        rows.append({"method": k, "r2_all": pearson_r2(y, v), "mae_all": mae(y, v),
                     "r2_no_clean": pearson_r2(y[~clean], v[~clean]), "mae_no_clean": mae(y[~clean], v[~clean])})
    rows.append({"method": "const_median", "r2_all": np.nan, "mae_all": np.abs(y - np.median(y)).mean(),
                 "r2_no_clean": np.nan, "mae_no_clean": np.abs(y[~clean] - np.median(y[~clean])).mean()})
    ind = (val["pred_pers"].to_numpy() >= WALL_V - 0.01).astype(float)
    rows.append({"method": "flag_no_defects_2022", "r2_all": pearson_r2(y, ind), "mae_all": np.nan,
                 "r2_no_clean": np.nan, "mae_no_clean": np.nan})
    save(pd.DataFrame(rows), "validation_leverage.csv")
    rng, v = np.random.default_rng(0), val["pred_blend"].to_numpy()
    bs = [pearson_r2(y[i], v[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(4000))]
    print("бутстреп r² итога на 50 сегментах, 5/50/95 %:", np.nanpercentile(bs, [5, 50, 95]).round(2),
          "| сегменты без дефектов в факте:", np.where(clean)[0].tolist())


# ---------------------------------------------------------------- 2. перенос закона роста

def law_transfer() -> None:
    g_small, g_big = pl._fit([ANC[SMALL]]), pl._fit([ANC[VID]])
    rows = [{"depth_pct": d, "growth_small": g_small.pools[d].mean(), "growth_val": g_big.pools[d].mean(),
             "confirm_small": g_small.p_confirm[d], "confirm_val": g_big.p_confirm[d]} for d in (10, 12, 15, 20, 25, 30)]
    save(pd.DataFrame(rows), "growth_law_by_section.csv")
    s = ANC[VID]
    fact = data.segment_state(s["b"], LV, WALL_V)["thk_min"].to_numpy()
    rows = []
    for name, g in ((f"закон с {SMALL}", g_small), (f"закон с {VID}", g_big)):
        mc = g.simulate(s["a"]["x2"], s["a"]["dp"], s["a"]["wt"], LV, WALL_V, n_sim=300, seed=1,
                        pos_sigma=0.25, pos_bump=np.ones(len(s["a"]))).mean(axis=0)
        rows.append({"law": name, "applied_to": VID, "bias_mm": mc.mean() - fact.mean(), "mae": mae(fact, mc),
                     "mae_bias_removed": mae(fact, mc - mc.mean() + fact.mean()), "r2": pearson_r2(fact, mc)})
    save(pd.DataFrame(rows), "growth_law_transfer.csv")


# ---------------------------------------------------------------- 3. сдвиг только на концах

def ends_only() -> None:
    fact = data.segment_state(ANC[VID]["b"], LV, WALL_V)["thk_min"].to_numpy()
    clean = fact >= WALL_V - 0.01
    first, last = JS["sec"][JS["id_b"] == VID].min(), JS["sec"][JS["id_b"] == VID].max()
    ends = JS[(JS["sec"] <= first) | (JS["sec"] >= last)]
    a = V[(V["id"] == VID) & (V["date"] == A)].copy()
    sh_true, sh_lin = pl._shift_fn(JS)(a["x_raw"]), pl._shift_fn(ends)(a["x_raw"])
    e = np.abs(sh_lin - sh_true)
    print(f"\nошибка положения при прямой между концами {VID}: средняя {e.mean():.2f} м, макс {e.max():.2f} м")
    a["x_ends"], a["x_true"] = a["x_raw"] + sh_lin - ACC[VID], a["x_raw"] + sh_true - ACC[VID]
    g_small = pl._fit([ANC[SMALL]])
    rows = []
    for cond, xcol, g in (("сдвиг на концах, закон с соседнего участка", "x_ends", g_small),
                          ("сдвиг на концах, закон со всех опорных", "x_ends", GM),
                          ("сдвиг по всем швам, закон со всех опорных", "x_true", GM)):
        st = data.segment_state(a, LV, WALL_V, x_col=xcol)
        cand = {"прошлый ВТД": st["thk_min"].to_numpy(), "5-й перцентиль": st["rw_q05"].to_numpy()}
        for ps in (0.25, 1.0, 3.0):
            mc = g.simulate(a[xcol], a["dp"], a["wt"], LV, WALL_V, n_sim=500, seed=42, pos_sigma=ps,
                            pos_bump=np.ones(len(a))).mean(axis=0)
            cand[f"модель, σ={ps} м"] = mc
            cand[f"смесь w=0.5, σ={ps} м"] = pl.blend(mc, st["rw_q05"].to_numpy(), 0.5)
        for name, v in cand.items():
            rows.append({"conditions": cond, "method": name, "r2": pearson_r2(fact, v), "mae": mae(fact, v),
                         "r2_no_clean": pearson_r2(fact[~clean], v[~clean]), "mae_no_clean": mae(fact[~clean], v[~clean])})
    save(pd.DataFrame(rows), "ends_only_validation.csv")


# ---------------------------------------------------------------- 4. пара 2018 -> 2022 по швам

def joints_2018_2022() -> pd.DataFrame:
    """Сопоставление швов 2018 и 2022 годов слежением: номера труб не совпадают, но шаг труб около 10 м,
    а сдвиг меняется плавно, поэтому ближайший шов в пределах 1,2 м от ожидаемого места однозначен."""
    raw = pd.read_csv(RAW / data.FILES["vtd"], low_memory=False,
                      usecols=["id", "date", "Distance_raw", "Section Number", "Relative Distance", "Pipe Length"])
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw[raw["id"].isin(PIPE)].dropna(subset=["Section Number", "Relative Distance"])
    raw["js"] = raw["Distance_raw"] - raw["Relative Distance"]
    j = raw.groupby(["date", "Section Number"]).agg(js=("js", "median"), pl=("Pipe Length", "median")).reset_index()
    a, b = j[j["date"] == D18].sort_values("js"), j[j["date"] == A].sort_values("js")
    jb = np.unique(np.round(np.concatenate([b["js"].to_numpy(), (b["js"] + b["pl"]).dropna().to_numpy()]), 2))
    jb = jb[np.r_[True, np.diff(jb) > 0.15]]
    # стартовый сдвиг: по первой трубе, чья длина узнаётся в обоих пропусках
    sh, rows = 6.6, []
    for x in a["js"].to_numpy():
        i = np.abs(jb - (x - sh)).argmin()
        if abs(jb[i] - (x - sh)) < 1.2:
            sh = x - jb[i]
            rows.append({"js18": x, "js22": jb[i], "shift_22_minus_18": -sh})
    m = pd.DataFrame(rows)
    t = m[(m["js22"] >= ACC[TID]) & (m["js22"] < ACC[TID] + L_T)].copy()
    line = np.interp(t["js22"], [t["js22"].iloc[0], t["js22"].iloc[-1]],
                     [t["shift_22_minus_18"].iloc[0], t["shift_22_minus_18"].iloc[-1]])
    t["residual"] = t["shift_22_minus_18"] - line
    print(f"\nсдвиг 2018 -> 2022 по {len(t)} швам целевого участка: от {t['shift_22_minus_18'].iloc[0]:.1f} до "
          f"{t['shift_22_minus_18'].iloc[-1]:.1f} м; отклонение от прямой: СКО {np.sqrt((t['residual'] ** 2).mean()):.2f} м, "
          f"макс {t['residual'].abs().max():.2f} м")
    t.to_csv(OUT / "joint_shift_2018_2022.csv", index=False)

    a18 = V[V["id"].isin(PIPE) & (V["date"] == D18)].copy()
    fact = data.segment_state(V[(V["id"] == TID) & (V["date"] == A)], L_T, WALL_T)["thk_min"].to_numpy()
    sh_full = np.interp(a18["x_raw"], m["js18"], m["shift_22_minus_18"])
    sh_ends = np.interp(a18["x_raw"], [t["js18"].iloc[0], t["js18"].iloc[-1]],
                        [t["shift_22_minus_18"].iloc[0], t["shift_22_minus_18"].iloc[-1]])
    rows = []
    for name, shv in (("без совмещения", 0.0 * sh_full), ("сдвиг только на концах", sh_ends), ("сдвиг по всем швам", sh_full)):
        aa = a18.assign(xt=a18["x_raw"] + shv - ACC[TID])
        aa = aa[(aa["xt"] >= 0) & (aa["xt"] < L_T)]
        st = data.segment_state(aa, L_T, WALL_T, x_col="xt")
        rows.append({"alignment": name, "r2_prev_run": pearson_r2(fact, st["thk_min"]), "r2_q05": pearson_r2(fact, st["rw_q05"])})
    save(pd.DataFrame(rows), "ends_only_2018_2022_target.csv")
    return m


def replaced_between_2018_2022(m: pd.DataFrame) -> None:
    """Какие трубы с дефектами 2018 года к 2022 году стали чистыми (не больше двух записей)."""
    raw = pd.read_csv(RAW / data.FILES["vtd"], low_memory=False)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw[raw["id"].isin(PIPE)]
    ml = raw["Residual Wall Thickness"].notna()
    d18 = raw[(raw["date"] == D18) & ml].copy()
    d18["dp"] = d18["Defect Depth in %"] * 100
    d18["x22"] = d18["Distance_raw"] + np.interp(d18["Distance_raw"], m["js18"], m["shift_22_minus_18"])
    d22 = raw[raw["date"] == A].copy()
    d22["js"] = d22["Distance_raw"] - d22["Relative Distance"]
    d22["ml"] = d22["Residual Wall Thickness"].notna()
    p22 = d22.groupby("Section Number").agg(js=("js", "median"), pl=("Pipe Length", "median"), n=("ml", "sum")).sort_values("js")
    idx = np.clip(np.searchsorted(p22["js"].to_numpy(), d18["x22"].to_numpy()) - 1, 0, len(p22) - 1)
    inside = d18["x22"].to_numpy() < p22["js"].to_numpy()[idx] + p22["pl"].to_numpy()[idx]
    d18["n22"] = np.where(inside, p22["n"].to_numpy()[idx], 0)     # вне труб с записями 2022: труба без единой записи
    t = d18[(d18["x22"] >= ACC[TID]) & (d18["x22"] < ACC[TID] + L_T)]
    g = t.groupby("Section Number").agg(dpmax=("dp", "max"), n22=("n22", "median"))
    g["clean_2022"] = g["n22"] <= 2
    g["depth_bin"] = pd.cut(g["dpmax"], [0, 20, 30, 40, 100], right=False)
    save(g.groupby("depth_bin", observed=True)["clean_2022"].agg(pipes="size", became_clean="sum").reset_index(),
         "replaced_2018_2022.csv")


# ---------------------------------------------------------------- 5-6. двойник целевого участка

def twin() -> None:
    src = V[V["id"].isin(PIPE) & (V["date"] == A)].copy()
    src["xt"] = src["x_raw"].to_numpy() + pl._shift_fn(JS)(src["x_raw"]) - ACC[TID]
    src = src[(src["xt"] >= 0) & (src["xt"] < L_T)].reset_index(drop=True)
    x, dp, wt, sec = (src[k].to_numpy() for k in ("xt", "dp", "wt", "sec"))
    pred = pd.read_csv(ROOT / "outputs" / "case1_segments.csv")
    final = pred["thickness_pred"].to_numpy()
    prev = data.segment_state(src, L_T, WALL_T, x_col="xt")
    q05 = prev["rw_q05"].to_numpy()
    pipes = src.groupby("sec")["dp"].max()
    cut = sorted(src.loc[src["rec_repair"] == pl.CUT_OUT, "sec"].unique())

    def truth(seed: int, replaced=(), k: float = 0.0) -> np.ndarray:
        """Один розыгрыш «факта 2023»: все дефекты съехали на k метров, трубы из replaced заменены."""
        xs = x + k
        keep = (xs >= 0) & (xs < L_T)
        rm = np.isin(sec, list(replaced))
        return GM.simulate(xs[keep], dp[keep], wt[keep], L_T, WALL_T, n_sim=1, seed=seed, pos_sigma=0.0,
                           repair_mask=rm[keep], repair_prob=1.0, repair_groups=sec[keep])[0]

    reps = {"ничего": [], "3 трубы с предписанием": cut}
    for th in (40, 35, 33, 31):
        reps[f"все трубы с дефектом >= {th} %"] = pipes.index[pipes >= th].tolist()
    T, rows = {}, []
    for name, secs in reps.items():
        for k in (0, 3, 5, 10, 20):
            T[name, k] = [truth(3000 + s, secs, k) for s in range(8)]
            rows.append({"replaced": name, "pipes": len(secs), "pos_error_m": k,
                         "r2": np.mean([pearson_r2(y, final) for y in T[name, k]]),
                         "mae": np.mean([mae(y, final) for y in T[name, k]])})
    g = pd.DataFrame(rows)
    save(g.pivot(index="replaced", columns="pos_error_m", values="r2").reindex(list(reps)), "twin_r2.csv", index=True)
    save(g.pivot(index="replaced", columns="pos_error_m", values="mae").reindex(list(reps)), "twin_mae.csv", index=True)

    clean, deep = ~prev["has_def"].to_numpy(), prev["dp_max"].to_numpy() >= 40
    print(f"\nцелевой участок: сегментов без дефектов в 2022 году {clean.sum()}, с дефектом >= 40 % {deep.sum()}")
    for name, mask in (("все", np.ones(len(final), bool)), ("без чистых", ~clean), ("без чистых и глубоких", ~clean & ~deep)):
        print(f"  двойник, ничего не заменено, k=0, {name}: r² "
              f"{np.mean([pearson_r2(y[mask], final[mask]) for y in T['ничего', 0]]):.2f}")

    # --- варианты правок на тех же розыгрышах
    def sim(ps: float, replaced=(), prob: float = 0.0) -> np.ndarray:
        rm = np.isin(sec, list(replaced))
        return GM.simulate(x, dp, wt, L_T, WALL_T, n_sim=400, seed=42, pos_sigma=ps, pos_bump=np.ones(len(x)),
                           repair_mask=rm, repair_prob=prob, repair_groups=sec).mean(axis=0)

    seg = data.seg_of(x, L_T)
    n_seg = len(final)
    m_none, m_all = sim(1.0), sim(1.0, pipes.index[pipes >= 30].tolist(), 1.0)
    pseg = pd.Series(pd.Series(dp).groupby(sec).transform("max").to_numpy()).groupby(seg).max().reindex(range(n_seg)).fillna(0).to_numpy()

    def by_depth(p50, p40, p35, p30):
        p = np.select([pseg >= 50, pseg >= 40, pseg >= 35, pseg >= 30], [p50, p40, p35, p30], 0.0)
        return (1 - p) * m_none + p * m_all

    m3 = sim(3.0, cut, c["repair_prob"])
    variants = {
        "сданный прогноз": final,
        "только модель роста": pred["model_only"].to_numpy(),
        "σ положения 3 м, вес перцентиля 0,25": pl.blend(m3, q05, 0.25),
        "вероятность замены по глубине 0,8/0,5/0,15/0,07": by_depth(0.8, 0.5, 0.15, 0.07),
        "сданный, обрезан снизу по 10-му перцентилю": np.maximum(final, np.quantile(final, 0.10)),
    }
    cols = [("ничего", 0), ("3 трубы с предписанием", 0), ("3 трубы с предписанием", 3), ("3 трубы с предписанием", 5),
            ("все трубы с дефектом >= 40 %", 0), ("все трубы с дефектом >= 40 %", 3),
            ("все трубы с дефектом >= 35 %", 0), ("все трубы с дефектом >= 31 %", 0)]
    r = pd.DataFrame({f"{a} | {k} м": {n: np.mean([pearson_r2(y, v) for y in T[a, k]]) for n, v in variants.items()}
                      for a, k in cols})
    save(r, "fix_variants_r2.csv", index=True)


if __name__ == "__main__":
    leverage()
    law_transfer()
    ends_only()
    shifts = joints_2018_2022()
    replaced_between_2018_2022(shifts)
    twin()
