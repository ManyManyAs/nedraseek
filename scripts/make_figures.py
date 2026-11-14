"""Графики для слайдов и README по таблицам из outputs/.

Запуск после run_pipeline.py:  python scripts/make_figures.py
Картинки сохраняются в outputs/figures/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pipetwin import data, features  # noqa: E402

OUT = ROOT / "outputs"
FIG = OUT / "figures"

# Палитра: проверенные первые слоты категориальной шкалы и нейтральные тона текста.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelcolor": INK2, "axes.edgecolor": GRID, "axes.linewidth": 1,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK, "legend.frameon": False,
})


def _save(fig, name: str) -> None:
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  ", name)


def pred_vs_fact(vp: pd.DataFrame, val: pd.DataFrame) -> None:
    """Прогноз против факта на независимой проверке: итоговый метод и прошлый ВТД."""
    r2 = dict(zip(val["method"], val["r2"]))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharex=True, sharey=True)
    lo, hi = 3.6, 8.2
    for ax, col, color, title, key in (
        (axes[0], "pred_pers", ORANGE, "Прошлый ВТД, дистанции выровнены", "Прошлый ВТД, дистанции выровнены"),
        (axes[1], "pred_blend", BLUE, "Итоговая модель", "Модель роста + 5-й перцентиль (итоговая)"),
    ):
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.scatter(vp["fact"], vp[col], s=46, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.set_title(title)
        ax.text(0.04, 0.95, f"r² = {r2[key]:.2f}".replace(".", ","), transform=ax.transAxes, va="top",
                fontsize=13, fontweight="bold", color=INK)
        ax.set_xlim(lo, hi), ax.set_ylim(lo, hi)
        ax.set_xlabel("Факт 2023 года, мм")
    axes[0].set_ylabel("Прогноз по пропуску 2022 года, мм")
    fig.suptitle("Минимальная толщина сегмента: прогноз и факт, 50 сегментов участка 1250011604",
                 x=0.01, ha="left", fontsize=12, color=INK2, y=1.0)
    fig.tight_layout()
    _save(fig, "pred_vs_fact.png")


def approaches(val: pd.DataFrame) -> None:
    """Сравнение подходов: r² и ошибка в миллиметрах на одной и той же проверке."""
    names = {
        "Прошлый ВТД без выравнивания дистанций": "Прошлый ВТД",
        "Прошлый ВТД, дистанции выровнены": "Прошлый ВТД + выравнивание",
        "5-й перцентиль толщины по дефектам сегмента": "5-й перцентиль толщины",
        "Модель роста дефектов": "Модель роста дефектов",
        "Модель роста + 5-й перцентиль (итоговая)": "Итог: модель + перцентиль",
    }
    v = val.assign(label=val["method"].map(names)).iloc[::-1]
    final = v["method"].str.contains("итоговая")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, col, title, fmt in ((axes[0], "r2", "r², больше лучше", "{:.2f}"),
                                (axes[1], "mae_mm", "Средняя ошибка, мм, меньше лучше", "{:.2f}")):
        colors = [BLUE if f else "#a9c8ee" for f in final]
        ax.barh(v["label"], v[col], color=colors, height=0.62)
        for y, (x, f) in enumerate(zip(v[col], final)):
            ax.text(x, y, "  " + fmt.format(x).replace(".", ","),
                    va="center", fontsize=11, color=INK if f else INK2, fontweight="bold" if f else "normal")
        ax.set_title(title, fontsize=12)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, v[col].max() * 1.22)
        ax.tick_params(axis="x", labelsize=9)
    axes[0].tick_params(axis="y", labelsize=11)
    fig.suptitle("Кейс 1: независимая проверка, прогноз 2022 → 2023 года, 50 сегментов",
                 x=0.01, ha="left", fontsize=12, color=INK2, y=1.02)
    _save(fig, "approaches.png")


def horizons(hz: pd.DataFrame, val: pd.DataFrame) -> None:
    """Точность по горизонту прогноза. Одна точка независимая, остальные диагностические."""
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    d = hz.copy()
    ax.scatter(d["horizon_years"], d["r2_prev_run_aligned"], s=70, facecolor=SURFACE, edgecolor=ORANGE,
               linewidth=2, zorder=3, label="Прошлый ВТД + выравнивание (диагностика)")
    for r in d.itertuples():
        ax.annotate(f"{r.n_segments} сегм.", (r.horizon_years, r.r2_prev_run_aligned), xytext=(8, -3),
                    textcoords="offset points", fontsize=9, color=INK2)
    fin = val.loc[val["method"].str.contains("итоговая"), "r2"].iloc[0]
    h = val["horizon_years"].iloc[0]
    ax.scatter([h], [fin], s=80, color=BLUE, marker="D", zorder=4, label="Итоговая модель (независимая проверка)")
    ax.annotate(f"{fin:.2f}".replace(".", ","), (h, fin), xytext=(10, 2), textcoords="offset points",
                fontsize=11, fontweight="bold", color=INK)
    ax.set_xlim(0, 6), ax.set_ylim(-0.04, 1)
    ax.set_xlabel("Горизонт прогноза, лет")
    ax.set_ylabel("r²")
    ax.set_title("Чем дальше прошлый пропуск, тем меньше он говорит о будущем минимуме")
    ax.legend(loc="upper right", fontsize=9)
    _save(fig, "r2_by_horizon.png")


def alignment(js: pd.DataFrame, P: pd.DataFrame, target: int) -> None:
    """Сдвиг дистанций 2022 → 2023 по началам общих труб и интерполяция через целевой участок."""
    # Пропуска 2023 года на целевом участке нет, поэтому он виден как самый длинный разрыв между началами труб.
    x = np.sort(js["js_a"].to_numpy())
    k = int(np.argmax(np.diff(x)))
    lo_t, hi_t = x[k], x[k + 1]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.axvspan(lo_t, hi_t, color="#eef3fb", zorder=0)
    ax.text((lo_t + hi_t) / 2, 12.3, f"Целевой участок {target}: пропуска 2023 года нет",
            ha="center", fontsize=10, color=INK2)
    ax.plot([lo_t, hi_t], [js.loc[js["js_a"] == lo_t, "shift"].iloc[0], js.loc[js["js_a"] == hi_t, "shift"].iloc[0]],
            color=BLUE, lw=2, ls=(0, (5, 3)), label="Интерполяция внутри участка")
    ax.scatter(js["js_a"], js["shift"], s=26, color=ORANGE, linewidth=0, zorder=3, label="Начала общих труб")
    ax.set_xlabel("Дистанция по трубопроводу в пропуске 2022 года, м")
    ax.set_ylabel("Сдвиг 2023 − 2022, м")
    ax.set_ylim(-0.5, 14)
    ax.set_title("Одометры снарядов расходятся на 2-13 м: без поправки дефект уходит в соседний сегмент")
    ax.legend(loc="center left", fontsize=9, bbox_to_anchor=(0.0, 0.62))
    _save(fig, "alignment.png")


def profile(pred: pd.DataFrame, prof: pd.DataFrame, pid: int, length: float, title: str, name: str,
            top: pd.DataFrame | None = None, start: pd.Series | None = None, scenario: bool = False) -> None:
    """Профиль толщины вдоль участка с интервалом и рельефом под ним."""
    x = (pred["segment_id"].to_numpy() + 0.5) * data.SEG_LEN
    fig, (ax, az) = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.1], "hspace": 0.08})
    ax.fill_between(x, pred["p10"], pred["p90"], color="#d6e4f7", lw=0, step="mid",
                    label="Сценарии: изношенная труба / без дефектов" if scenario else "Интервал P10-P90")
    if start is not None:
        ax.step(x, start, where="mid", color=MUTED, lw=0.7, alpha=0.55, label="Минимум в пропуске 2022 года")
    ax.step(x, pred["thickness_pred"], where="mid", color=BLUE, lw=1.6, label="Прогноз (среднее)")
    if top is not None and len(top):
        tx = (top["segment_id"].to_numpy() + 0.5) * data.SEG_LEN
        ax.scatter(tx, top["thickness_pred"], s=22, color=ORANGE, zorder=4, label="10 % самых опасных сегментов")
    ax.set_ylabel("Толщина стенки, мм")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    pf = features.profile_features(prof, pid, length)
    az.fill_between(pf["p_dist"], pf["p_z"].min() - 1, pf["p_z"], color="#e8e7e2", lw=0)
    az.plot(pf["p_dist"], pf["p_z"], color=INK2, lw=1)
    az.set_ylabel("Отметка, м")
    az.set_xlabel("Дистанция от начала участка, м")
    az.set_ylim(pf["p_z"].min() - 1, pf["p_z"].max() + 1)
    az.set_xlim(0, length)
    _save(fig, name)


def case2_signal(ref: pd.DataFrame) -> None:
    """Сила признака кейса 2 на других трубопроводах."""
    g = (ref.groupby("pipeline").apply(lambda d: pd.Series({"r2": np.average(d["r2"], weights=d["n_segments"]),
                                                           "n": d["n_segments"].sum()}), include_groups=False)
         .reset_index().sort_values("r2"))
    label = {"Куст-1 - ДНС-1": "Куст-1 - ДНС-1 (ВТД 2018)\nнефть, вода, газ", "УЗ-1 КПСОД - ДНС-1": "УЗ-1 - ДНС-1 (ВТД 2014)\nнефть, вода, газ"}
    fig, ax = plt.subplots(figsize=(8.6, 2.6))
    colors = [BLUE] * len(g)
    ax.barh([label.get(p, p) for p in g["pipeline"]], g["r2"], color=colors, height=0.55)
    for y, (v, n) in enumerate(zip(g["r2"], g["n"])):
        ax.text(v, y, f"  {v:.2f}".replace(".", ",") + f"   ({int(n)} сегм.)", va="center", fontsize=10, color=INK2)
    ax.set_xlim(0, 0.14)
    ax.set_xticks([0, 0.05, 0.10])
    ax.set_xticklabels(["0", "0,05", "0,10"])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("r² высотной отметки и толщины на первом пропуске ВТД до T = 01.07.2022")
    ax.set_title("Кейс 2: рельеф объясняет не больше 9 % разброса толщины")
    _save(fig, "case2_signal.png")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    s = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    t1, t2 = s["case1"]["target_id"], s["case2"]["target_id"]
    raw = ROOT / "data" / "raw"
    P, prof = data.load_passport(raw), data.load_profile(raw)
    pp = P.set_index("id")
    val = pd.read_csv(OUT / "validation_case1.csv")
    vp = pd.read_csv(OUT / "validation_case1_points.csv")
    c1, c2 = pd.read_csv(OUT / "case1_segments.csv"), pd.read_csv(OUT / "case2_segments.csv")
    top = pd.read_csv(OUT / "top10_segments.csv")
    print("графики в", FIG)
    pred_vs_fact(vp, val)
    approaches(val)
    horizons(pd.read_csv(OUT / "horizons_case1.csv"), val)
    alignment(pd.read_csv(OUT / "joint_shifts_case1.csv"), P, t1)
    profile(c1, prof, t1, pp.loc[t1, "length_m"], f"Кейс 1, участок {t1}: прогноз минимальной толщины на пропуск 2023 года",
            "profile_case1.png", top=top[top["id"] == t1].merge(c1[["segment_id"]], on="segment_id"),
            start=c1["thk_start_run"])
    profile(c2, prof, t2, pp.loc[t2, "length_m"], f"Кейс 2, участок {t2}: прогноз от ввода в эксплуатацию",
            "profile_case2.png", top=top[top["id"] == t2], scenario=True)
    case2_signal(pd.read_csv(OUT / "validation_case2.csv"))


if __name__ == "__main__":
    main()
