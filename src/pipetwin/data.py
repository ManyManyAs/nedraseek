"""Загрузка и очистка реальных файлов организаторов (data/raw/).

Ключ везде: ID простого участка. Все правила очистки собраны здесь, сводка в README.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FILES = {
    "passport": "pot_south.csv",
    "profile": "altitudes_south.csv",
    "vtd": "vtd_south_train.csv",
}
SEG_LEN = 10.0

# Колонки, без которых расчёт не работает. Проверяются при чтении, чтобы при новой выгрузке
# сразу было видно, чего не хватает, а не ловить KeyError в середине пайплайна.
REQUIRED = {
    "profile": ["id", "total_distance", "z"],
    "vtd": ["id", "date", "Distance", "Distance_raw", "Wall Thickness", "Residual Wall Thickness",
            "Defect Depth in %", "Angle", "Feature Length", "Feature Width", "Defect Type",
            "Section Number", "Relative Distance", "Recommended Repair Method", "Repair Time (Years)",
            "is_beyond_pot", "Class", "Feature Description"],
}


def check_columns(df: pd.DataFrame, key: str) -> pd.DataFrame:
    missing = [c for c in REQUIRED.get(key, []) if c not in df.columns]
    if missing:
        raise ValueError(f"в файле {FILES[key]} нет колонок: {', '.join(missing)}")
    if df.empty:
        raise ValueError(f"файл {FILES[key]} пустой")
    return df


# ---------- сегменты ----------

def n_segments(length_m: float) -> int:
    """Число 10-метровых сегментов участка: номера 0..floor(L/10)."""
    return int(np.floor(length_m / SEG_LEN)) + 1


def seg_of(x, length_m: float) -> np.ndarray:
    """Номер сегмента для дистанции внутри участка.

    Дистанции вне [0, L] (сбой одометра, записи за концом участка) прижимаются к краям.
    """
    x = np.clip(np.asarray(x, dtype=float), 0.0, length_m)
    return np.clip(np.floor(x / SEG_LEN).astype(int), 0, n_segments(length_m) - 1)


# ---------- паспорт ----------

def _col(df: pd.DataFrame, part: str) -> str:
    hits = [c for c in df.columns if part in c]
    if not hits:
        raise KeyError(f"в паспорте нет колонки с '{part}'")
    return hits[0]


def load_passport(raw: str | Path) -> pd.DataFrame:
    p = pd.read_csv(Path(raw) / FILES["passport"])
    out = pd.DataFrame({
        "id": p[_col(p, "ID простого участка")].astype("int64"),
        "pipeline": p[_col(p, "Наименование основного участка")],
        "name": p[_col(p, "Наименование простого участка")],
        "length_m": p[_col(p, "L простого участка")] * 1000.0,
        "d_mm": p["Общие данные_D, мм"],
        "wall_mm": p[_col(p, "Толщина  стенки по исполнительной")],
        "start_date": pd.to_datetime(p["Общие данные_Дата ввода"], format="%d.%m.%Y"),
    })
    out["n_seg"] = out["length_m"].map(n_segments)
    return out.sort_values(["pipeline", "id"]).reset_index(drop=True)


def load_profile(raw: str | Path) -> pd.DataFrame:
    a = check_columns(pd.read_csv(Path(raw) / FILES["profile"]), "profile")
    a = a.rename(columns={"total_distance": "dist_m", "z": "z_m"})
    # Повторы дистанции усредняются, а не берётся первая строка: так порядок выгрузки не важен.
    return a.groupby(["id", "dist_m"], as_index=False)["z_m"].mean().sort_values(["id", "dist_m"]).reset_index(drop=True)


# ---------- ВТД ----------

_INTERNAL = {"Внутренний": True, "внутр.": True, "Внешний": False, "внешн.": False}


def load_vtd(raw: str | Path) -> pd.DataFrame:
    """Записи потери металла по всем пропускам в единой шкале.

    Правила очистки:
      1. оставляем записи, у которых есть и остаточная толщина, и глубина: это дефекты потери
         металла; швы, арматура, вмятины без потери металла отбрасываются;
      2. пропуск 2018 года хранит глубину долей (0.18), остальные: процентами (18): приводим к %;
      3. написания типа дефекта («внутр.» / «Внутренний») сводятся к флагу internal;
      4. пустой угол заменяется на 180° (низ трубы); в текущей выгрузке таких записей нет.
    """
    v = check_columns(pd.read_csv(Path(raw) / FILES["vtd"], low_memory=False), "vtd")
    v = v.rename(columns={
        "Distance": "x", "Distance_raw": "x_raw", "Wall Thickness": "wt",
        "Residual Wall Thickness": "rw", "Defect Depth in %": "dp", "Angle": "ang",
        "Feature Length": "fl", "Feature Width": "fw", "Class": "cls",
        "Feature Description": "descr", "Defect Type": "dtype", "is_beyond_pot": "beyond",
        "Section Number": "sec", "Relative Distance": "rel",
        "Recommended Repair Method": "rec_repair", "Repair Time (Years)": "repair_time",
    })
    v["date"] = pd.to_datetime(v["date"])
    v = v[v["rw"].notna() & v["dp"].notna()].copy()
    for d, g in v.groupby("date"):
        if g["dp"].max() <= 1.5:  # глубина записана долей
            v.loc[g.index, "dp"] = g["dp"] * 100.0
    v["internal"] = v["dtype"].map(_INTERNAL).astype("boolean").fillna(True).astype(bool)
    v["ang"] = v["ang"].fillna(180.0)
    keep = ["id", "date", "x", "x_raw", "wt", "rw", "dp", "ang", "fl", "fw", "cls", "descr", "internal", "beyond",
            "sec", "rel", "rec_repair", "repair_time"]
    # Стабильный порядок: результат не должен зависеть от порядка строк в выгрузке.
    return v[keep].sort_values(["id", "date", "x", "ang", "dp", "rw", "fl", "fw"], kind="mergesort").reset_index(drop=True)


def load_joints(raw: str | Path) -> pd.DataFrame:
    """Начала труб (секций между кольцевыми швами) в каждом пропуске: одна строка на (пропуск, номер трубы).

    Снаряд для каждой записи даёт номер трубы и расстояние от её начала. Начало трубы по одометру -
    дистанция записи минус это расстояние; берётся медиана по всем записям трубы, включая швы и арматуру.
    Номера труб пропусков 2022 и 2023 годов совпадают, поэтому по ним дистанции двух пропусков
    совмещаются точнее, чем по плотности дефектов.
    """
    v = pd.read_csv(Path(raw) / FILES["vtd"], low_memory=False,
                    usecols=["id", "date", "Distance_raw", "Section Number", "Relative Distance"])
    v = v.dropna(subset=["Section Number", "Relative Distance", "Distance_raw"])
    v["date"] = pd.to_datetime(v["date"])
    v["js"] = v["Distance_raw"] - v["Relative Distance"]
    j = (v.groupby(["date", "Section Number"])
          .agg(js=("js", "median"), n=("js", "size"), id=("id", "min")).reset_index()
          .rename(columns={"Section Number": "sec"}))
    return j.sort_values(["date", "js", "sec"]).reset_index(drop=True)


def run_dates(vtd: pd.DataFrame, pipe_id: int) -> list[pd.Timestamp]:
    return sorted(vtd.loc[vtd["id"] == pipe_id, "date"].unique())


def segment_state(defects: pd.DataFrame, length_m: float, wall_mm: float, x_col: str = "x") -> pd.DataFrame:
    """Состояние сегментов по набору дефектов одного пропуска одного участка.

    thk_min: минимальная остаточная толщина; в сегменте без дефектов: толщина стенки.
    """
    n = n_segments(length_m)
    d = defects.assign(seg=seg_of(defects[x_col], length_m))
    g = d.groupby("seg")
    wt = g["wt"].median().reindex(range(n))
    out = pd.DataFrame({
        "segment_id": np.arange(n),
        "thk_min": g["rw"].min().reindex(range(n)).to_numpy(),
        "n_def": g.size().reindex(range(n)).fillna(0).astype(int).to_numpy(),
        "dp_max": g["dp"].max().reindex(range(n)).fillna(0.0).to_numpy(),
        "rw_q05": g["rw"].quantile(0.05).reindex(range(n)).to_numpy(),
        "wt": wt.interpolate(limit_direction="both").fillna(wall_mm).to_numpy(),
    })
    out["has_def"] = out["n_def"] > 0
    out["thk_min"] = out["thk_min"].fillna(out["wt"])
    out["rw_q05"] = out["rw_q05"].fillna(out["wt"])
    return out
