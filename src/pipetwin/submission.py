"""Запись и строгая проверка result.csv."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["id", "segment_id", "thickness_pred"]


def validate_result(res: pd.DataFrame, grid: pd.DataFrame) -> list[str]:
    """Возвращает список ошибок формата; пустой список = файл корректен."""
    errs = []
    if list(res.columns) != COLS:
        errs.append(f"колонки {list(res.columns)}, ожидались {COLS}")
        return errs
    num = pd.to_numeric(res["thickness_pred"], errors="coerce")
    if num.isna().any():
        errs.append(f"нечисловые или пустые значения thickness_pred: {int(num.isna().sum())}")
        return errs
    # Проверяем таблицу в той точности, в какой она будет записана.
    res = res.assign(thickness_pred=num.round(4))
    if res["thickness_pred"].isna().any():
        errs.append("есть пропуски в thickness_pred")
    if not np.isfinite(res["thickness_pred"].to_numpy(dtype=float)).all():
        errs.append("есть нечисловые/бесконечные значения")
    if (res["thickness_pred"] < 0).any():
        errs.append("есть отрицательная толщина")
    if res.duplicated(["id", "segment_id"]).any():
        errs.append("дубли (id, segment_id)")
    want = set(map(tuple, grid[["id", "segment_id"]].to_numpy().tolist()))
    got = set(map(tuple, res[["id", "segment_id"]].to_numpy().tolist()))
    if want - got:
        errs.append(f"не хватает сегментов: {len(want - got)}")
    if got - want:
        errs.append(f"лишние сегменты: {len(got - want)}")
    for pid, g in res.groupby("id"):
        if g["thickness_pred"].nunique() < 2:
            errs.append(f"труба {pid}: прогноз константа, r² не определён")
    return errs


def write_result(res: pd.DataFrame, grid: pd.DataFrame, path: str | Path) -> None:
    res = res[COLS].sort_values(["id", "segment_id"]).reset_index(drop=True)
    errs = validate_result(res, grid)
    if errs:
        raise ValueError("result.csv не прошёл проверку: " + "; ".join(errs))
    res.to_csv(path, index=False, float_format="%.4f", lineterminator="\n")  # десятичный разделитель — точка
