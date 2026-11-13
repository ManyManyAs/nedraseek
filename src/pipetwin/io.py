"""Загрузка конфигурации и исходных CSV с приведением к каноническим колонкам."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

REQUIRED = {
    "passport": ["id", "wall_mm", "start_date", "length_m"],
    "profile": ["id", "dist_m", "z_m"],
    "vtd": ["id", "date", "dist_m", "depth_pct"],
    "uzt": ["id", "date", "dist_m", "thickness_mm"],
    "regime": ["id", "date", "hseg"],
}
DATE_COLS = {"passport": ["start_date"], "vtd": ["date"], "uzt": ["date"], "regime": ["date"]}
OPTIONAL_TABLES = {"vtd", "uzt"}  # во втором кейсе их нет


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_csv(path: Path) -> pd.DataFrame:
    """Читает CSV, сам определяя разделитель (',' или ';')."""
    return pd.read_csv(path, sep=None, engine="python")


def load_table(name: str, cfg: dict) -> pd.DataFrame:
    path = Path(cfg["paths"]["raw"]) / f"{name}.csv"
    if not path.exists():
        if name in OPTIONAL_TABLES:
            return pd.DataFrame(columns=REQUIRED[name])
        raise FileNotFoundError(f"Нет файла {path}")
    df = _read_csv(path).rename(columns=(cfg.get("columns", {}).get(name) or {}))
    missing = [c for c in REQUIRED[name] if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name}.csv: нет колонок {missing}. Есть: {list(df.columns)}. "
            f"Заполните columns.{name} в config.yaml."
        )
    for c in DATE_COLS.get(name, []):
        df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=False)
    return df


def load_all(cfg: dict) -> dict[str, pd.DataFrame]:
    return {name: load_table(name, cfg) for name in REQUIRED}


def load_sample_result(cfg: dict) -> pd.DataFrame | None:
    """sample_result.csv — источник истины по списку сегментов, если он есть."""
    path = Path(cfg["paths"]["raw"]) / "sample_result.csv"
    return _read_csv(path) if path.exists() else None
