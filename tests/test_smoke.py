"""Сквозной прогон на синтетике: пайплайн отрабатывает и выдаёт корректный result.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import make_synthetic  # noqa: E402
import run_pipeline  # noqa: E402
from pipetwin import segments, validate  # noqa: E402


def test_segment_id():
    assert list(segments.segment_id([0, 9.99, 10, 25.3])) == [0, 0, 1, 2]


def test_pearson_r2():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(validate.pearson_r2(y, 2 * y + 5) - 1.0) < 1e-12  # сдвиг и масштаб не влияют
    assert np.isnan(validate.pearson_r2(y, np.ones(4)))          # константа -> NaN


def test_pipeline_end_to_end(tmp_path):
    raw = make_synthetic.make(tmp_path / "raw")
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["paths"] = {"raw": str(raw), "out": str(tmp_path / "out"), "result": str(tmp_path / "result.csv")}
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    run_pipeline.main(str(cfg_path))
    res = pd.read_csv(tmp_path / "result.csv")
    assert list(res.columns) == ["id", "segment_id", "thickness_pred"]
    assert len(res) == 301 + 199 and not res.duplicated(["id", "segment_id"]).any()
    assert res["thickness_pred"].between(0, 8.0).all()
    assert res.groupby("id")["thickness_pred"].nunique().min() > 1

    res2_path = tmp_path / "result2.csv"
    cfg["paths"]["result"] = str(res2_path)
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    run_pipeline.main(str(cfg_path))
    assert res.equals(pd.read_csv(res2_path))  # воспроизводимость
