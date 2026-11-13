"""Тесты на реальных данных из data/raw/ и на синтетике для выравнивания."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pipetwin import align, case1, data, submission  # noqa: E402
from pipetwin.validate import pearson_r2, top_recall  # noqa: E402

RAW = ROOT / "data" / "raw"


def test_segments():
    assert data.n_segments(5248.4) == 525 and data.n_segments(44.7) == 5 and data.n_segments(100.0) == 11
    assert list(data.seg_of([-0.3, 0, 9.99, 10, 44.69, 501.0], 44.7)) == [0, 0, 0, 1, 4, 4]


def test_pearson_r2_and_recall():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(pearson_r2(y, 2 * y + 5) - 1.0) < 1e-12      # сдвиг и масштаб не влияют
    assert abs(pearson_r2(y, -y) - 1.0) < 1e-12             # знак тоже
    assert np.isnan(pearson_r2(y, np.ones(4)))              # константа -> NaN
    assert top_recall(np.arange(20.0), np.arange(20.0)) == 1.0


def test_vtd_cleaning():
    v = data.load_vtd(RAW)
    assert v["rw"].notna().all() and v["dp"].notna().all()
    med = v.groupby("date")["dp"].median()
    assert med.min() > 5 and med.max() < 30                 # все пропуски в процентах, включая 2018
    err = (v["rw"] - v["wt"] * (1 - v["dp"] / 100)).abs()
    assert err.quantile(0.999) < 0.05                       # толщина согласована с глубиной


def test_shift_recovery():
    rng = np.random.default_rng(0)
    xa = np.sort(rng.uniform(0, 500, 600))
    xb = np.concatenate([xa + 7.0 + rng.normal(0, 0.05, len(xa)), rng.uniform(0, 500, 60)])
    k, c = align.estimate_shift(xa, xb, 50, 450, 0.0, span=15)
    assert abs(k - 7.0) <= 0.25 and c > 0.7
    k0, _ = align.estimate_shift(xa, xa, 50, 450, 0.0, span=15)
    assert abs(k0) <= 0.25


def test_validate_result_catches_errors():
    grid = pd.DataFrame({"id": [1] * 3, "segment_id": [0, 1, 2]})
    ok = grid.assign(thickness_pred=[7.0, 6.5, 6.8])
    assert submission.validate_result(ok, grid) == []
    assert submission.validate_result(ok.iloc[:2], grid)                              # не хватает сегмента
    assert submission.validate_result(ok.assign(thickness_pred=7.0), grid)            # константа
    assert submission.validate_result(ok.assign(thickness_pred=[7.0, np.nan, 6.8]), grid)
    assert submission.validate_result(pd.concat([ok, ok.iloc[:1]]), grid)             # дубль
    assert submission.validate_result(ok.rename(columns={"thickness_pred": "pred"}), grid)


def test_pipeline_end_to_end(tmp_path):
    import run_pipeline
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["case1"]["n_sim"] = 100
    outs = []
    for k in (1, 2):
        cfg["paths"] = {"raw": str(RAW), "out": str(tmp_path / f"out{k}"), "result": str(tmp_path / f"r{k}.csv")}
        p = tmp_path / f"c{k}.yaml"
        p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
        run_pipeline.main(str(p))
        outs.append(pd.read_csv(tmp_path / f"r{k}.csv"))
    res = outs[0]
    assert list(res.columns) == ["id", "segment_id", "thickness_pred"]
    assert res.groupby("id").size().to_dict() == {1250009583: 1475, 1250022871: 525}
    assert not res.duplicated(["id", "segment_id"]).any()
    assert res["thickness_pred"].between(0.5, 9.5).all()
    assert res.equals(outs[1])                                                        # воспроизводимость
    val = pd.read_csv(tmp_path / "out1" / "validation_case1.csv")
    r2 = dict(zip(val["method"], val["r2"]))
    assert r2["Прошлый ВТД, дистанции выровнены"] > r2["Прошлый ВТД без выравнивания дистанций"] + 0.2


def test_angle_wraps_around_zero():
    """Углы 359° и 1° в одной точке: это один дефект."""
    a = pd.DataFrame({"x2": [1.0, 5.0], "ang": [359.0, 180.0], "dp": [20.0, 15.0]})
    b = pd.DataFrame({"x": [1.0, 5.0], "ang": [1.0, 180.0], "dp": [22.0, 15.0]})
    pairs, new, conf = case1.match_defects(a, b, d_mm=219.0, tol_m=0.08)
    assert len(pairs) == 2 and len(new) == 0 and conf.all()
    assert sorted(pairs["dp_a"]) == [15.0, 20.0]
    b2 = pd.DataFrame({"x": [1.0], "ang": [0.0]}).assign(dp=10.0)
    a2 = pd.DataFrame({"x2": [1.0], "ang": [360.0], "dp": [10.0]})
    assert len(case1.match_defects(a2, b2, 219.0, 0.08)[0]) == 1      # 0° и 360° — одно и то же


def test_wrong_segment_key_is_caught():
    """ответ сверяется с независимой сеткой, а не с самим собой."""
    grid = pd.DataFrame({"id": [1, 1, 1], "segment_id": [0, 1, 2]})
    bad = pd.DataFrame({"id": [1, 1, 1], "segment_id": [0, 1, 999999], "thickness_pred": [7.0, 6.5, 6.8]})
    assert submission.validate_result(bad, grid)
