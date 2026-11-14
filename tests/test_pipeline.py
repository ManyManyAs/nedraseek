"""Тесты на реальных данных из data/raw/ и на синтетике для выравнивания."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
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
    assert r2["Модель роста + 5-й перцентиль (итоговая)"] > r2["Прошлый ВТД, дистанции выровнены"]

def test_angle_wraps_around_zero():
    """Углы 359° и 1° в одной точке: это один дефект."""
    a = pd.DataFrame({"x2": [1.0, 5.0], "ang": [359.0, 180.0], "dp": [20.0, 15.0]})
    b = pd.DataFrame({"x": [1.0, 5.0], "ang": [1.0, 180.0], "dp": [22.0, 15.0]})
    pairs, new, conf = case1.match_defects(a, b, d_mm=219.0, tol_m=0.08)
    assert len(pairs) == 2 and len(new) == 0 and conf.all()
    assert sorted(pairs["dp_a"]) == [15.0, 20.0]
    b2 = pd.DataFrame({"x": [1.0], "ang": [0.0]}).assign(dp=10.0)
    a2 = pd.DataFrame({"x2": [1.0], "ang": [360.0], "dp": [10.0]})
    assert len(case1.match_defects(a2, b2, 219.0, 0.08)[0]) == 1      # 0° и 360°: одно и то же


def test_wrong_segment_key_is_caught():
    """Ответ сверяется с независимой сеткой из паспорта, а не с самим собой."""
    grid = pd.DataFrame({"id": [1, 1, 1], "segment_id": [0, 1, 2]})
    bad = pd.DataFrame({"id": [1, 1, 1], "segment_id": [0, 1, 999999], "thickness_pred": [7.0, 6.5, 6.8]})
    assert submission.validate_result(bad, grid)


def test_constant_and_written_precision():
    """Константный ряд даёт NaN, а не случайное число."""
    assert np.isnan(pearson_r2([0.1, 0.1, 0.1], [1.0, 2.0, 3.0]))
    grid = pd.DataFrame({"id": [1] * 3, "segment_id": [0, 1, 2]})
    tiny = grid.assign(thickness_pred=[7.00001, 7.00002, 7.00003])     # после округления до 4 знаков: константа
    assert submission.validate_result(tiny, grid)
    assert submission.validate_result(grid.assign(thickness_pred=["7", "x", "6"]), grid)


def test_result_does_not_depend_on_row_order():
    """Перестановка строк ВТД не меняет прогноз."""
    from pipetwin import pipeline
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["case1"]["n_sim"] = 50
    P, V = data.load_passport(RAW), data.load_vtd(RAW)
    p1 = pipeline.run_case1(cfg, P, V)["pred"]["thickness_pred"].to_numpy()
    raw_shuffled = V.sample(frac=1.0, random_state=7)
    V2 = raw_shuffled.sort_values(["id", "date", "x", "ang", "dp", "rw", "fl", "fw"], kind="mergesort").reset_index(drop=True)
    p2 = pipeline.run_case1(cfg, P, V2)["pred"]["thickness_pred"].to_numpy()
    assert np.allclose(p1, p2)


def _case1(n_sim=60, **over):
    from pipetwin import pipeline
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["case1"].update(n_sim=n_sim, **over)
    P, V, J = data.load_passport(RAW), data.load_vtd(RAW), data.load_joints(RAW)
    return pipeline, cfg, P, V, J


def test_heldout_block_does_not_leak():
    """Изменение скрытого блока пропуска B не меняет прогноз для этого блока."""
    pipeline, cfg, P, V, J = _case1()
    base = pipeline.run_case1(cfg, P, V, J)["val_points"]
    vid, B = int(base["id"].iloc[0]), pd.Timestamp(cfg["case1"]["target_run"])
    lo, hi = 200.0, 300.0                                   # блок 2 из пяти: сегменты 20–29
    V2 = V.copy()
    m = (V2["id"] == vid) & (V2["date"] == B) & (V2["x"] >= lo) & (V2["x"] < hi)
    assert m.sum() > 100
    V2.loc[m, "dp"] = 11.0                                  # портим глубины и толщины скрытого блока
    V2.loc[m, "rw"] = V2.loc[m, "wt"] * 0.89
    J2 = J.copy()
    acc = 5293.1
    jm = (J2["date"] == B) & (J2["js"] - acc >= lo) & (J2["js"] - acc < hi)
    assert jm.sum() >= 5
    J2.loc[jm, "js"] += 3.0                                 # и положение труб внутри блока
    pert = pipeline.run_case1(cfg, P, V2, J2)["val_points"]
    blk = base["block"] == 2
    for col in ("pred_pers", "pred_q05", "pred_mc", "pred_blend", "pred_p10", "pred_p90"):
        assert np.allclose(base.loc[blk, col], pert.loc[blk, col]), col
    assert not np.allclose(base.loc[blk, "fact"], pert.loc[blk, "fact"])   # факт при этом изменился


def test_repair_scenario_raises_flagged_segments():
    """Сценарий замены труб с предписанием «врезка катушки» действует только на их сегменты."""
    pipeline, cfg, P, V, J = _case1(repair_prob=0.0)
    p0 = pipeline.run_case1(cfg, P, V, J)["pred"]
    cfg["case1"]["repair_prob"] = 1.0
    p1 = pipeline.run_case1(cfg, P, V, J)["pred"]
    flag = p1["repair_flag"].to_numpy()
    assert 3 <= flag.sum() <= 12
    d = p1["model_only"].to_numpy() - p0["model_only"].to_numpy()
    assert d[flag].max() > 2.0                               # заменённая труба: почти номинальная стенка
    far = np.ones(len(p1), dtype=bool)
    for i in np.where(flag)[0]:
        far[max(0, i - 2): i + 3] = False
    assert np.abs(d[far]).mean() < 0.08 and abs(d[far].mean()) < 0.02   # остальные: только шум розыгрыша, без смещения


def test_case2_reference_and_shape():
    """Кейс 2: прогноз не константа, в пределах стенки, а проверка признака воспроизводится."""
    from pipetwin import pipeline
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    P, V, prof = data.load_passport(RAW), data.load_vtd(RAW), data.load_profile(RAW)
    r = pipeline.run_case2(cfg, P, prof, V)
    pred, ref = r["pred"], r["reference"]
    assert len(pred) == 1475 and pred["thickness_pred"].nunique() > 100
    assert (pred["p10"] <= pred["thickness_pred"]).all() and (pred["thickness_pred"] <= pred["p90"]).all()
    assert pred["thickness_pred"].between(0.3 * 4.5, 4.5).all()
    assert int(cfg["case2"]["target_id"]) not in set(ref["id"])          # целевой участок в проверке не участвует
    assert (pd.to_datetime(ref["run"]) < pd.Timestamp(cfg["case2"]["target_date"])).all()   # только пропуски до T
    assert ref["r2"].notna().all() and ref["r2"].between(0, 0.3).all()   # сигнал слабый, это ожидаемо


def test_missing_column_is_reported(tmp_path):
    raw = Path(__file__).resolve().parents[1] / "data" / "raw"
    a = pd.read_csv(raw / data.FILES["profile"]).drop(columns=["z"])
    a.to_csv(tmp_path / data.FILES["profile"], index=False)
    with pytest.raises(ValueError, match="z"):
        data.load_profile(tmp_path)
