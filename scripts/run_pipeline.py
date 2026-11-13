"""Единая точка входа: исходные CSV из data/raw/ -> result.csv, валидация, вспомогательные таблицы.

Запуск из корня репозитория:  python scripts/run_pipeline.py [--config config.yaml]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipetwin import data, pipeline, submission  # noqa: E402


def main(config: str = "config.yaml") -> pd.DataFrame:
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    np.random.seed(cfg["seed"])
    raw, out = cfg["paths"]["raw"], Path(cfg["paths"]["out"])
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)

    # 1. Загрузка и очистка
    P, V, prof = data.load_passport(raw), data.load_vtd(raw), data.load_profile(raw)
    print(f"[1] участков: {len(P)}, дефектов потери металла: {len(V)}, пропусков ВТД: {V['date'].nunique()}")

    # 2. Кейс 1
    r1 = pipeline.run_case1(cfg, P, V)
    lo, hi = r1["drift_ends"]
    print(f"[2] кейс 1, участок {cfg['case1']['target_id']}: сдвиг дистанций {lo:.1f} м в начале, {hi:.1f} м в конце")
    print("    проверка на соседних участках с известным целевым пропуском:")
    print(r1["validation"].round(3).to_string(index=False))
    print(f"    покрытие интервала P10–P90: {r1['coverage']:.0%}")
    print("    точность «прошлого ВТД» по горизонтам:")
    print(r1["horizons"].round(3).to_string(index=False))

    # 3. Кейс 2
    r2 = pipeline.run_case2(cfg, P, prof)
    print(f"[3] кейс 2, участок {cfg['case2']['target_id']}: {len(r2)} сегментов, физическая оценка по рельефу")

    # 4. Запись
    both = pd.concat([r1["pred"], r2], ignore_index=True)
    # Ожидаемая сетка строится независимо от ответа: из паспортной длины целевых участков.
    pp = P.set_index("id")
    grid = pd.concat([pd.DataFrame({"id": int(i), "segment_id": range(int(pp.loc[int(i), "n_seg"]))})
                      for i in (cfg["case1"]["target_id"], cfg["case2"]["target_id"])], ignore_index=True)
    submission.write_result(both[["id", "segment_id", "thickness_pred"]], grid, cfg["paths"]["result"])
    both[["id", "segment_id", "thickness_pred", "p10", "p90"]].round(4).to_csv(out / "result_with_intervals.csv", index=False)
    r1["validation"].round(4).to_csv(out / "validation_case1.csv", index=False)
    r1["horizons"].round(4).to_csv(out / "horizons_case1.csv", index=False)
    r1["alignment"].round(3).to_csv(out / "alignment_case1.csv", index=False)
    r1["pred"].round(4).to_csv(out / "case1_segments.csv", index=False)
    r2.round(4).to_csv(out / "case2_segments.csv", index=False)
    pd.concat([pipeline.top_segments(r1["pred"]), pipeline.top_segments(r2)]).round(3).to_csv(out / "top10_segments.csv", index=False)
    gm = r1["model"]
    summary = {
        "case1": {"target_id": int(cfg["case1"]["target_id"]), "n_segments": int(len(r1["pred"])),
                  "validation": r1["validation"].round(4).to_dict("records"), "p10_p90_coverage": round(r1["coverage"], 3),
                  "matched_pairs": int(gm.n_pairs), "new_defects_per_old": round(gm.new_rate, 3),
                  "shift_start_m": round(lo, 2), "shift_end_m": round(hi, 2)},
        "case2": {"target_id": int(cfg["case2"]["target_id"]), "n_segments": int(len(r2)), "method": "relief score"},
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[4] записан {cfg['paths']['result']}: {len(both)} строк; таблицы — в {out}/")
    return both


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
