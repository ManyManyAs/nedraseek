"""Единая точка входа: исходные CSV -> result.csv + таблицы валидации.

Запуск из корня репозитория:  python scripts/run_pipeline.py [--config config.yaml]
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipetwin import baselines, io, segments, submission, validate  # noqa: E402


def main(config: str = "config.yaml") -> pd.DataFrame:
    cfg = io.load_config(config)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    seg_len = cfg["segment_len_m"]
    out_dir = Path(cfg["paths"]["out"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Загрузка и сетка сегментов
    t = io.load_all(cfg)
    grid = segments.build_grid(t["passport"], seg_len, io.load_sample_result(cfg))
    print(f"[1] сегментов: {len(grid)} по трубам: {grid.groupby('id').size().to_dict()}")

    # 2. Привязка данных к сегментам
    vtd_seg = segments.vtd_by_segment(t["vtd"], t["passport"], seg_len, cfg.get("vtd_exclude_types"))
    prof = segments.profile_features(grid, t["profile"], seg_len, cfg["case2"]["lowpoint_window_m"])
    seg2hseg = segments.map_regime_to_segments(grid, t["regime"])

    # 3. Прогноз по каждой трубе: кейс определяется наличием прошлых ВТД
    preds, bt = [], []
    for pid in grid["id"].unique():
        runs = sorted(pd.to_datetime(vtd_seg.loc[vtd_seg["id"] == pid, "date"].unique()))
        if runs:
            t_date = pd.Timestamp((cfg.get("t_dates") or {}).get(pid) or t["regime"].loc[t["regime"]["id"] == pid, "date"].max())
            if len(runs) >= 2:
                p = baselines.linear_growth(grid, vtd_seg, pid, runs[-2], runs[-1], t_date)
                method = "linear_growth"
            else:
                p = baselines.persistence(grid, vtd_seg, pid, runs[-1])
                method = "persistence"
            bt.append(validate.backtest_case1(grid, vtd_seg, pid))
            print(f"[3] труба {pid}: кейс 1, пропусков ВТД {len(runs)}, метод {method}")
        else:
            p = baselines.physics_case2(grid, prof, seg2hseg, t["regime"], pid, cfg)
            print(f"[3] труба {pid}: кейс 2, метод physics_case2")
        preds.append(p)

    # 4. Валидация и запись
    if bt and sum(len(b) for b in bt):
        bt_df = pd.concat(bt, ignore_index=True)
        bt_df.to_csv(out_dir / "backtest_case1.csv", index=False)
        print("[4] бэктест кейса 1:\n" + bt_df.to_string(index=False))
    res = pd.concat(preds, ignore_index=True)
    submission.write_result(res, grid, cfg["paths"]["result"])
    print(f"[4] записан {cfg['paths']['result']}: {len(res)} строк")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
