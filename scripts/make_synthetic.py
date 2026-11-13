"""Синтетические данные в формате кейса — ТОЛЬКО для проверки, что пайплайн запускается.

Это не реальные данные и не аргумент о качестве модели.
Запуск:  python scripts/make_synthetic.py [папка, по умолчанию data/synthetic]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def make(out: str | Path = "data/synthetic", seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pipes = [  # труба 1 — кейс 1 (три прошлых ВТД), труба 2 — кейс 2 (без ВТД)
        dict(id=1, diameter_mm=219, wall_mm=8.0, steel="09Г2С", coating=0, start_date="2010-01-01", length_m=3005.0),
        dict(id=2, diameter_mm=159, wall_mm=6.0, steel="20", coating=0, start_date="2014-06-01", length_m=1990.0),
    ]
    pd.DataFrame(pipes).to_csv(out / "passport.csv", index=False)

    prof, reg, vtd, uzt = [], [], [], []
    for p in pipes:
        x = np.arange(0, p["length_m"] + 1, 25.0)
        z = 100 + 8 * np.sin(x / 300) + rng.normal(0, 0.3, len(x)).cumsum() * 0.2
        prof.append(pd.DataFrame({"id": p["id"], "dist_m": x, "z_m": z}))
        days = pd.date_range(p["start_date"], "2024-01-01", freq="D")
        for h in range(1, 5):
            n = len(days)
            reg.append(pd.DataFrame({
                "id": p["id"], "date": days, "hseg": h,
                "q_liq": rng.normal(500, 30, n), "watercut": np.clip(np.linspace(30, 90, n) + rng.normal(0, 2, n), 0, 100),
                "p": 15 - h + rng.normal(0, 0.2, n), "t": 45 - 3 * h + rng.normal(0, 1, n),
                "v_liq": rng.normal(0.8, 0.1, n), "ph": rng.normal(6.2, 0.1, n),
                "co2": rng.normal(1.5, 0.1, n), "inhibitor": (rng.random(n) > 0.4) * 20.0,
            }))
        if p["id"] == 1:
            n_seg = int(np.ceil(p["length_m"] / 10))
            zi = np.interp((np.arange(n_seg) + 0.5) * 10, x, z)
            risk = (zi.max() - zi) / (zi.max() - zi.min())
            for run in ["2015-05-01", "2019-05-01", "2022-05-01"]:
                yrs = (pd.Timestamp(run) - pd.Timestamp(p["start_date"])).days / 365.25
                depth = np.clip(risk * 2.5 * yrs + rng.normal(0, 4, n_seg), 0, 80)
                seg = np.where(depth > 10)[0]  # порог чувствительности снаряда
                vtd.append(pd.DataFrame({
                    "id": 1, "date": run, "dist_m": seg * 10 + rng.uniform(0, 10, len(seg)),
                    "clock": rng.choice([5, 6, 7], len(seg)), "length_mm": rng.uniform(10, 80, len(seg)),
                    "width_mm": rng.uniform(10, 60, len(seg)), "depth_pct": depth[seg], "type": "коррозия"}))
            pts = rng.choice(n_seg, 15, replace=False)
            uzt.append(pd.DataFrame({"id": 1, "date": "2021-03-01", "point": np.arange(15), "dist_m": pts * 10 + 5.0,
                                     "angle": 180, "thickness_mm": 8.0 * (1 - risk[pts] * 0.3) + rng.normal(0, 0.1, 15)}))
    pd.concat(prof).to_csv(out / "profile.csv", index=False)
    pd.concat(reg).to_csv(out / "regime.csv", index=False)
    pd.concat(vtd).to_csv(out / "vtd.csv", index=False)
    pd.concat(uzt).to_csv(out / "uzt.csv", index=False)
    return out


if __name__ == "__main__":
    print("синтетика записана в", make(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic"))
