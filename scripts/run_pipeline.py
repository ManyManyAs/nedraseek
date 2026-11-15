"""data/raw/ -> result.csv и таблицы в outputs/.

Запуск из корня:  python scripts/run_pipeline.py [--config config.yaml]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipetwin import data, pipeline, submission  # noqa: E402


def main(config: str = "config.yaml") -> pd.DataFrame:
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    raw, out = cfg["paths"]["raw"], Path(cfg["paths"]["out"])
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)

    # 1. Загрузка и очистка
    P, V, prof = data.load_passport(raw), data.load_vtd(raw), data.load_profile(raw)
    J = data.load_joints(raw)
    pp = P.set_index("id")
    print(f"[1] участков: {len(P)}, дефектов потери металла: {len(V)}, пропусков ВТД: {V['date'].nunique()}")

    # 2. Кейс 1
    r1 = pipeline.run_case1(cfg, P, V, J)
    t1 = int(cfg["case1"]["target_id"])
    lo, hi = r1["drift_ends"]
    print(f"[2] кейс 1, участок {t1}: сдвиг дистанций {lo:.2f} м в начале, {hi:.2f} м в конце; "
          f"ошибка положения в середине ±{r1['pos_sigma']:.1f} м")
    print(f"    независимая проверка: участок {r1['val_section']}, блоки сегментов скрыты целиком")
    print(r1["validation"].round(3).to_string(index=False))
    print(f"    покрытие интервала P10–P90: {r1['coverage']:.0%}")
    print("    r² по блокам:")
    print(r1["by_block"].round(2).to_string(index=False))
    if len(r1["repairs"]):
        print(f"    трубы с предписанием «врезка катушки» (вероятность замены {cfg['case1']['repair_prob']}):")
        print(r1["repairs"].round(2).to_string(index=False))

    # 3. Кейс 2
    c2 = pipeline.run_case2(cfg, P, prof, V)
    r2 = c2["pred"]
    t2 = int(cfg["case2"]["target_id"])
    print(f"[3] кейс 2, участок {t2}: {len(r2)} сегментов, прогноз по сглаженной высотной отметке")
    print(f"    сила признака на первых пропусках ВТД других трубопроводов до T = {cfg['case2']['target_date']}:")
    print(c2["by_pipeline"].round(3).to_string(index=False))

    # 4. Запись
    both = pd.concat([r1["pred"], r2], ignore_index=True)
    # Ожидаемая сетка строится независимо от ответа: из паспортной длины целевых участков.
    grid = pd.concat([pd.DataFrame({"id": i, "segment_id": range(int(pp.loc[i, "n_seg"]))}) for i in (t1, t2)],
                     ignore_index=True)
    submission.write_result(both[["id", "segment_id", "thickness_pred"]], grid, cfg["paths"]["result"])
    both[["id", "segment_id", "thickness_pred", "p10", "p90"]].round(4).to_csv(out / "result_with_intervals.csv", index=False)
    r1["validation"].round(4).to_csv(out / "validation_case1.csv", index=False)
    r1["by_block"].round(4).to_csv(out / "validation_case1_blocks.csv", index=False)
    r1["val_points"].round(4).to_csv(out / "validation_case1_points.csv", index=False)
    r1["horizons"].round(4).to_csv(out / "horizons_case1.csv", index=False)
    r1["joint_shifts"].drop(columns=["date_a", "date_b"]).round(3).to_csv(out / "joint_shifts_case1.csv", index=False)
    r1["linearity"].round(3).to_csv(out / "drift_linearity_case1.csv", index=False)
    r1["repairs"].round(3).to_csv(out / "repairs_case1.csv", index=False)
    r1["pred"].round(4).to_csv(out / "case1_segments.csv", index=False)
    r2.round(4).to_csv(out / "case2_segments.csv", index=False)
    c2["reference"].round(4).to_csv(out / "validation_case2.csv", index=False)
    top = pd.concat([pipeline.top_segments(r1["pred"], pp.loc[t1, "length_m"]),
                     pipeline.top_segments(r2, pp.loc[t2, "length_m"])])
    top.round(3).to_csv(out / "top10_segments.csv", index=False)
    pipeline.top_intervals(top).round(3).to_csv(out / "top10_intervals.csv", index=False)
    # Сводная таблица подходов для слайдов: одна выборка, одна схема проверки.
    r1["validation"].assign(case=1).round(3).to_csv(out / "approaches.csv", index=False)
    gm = r1["model"]
    c1 = cfg["case1"]
    summary = {
        "case1": {"target_id": t1, "target_date": cfg["case1"]["target_run"], "n_segments": int(len(r1["pred"])),
                  "validation_section": int(r1["val_section"]),
                  "validation_scheme": f"{c1['val_blocks']} блоков сегментов скрыты по очереди, зазор {c1['val_gap_m']} м; "
                                       "сдвиг и закон роста оцениваются без скрытого блока",
                  "validation": r1["validation"].round(4).to_dict("records"),
                  "p10_p90_coverage": round(r1["coverage"], 3),
                  "matched_pairs": int(gm.n_pairs), "new_defects_per_old": round(gm.new_rate, 3),
                  "shift_start_m": round(lo, 2), "shift_end_m": round(hi, 2),
                  "pos_sigma_final_m": round(r1["pos_sigma"], 2), "pos_sigma_validation_m": c1["val_pos_sigma_m"],
                  "blend_w": c1["blend_w"], "repair_prob": c1["repair_prob"],
                  "repair_joints": r1["repairs"].round(2).to_dict("records"),
                  "repair_sensitivity_segments": r1["pred"].loc[r1["pred"]["repair_flag"],
                      ["segment_id", "if_no_repair", "if_repaired", "thickness_pred"]].round(2).to_dict("records")},
        "case2": {"target_id": t2, "target_date": cfg["case2"]["target_date"], "n_segments": int(len(r2)),
                  "method": "линейная функция высотной отметки, сглаженной в окне %d м" % cfg["case2"]["smooth_window_m"],
                  "reference_mean_loss": round(c2["mean_loss"], 3), "reference_p90_loss": round(c2["high_loss"], 3),
                  "feature_r2_by_pipeline": c2["by_pipeline"].round(4).to_dict("records")},
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[4] записан {cfg['paths']['result']}: {len(both)} строк; таблицы: в {out}/")
    return both


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
