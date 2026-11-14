"""Кейс 2: прогноз от ввода в эксплуатацию, без единого пропуска ВТД на участке.

Что известно для такого участка: паспорт, рельеф и режим. Режим один на весь участок, поэтому различать
сегменты может только рельеф. Проверка на первых пропусках ВТД трёх других трубопроводов показала:

  * связь рельефа с толщиной слабая (r² 0,01–0,08) и имеет разный знак на разных трубопроводах;
  * самый устойчивый признак: высотная отметка, сглаженная в окне 1 км; мелкие формы рельефа
    (ямы в сотни метров, уклон, кривизна) сигнала не дают.

Метрика организаторов (квадрат корреляции) не зависит от знака связи. Поэтому прогноз строится
как линейная функция сглаженной отметки: он использует единственный признак с подтверждённым сигналом
и не смешивает признаки с неизвестными знаками, которые могли бы погасить друг друга.
Направление выбрано по физике (в низких местах скапливаются вода и осадок) и важно только для списка
опасных сегментов. Точность прогноза ожидаемо низкая; это отражено в документации.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import SEG_LEN, segment_state
from .features import profile_features
from .validate import pearson_r2


def elevation_score(pf: pd.DataFrame, window_m: float = 1000.0) -> np.ndarray:
    """Оценка риска: чем ниже сглаженная отметка, тем выше. Стандартизована внутри участка."""
    w = max(3, int(window_m / SEG_LEN) | 1)
    w = min(w, max(3, (len(pf) // 3) | 1))   # на коротком участке окно не должно съесть весь рельеф
    z = pf["p_z"].rolling(w, center=True, min_periods=1).mean()
    sd = z.std()
    return (-(z - z.mean()) / (sd if sd > 0 else 1.0)).to_numpy()


def predict(profile: pd.DataFrame, pipe_id: int, length_m: float, wall_mm: float, mean_loss: float,
            high_loss: float, spread: float = 0.5, window_m: float = 1000.0) -> pd.DataFrame:
    """mean_loss и high_loss: средняя и высокая (90-й перцентиль) относительная потеря стенки
    на первых пропусках ВТД других трубопроводов; они задают масштаб, а не форму прогноза."""
    pf = profile_features(profile, pipe_id, length_m)
    if pf["p_z"].std() == 0:
        raise ValueError(f"участок {pipe_id}: профиль трассы отсутствует или плоский: различить сегменты нечем")
    score = elevation_score(pf, window_m)
    loss = np.clip(mean_loss * (1.0 + spread * score), 0.0, 0.9)
    return pd.DataFrame({
        "segment_id": pf["segment_id"],
        "thickness_pred": wall_mm * (1.0 - loss),
        "p10": wall_mm * (1.0 - np.clip(high_loss * (1.0 + spread * score), 0.0, 0.95)),  # сценарий «как у изношенных труб»
        "p90": np.full(len(pf), float(wall_mm)),                                           # сценарий «дефектов нет»
        "risk_score": score,
    })


def reference_table(P: pd.DataFrame, V: pd.DataFrame, profile: pd.DataFrame, exclude_ids, window_m: float = 1000.0,
                    before=None):
    """Первые пропуски ВТД других трубопроводов: масштаб потерь и проверка признака.

    before: дата T. Пропуски с этой даты и позже не используются, чтобы прогноз не видел будущего.
    Возвращает (таблица по участкам, средняя потеря, 90-й перцентиль потери).
    Признак не обучается, поэтому r² по каждому участку служит независимой оценкой его силы.
    """
    if before is not None:
        V = V[V["date"] < pd.Timestamp(before)]
    rows, losses = [], []
    for r in P.itertuples():
        if r.id in exclude_ids:
            continue
        runs = sorted(V.loc[V["id"] == r.id, "date"].unique())
        if not runs:
            continue
        d = V[(V["id"] == r.id) & (V["date"] == runs[0])]
        st = segment_state(d, r.length_m, float(d["wt"].median()))
        losses.append(((st["wt"] - st["thk_min"]) / st["wt"]).to_numpy())
        if r.n_seg < 30:
            continue
        pf = profile_features(profile, r.id, r.length_m)
        if pf["p_z"].std() == 0:
            continue
        score = elevation_score(pf, window_m)
        c = float(np.corrcoef(score, -st["thk_min"])[0, 1]) if st["thk_min"].std() > 0 else np.nan
        rows.append({"pipeline": r.pipeline, "id": r.id, "run": pd.Timestamp(runs[0]).date(),
                     "years_in_service": round((pd.Timestamp(runs[0]) - r.start_date).days / 365.25, 1),
                     "n_segments": r.n_seg, "share_with_defects": round(float(st["has_def"].mean()), 3),
                     "r2": pearson_r2(st["thk_min"], -score), "low_is_thinner": bool(c > 0) if c == c else None})
    loss = np.concatenate(losses)
    return pd.DataFrame(rows), float(loss.mean()), float(np.quantile(loss, 0.9))
