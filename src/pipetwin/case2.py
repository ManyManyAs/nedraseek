"""Кейс 2: прогноз от ввода в эксплуатацию, без единого пропуска ВТД на участке.

Проверка «оставь один трубопровод» показала, что переносимой статистической связи между
рельефом и утонением в данных нет (r² около нуля на всех трубопроводах, кроме старой
обводнённой трубы). Поэтому здесь не обученная модель, а прозрачная физическая оценка:
при низкой скорости потока отделившаяся вода и осадок собираются в низких местах трассы,
и стенка там тоньшает быстрее. Оценка честно слабая; это отражено в документации.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import profile_features


def relief_score(pf: pd.DataFrame) -> np.ndarray:
    """Оценка риска 0..1: низкая абсолютная отметка и глубина низины в масштабе 1 км и 300 м."""
    low_z = (-pf["p_z"]).rank(pct=True)
    pit_l = pf["p_pit1000"].rank(pct=True)
    pit_s = pf["p_pit300"].rank(pct=True)
    s = 0.4 * low_z + 0.4 * pit_l + 0.2 * pit_s
    s = s.rolling(3, center=True, min_periods=1).mean()  # соседние сегменты похожи
    return ((s - s.min()) / max(s.max() - s.min(), 1e-9)).to_numpy()


def predict(profile: pd.DataFrame, pipe_id: int, length_m: float, wall_mm: float,
            mean_loss_frac: float = 0.10, spread: float = 0.6) -> pd.DataFrame:
    pf = profile_features(profile, pipe_id, length_m)
    score = relief_score(pf)
    loss = mean_loss_frac * (1.0 + spread * (score - score.mean()) / max(score.std(), 1e-9) / 2.0)
    loss = np.clip(loss, 0.0, 0.9)
    pred = wall_mm * (1.0 - loss)
    return pd.DataFrame({
        "segment_id": pf["segment_id"], "thickness_pred": pred,
        "p10": wall_mm * (1.0 - np.clip(loss * 2.5, 0, 0.95)),   # грубая граница: потеря в 2,5 раза выше
        "p90": np.full(len(pf), wall_mm),                         # дефектов нет — стенка номинальная
        "risk_score": score,
    })
