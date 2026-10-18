import numpy as np
import pandas as pd

from src.train import (
    BASE_FEATURES,
    LANE_FEATURES,
    calibration_table,
    evaluate,
    lane_coefficients,
    plot_auc,
    prepare,
)


def synthetic_dataset(n_games: int = 240, minute: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = sorted(set(BASE_FEATURES + LANE_FEATURES))
    df = pd.DataFrame(rng.normal(0, 1, (n_games, len(cols))), columns=cols)
    df["gold_diff"] = df[["top_gold_diff", "jng_gold_diff", "mid_gold_diff",
                          "bot_gold_diff", "sup_gold_diff"]].sum(axis=1)
    p = 1 / (1 + np.exp(-1.5 * df["gold_diff"]))
    df["blue_win"] = (rng.random(n_games) < p).astype(int)
    df["match_id"] = [f"EUW1_{i}" for i in range(n_games)]
    df["minute"] = minute
    return df


def test_evaluate_returns_all_models_and_sane_scores():
    metrics = evaluate(synthetic_dataset(), minutes=(10,), n_splits=3)
    assert set(metrics["model"]) == {"logreg_base", "gbm_base", "logreg_lanes"}
    row = metrics[metrics["model"] == "logreg_base"].iloc[0]
    assert row["auc"] > 0.7
    assert row["auc_lo"] <= row["auc"] <= row["auc_hi"]
    assert 0 <= row["brier"] <= 1


def test_small_minutes_are_skipped():
    metrics = evaluate(synthetic_dataset(n_games=20), minutes=(10,), n_splits=3)
    assert metrics.empty


def test_prepare_drops_rows_with_missing_roles():
    df = synthetic_dataset(n_games=100)
    df.loc[:9, "mid_gold_diff"] = np.nan
    assert len(prepare(df)) == 90


def test_lane_coefficients_and_calibration():
    df = synthetic_dataset()
    coefs = lane_coefficients(df, 10)
    assert set(coefs["feature"]) == set(LANE_FEATURES)
    calib = calibration_table(df, 10, n_splits=3)
    assert {"predit", "observe"} <= set(calib.columns)


def test_plot_is_written(tmp_path):
    metrics = evaluate(synthetic_dataset(), minutes=(10,), n_splits=3)
    out = tmp_path / "auc.png"
    plot_auc(metrics, out)
    assert out.exists() and out.stat().st_size > 0
