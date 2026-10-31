"""Modèles de courbe : phases de jeu, coefficients positifs, comparaison statistique."""
import numpy as np
import pandas as pd
import pytest

from src.monotone import MonotoneLogit
from src.train import BASE_FEATURES, EXTENDED_FEATURES
from src.winprob import (
    compare_models,
    design_for,
    design_matrix,
    evaluate_curve_model,
    fit_curve_model,
    has_positive_effects,
    jump_contributions,
    learning_curve,
    load_dataset,
    paired_logloss_gain,
)


def collinear_games(n_games=300, seed=0, extended=False) -> pd.DataFrame:
    """L'or suit l'avantage réel ; les kills et les Nashor sont très corrélés à l'or."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_games):
        strength = rng.normal(0, 1)
        win = int(strength + rng.logistic(0, 0.5) > 0)
        for m in range(2, 31, 2):
            gold = strength * 120 * m + rng.normal(0, 150)
            row = dict.fromkeys(EXTENDED_FEATURES if extended else BASE_FEATURES, 0.0)
            row.update(gold_diff=gold, xp_diff=gold * 0.8 + rng.normal(0, 100), cs_diff=gold / 50,
                       kills_diff=np.round(gold / 450 + rng.normal(0, 0.4)),
                       barons_diff=float(m > 20 and gold > 2500))
            row.update(match_id=f"EUW1_{g}", patch="16.20", game_creation=g, minute=m, blue_win=win)
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- MonotoneLogit
def test_monotone_logit_keeps_every_effect_positive_on_correlated_data():
    rng = np.random.default_rng(0)
    signal = rng.normal(0, 1, 2000)
    # la seconde variable, à signal égal, est liée à la défaite : une régression libre lui donne un poids négatif
    noise = rng.normal(0, 1, 2000)
    X = np.column_stack([signal + 0.3 * noise, signal - 0.3 * noise])
    y = (signal + 0.5 * noise + rng.logistic(0, 0.3, 2000) > 0).astype(int)
    model = MonotoneLogit(C=10).fit(X, y)
    assert model.coef_.shape == (1, 2)
    assert (model.coef_ >= 0).all()
    assert model.coef_[0, 0] > 0 and model.coef_[0, 1] == 0  # le poids négatif est ramené à zéro


def test_monotone_logit_matches_a_free_logistic_when_effects_are_already_positive():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (2000, 3))
    y = (X @ np.array([1.0, 0.5, 0.2]) + rng.logistic(0, 1, 2000) > 0).astype(int)
    ours = MonotoneLogit(C=1).fit(X, y)
    free = LogisticRegression(C=1, tol=1e-6).fit(X, y)
    assert ours.coef_ == pytest.approx(free.coef_, abs=1e-3)
    assert ours.intercept_ == pytest.approx(free.intercept_, abs=1e-3)
    assert ours.predict_proba(X)[:, 1] == pytest.approx(free.predict_proba(X)[:, 1], abs=1e-3)


# ----------------------------------------------------------------------------- phases de jeu
def test_phase_design_interpolates_the_weight_between_two_knots():
    df = pd.DataFrame({**dict.fromkeys(BASE_FEATURES, 0.0), "minute": [5, 12, 15, 60]})
    df["gold_diff"] = 100.0
    X = design_matrix(df, kind="phases")
    assert list(X["gold_diff@5"]) == [100, 0, 0, 0]
    assert list(X["gold_diff@10"]) == [0, pytest.approx(60), 0, 0]  # 12 min : 3/5 de la phase 10
    assert list(X["gold_diff@15"]) == [0, pytest.approx(40), 100, 0]
    assert list(X["gold_diff@40"]) == [0, 0, 0, 100]  # au-delà de la dernière phase : pas d'extrapolation


def test_phase_design_sums_to_the_variable_at_every_minute():
    df = pd.DataFrame({**dict.fromkeys(BASE_FEATURES, 0.0), "minute": list(range(2, 41))})
    df["gold_diff"] = 7.0
    X = design_matrix(df, kind="phases")
    assert list(X[[c for c in X.columns if c.startswith("gold_diff@")]].sum(axis=1)) == pytest.approx([7.0] * 39)


def test_design_for_rebuilds_the_columns_of_a_saved_linear_model_in_their_order():
    df = collinear_games(60)
    model = fit_curve_model(df)
    X = design_for(model, df)
    assert list(X.columns) == list(model.feature_names_in_)
    assert "gold_diff*t" in X.columns


def test_design_for_rebuilds_the_columns_of_a_monotone_model_on_extended_features():
    df = collinear_games(60, extended=True)
    model = fit_curve_model(df, kind="monotone", features=EXTENDED_FEATURES)
    X = design_for(model, df)
    assert list(X.columns) == list(model.feature_names_in_)
    assert "souls_diff@25" in X.columns


# ----------------------------------------------------------------------------- explication des sauts
def test_free_model_on_correlated_data_is_not_trusted_for_explanations():
    df = collinear_games()
    assert not has_positive_effects(fit_curve_model(df))
    assert has_positive_effects(fit_curve_model(df, kind="monotone"))


def test_contributions_of_a_monotone_model_never_oppose_a_pure_advantage():
    df = collinear_games()
    model = fit_curve_model(df, kind="monotone")
    rows = df.iloc[:2].copy()
    rows[BASE_FEATURES] = 0.0
    rows["minute"] = [21, 22]
    rows.loc[rows.index[1], ["gold_diff", "kills_diff", "barons_diff"]] = [1500.0, 2.0, 1.0]
    X = design_for(model, rows)
    parts = jump_contributions(model, X.iloc[0], X.iloc[1], jump=10.0)
    assert sum(parts.values()) == pytest.approx(10.0, abs=0.2)
    assert all(points >= 0 for points in parts.values())
    assert parts["ressources"] > 0


# ----------------------------------------------------------------------------- comparaison
def test_paired_gain_is_computed_game_by_game():
    y = np.array([1, 1, 0, 0])
    groups = np.array(["a", "a", "b", "b"])
    worse = np.full(4, 0.5)
    better = np.array([0.8, 0.8, 0.2, 0.2])
    gain, lo, hi = paired_logloss_gain(y, worse, better, groups, n=200)
    assert gain == pytest.approx(np.log(0.8 / 0.5))  # toutes les lignes gagnent autant
    assert lo == pytest.approx(gain) and hi == pytest.approx(gain)


def test_paired_gain_interval_contains_zero_when_models_only_differ_by_noise():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    groups = np.arange(2000)
    a = np.clip(0.5 + rng.normal(0, 0.03, 2000), 0.01, 0.99)
    b = np.clip(0.5 + rng.normal(0, 0.03, 2000), 0.01, 0.99)
    _, lo, hi = paired_logloss_gain(y, a, b, groups)
    assert lo < 0 < hi


def test_compare_models_reports_each_model_and_phase_against_the_reference():
    table = compare_models(collinear_games(), n_splits=3, n_boot=50)
    assert set(table["modele"]) == {"lineaire", "phases", "monotone"}  # pas de colonnes étendues ici
    assert {"global", "2-5", "6-10", "11-15", "16-25", "26-40"} == set(table["phase"])
    reference = table[(table["modele"] == "lineaire") & (table["phase"] == "global")].iloc[0]
    assert reference["gain"] == 0 and reference["gain_lo"] == 0 and reference["gain_hi"] == 0
    assert table["ecart_calibration"].between(0, 1).all()
    glob = table[table["phase"] == "global"].set_index("modele")
    assert glob.loc["monotone", "n_parties"] == 300
    assert glob.loc["monotone", "auc"] > 0.75


def test_compare_models_adds_the_extended_model_when_the_columns_exist():
    table = compare_models(collinear_games(80, extended=True), n_splits=2, n_boot=10)
    assert "monotone_etendu" in set(table["modele"])


def test_compare_models_can_measure_on_the_most_recent_games_only():
    df = collinear_games(100)
    table = compare_models(df, time_split=0.2, n_boot=10)
    assert (table[table["phase"] == "global"]["n_parties"] == 20).all()


def test_monotone_model_stays_close_to_the_free_one_when_data_is_correlated():
    table = compare_models(collinear_games(), n_splits=3, n_boot=50)
    glob = table[table["phase"] == "global"].set_index("modele")
    assert glob.loc["monotone", "auc"] == pytest.approx(glob.loc["lineaire", "auc"], abs=0.02)


def test_learning_curve_keeps_one_test_set_and_grows_the_training_set():
    table = learning_curve(collinear_games(), sizes=(50, 100, 10_000))
    assert list(table["n_train"]) == [50, 100, 240]  # 240 = toutes les parties hors test
    assert (table["n_test"] == 60).all()
    assert table["log_loss"].iloc[-1] < table["log_loss"].iloc[0] + 0.02


# ----------------------------------------------------------------------------- évaluation
def test_curve_evaluation_reports_a_confidence_interval():
    scores = evaluate_curve_model(collinear_games(120), n_splits=3)
    assert (scores["auc_lo"] <= scores["auc"]).all() and (scores["auc"] <= scores["auc_hi"]).all()


def test_time_split_and_patch_holdout_report_a_confidence_interval():
    from src.winprob import evaluate_patch_holdout, evaluate_time_split

    df = collinear_games(200)
    df["minute"] = df["minute"].where(df["minute"] != 6, 5)  # les minutes repères 5, 10... existent
    ts = evaluate_time_split(df, 0.3)
    assert {"auc_lo", "auc_hi"} <= set(ts.columns)
    df["patch"] = np.where(df["game_creation"] < 100, "16.19", "16.20")
    ph = evaluate_patch_holdout(df, min_games=100)
    assert {"auc_lo", "auc_hi"} <= set(ph.columns)


# ----------------------------------------------------------------------------- données et erreurs
def test_load_dataset_explains_a_missing_file(tmp_path):
    with pytest.raises(SystemExit, match="build_dataset"):
        load_dataset(tmp_path / "absent.csv")


def test_analyze_explains_a_missing_model(tmp_path):
    import argparse

    from src.winprob import cmd_analyze

    args = argparse.Namespace(model=str(tmp_path / "absent.joblib"), raw_dir=str(tmp_path), me_dir=str(tmp_path),
                              match_id=None, limit=1, skip_existing=False, top=3, out=str(tmp_path / "out"))
    with pytest.raises(SystemExit, match="winprob train"):
        cmd_analyze(args)


@pytest.mark.parametrize("command", [["compare", "--time-split", "0.2"], ["learning-curve"]])
def test_commands_explain_that_an_old_dataset_must_be_rebuilt(tmp_path, command):
    from src.winprob import main

    path = tmp_path / "ancien.csv"
    collinear_games(60).drop(columns="game_creation").to_csv(path, index=False)
    with pytest.raises(SystemExit, match="game_creation.*build_dataset"):
        main([*command, "--data", str(path)])


def test_train_explains_that_extended_features_need_a_rebuilt_dataset(tmp_path):
    import argparse

    from src.winprob import cmd_train

    path = tmp_path / "ancien.csv"
    collinear_games(60).to_csv(path, index=False)
    args = argparse.Namespace(data=str(path), model=str(tmp_path / "m.joblib"), time_split=0.0, patch_holdout=False,
                              calibration=False, kind="monotone", features="etendu")
    with pytest.raises(SystemExit, match="build_dataset"):
        cmd_train(args)
