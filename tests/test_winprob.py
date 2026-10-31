import numpy as np
import pandas as pd
import pytest

from src.features import build_rows
from src.winprob import (
    analyze_match,
    collapse_duplicates,
    describe_events,
    evaluate_curve_model,
    find_turning_points,
    fit_curve_model,
    plot_curve,
)

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


def participants():
    ps = []
    for i, role in enumerate(ROLES):
        ps.append({"participantId": i + 1, "teamId": 100, "teamPosition": role,
                   "championName": f"Blue{i + 1}", "puuid": f"b{i + 1}"})
        ps.append({"participantId": i + 6, "teamId": 200, "teamPosition": role,
                   "championName": f"Red{i + 1}", "puuid": f"r{i + 1}"})
    return ps


def fake_game(diffs, match_id="EUW1_1", events_by_minute=None):
    """Partie dont l'écart d'or bleu - rouge vaut diffs[m] à la minute m."""
    frames = []
    for m, d in enumerate(diffs):
        pf = {}
        for pid in range(1, 11):
            sign = 1 if pid <= 5 else -1
            pf[str(pid)] = {"totalGold": 3000 + 300 * m + sign * d / 10,
                            "xp": 1000 + 200 * m + sign * d / 10, "level": 1 + m // 2,
                            "minionsKilled": 8 * m, "jungleMinionsKilled": 0}
        frames.append({"timestamp": m * 60_000, "participantFrames": pf,
                       "events": (events_by_minute or {}).get(m, [])})
    blue_win = diffs[-1] > 0
    match = {
        "metadata": {"matchId": match_id},
        "info": {"gameVersion": "15.20.1", "gameCreation": 1,
                 "teams": [{"teamId": 100, "win": blue_win}, {"teamId": 200, "win": not blue_win}],
                 "participants": participants()},
    }
    return match, {"info": {"frames": frames}}


def training_frame(n_games=80, n_frames=26, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_games):
        drift = rng.normal(0, 1)
        diffs = np.cumsum(rng.normal(drift * 120, 250, n_frames))
        match, timeline = fake_game(diffs, f"EUW1_{g}")
        rows += build_rows(match, timeline, minutes=range(2, n_frames))
    return pd.DataFrame(rows)


def test_curve_model_is_predictive_late_game():
    df = training_frame()
    scores = evaluate_curve_model(df, n_splits=3).set_index("minute")
    assert scores.loc[20, "auc"] > 0.8
    assert scores.loc[20, "auc"] >= scores.loc[5, "auc"]


def test_find_turning_points_picks_biggest_jumps_and_skips_gaps():
    minutes = [2, 3, 4, 5, 7, 8]
    probs = [0.5, 0.52, 0.80, 0.82, 0.30, 0.35]  # saut 5->7 ignoré (minute manquante)
    tps = find_turning_points(minutes, probs, k=2, min_delta=0.03)
    assert [t["minute"] for t in tps] == [4, 8]
    assert tps[0]["delta"] > 0.25


def test_find_turning_points_ignores_small_moves():
    assert find_turning_points([2, 3, 4], [0.5, 0.51, 0.52], min_delta=0.03) == []


def test_describe_events_uses_side_and_prioritises_objectives():
    events = [
        {"type": "CHAMPION_KILL", "killerId": 3, "victimId": 8},
        {"type": "ELITE_MONSTER_KILL", "killerTeamId": 200, "monsterType": "BARON_NASHOR"},
        {"type": "BUILDING_KILL", "teamId": 100, "buildingType": "TOWER_BUILDING"},
    ]
    texts = describe_events(events, participants(), my_team=100)
    assert texts[0] == "Les ennemis prennent Nashor"
    assert texts[1] == "Les ennemis détruisent une tour"
    assert texts[2] == "Blue3 (alliés) tue Red3 (ennemis)"
    assert describe_events(events, participants(), None)[0] == "Les rouges prennent Nashor"


def test_describe_events_skips_a_monster_without_a_playing_team():
    events = [{"type": "ELITE_MONSTER_KILL", "killerTeamId": 300, "monsterType": "DRAGON"}]
    assert describe_events(events, participants(), my_team=100) == []


def test_analyze_match_finds_the_swing_and_flips_perspective():
    model = fit_curve_model(training_frame())
    diffs = [0] * 10 + [5000] * 10  # bascule à la minute 10
    baron = {10: [{"type": "ELITE_MONSTER_KILL", "killerTeamId": 100, "monsterType": "BARON_NASHOR",
                   "timestamp": 590_000}]}
    match, timeline = fake_game(diffs, events_by_minute=baron)

    blue = analyze_match(model, match, timeline, puuid="b1")
    assert blue["perspective"]["result"] == "victoire"
    tp = blue["turning_points"][0]
    assert tp["minute"] == 10
    assert tp["win_prob_after"] > tp["win_prob_before"] + 0.2
    assert "Les alliés prennent Nashor" in tp["events"]

    red = analyze_match(model, match, timeline, puuid="r1")
    assert red["perspective"]["result"] == "défaite"
    assert red["turning_points"][0]["win_prob_after"] < red["turning_points"][0]["win_prob_before"]
    assert "Les ennemis prennent Nashor" in red["turning_points"][0]["events"]


def test_design_matrix_does_not_extrapolate_past_training_minutes():
    from src.winprob import design_matrix

    df = training_frame().iloc[:2].copy()
    late, later = df.copy(), df.copy()
    late["minute"], later["minute"] = 40, 65
    assert design_matrix(late).equals(design_matrix(later))


def test_short_game_returns_none():
    model = fit_curve_model(training_frame())
    match, timeline = fake_game([0, 0])
    assert analyze_match(model, match, timeline) is None


def test_plot_is_written(tmp_path):
    model = fit_curve_model(training_frame())
    match, timeline = fake_game([0] * 10 + [5000] * 10)
    analysis = analyze_match(model, match, timeline, puuid="b1")
    out = tmp_path / "curve.png"
    plot_curve(analysis, out)
    assert out.exists() and out.stat().st_size > 0


def test_repeated_events_are_grouped():
    assert collapse_duplicates(["a", "a", "b", "a"]) == ["a (×3)", "b"]
    events = [{"type": "ELITE_MONSTER_KILL", "killerTeamId": 100, "monsterType": "HORDE"}] * 3
    assert describe_events(events, participants(), my_team=100) == [
        "Les alliés prennent des Larves du Néant (×3)"]


# ----------------------------------------------------------------------------- calibration
def test_calibration_by_phase_compares_predicted_and_observed_win_rates():
    from src.winprob import calibration_by_phase

    # 2-5 min : on annonce 80 % mais seule 1 partie sur 2 est gagnée
    y = np.array([1, 0, 1, 0, 1, 1])
    p = np.array([0.8, 0.8, 0.8, 0.8, 0.6, 0.6])
    minutes = np.array([3, 3, 4, 4, 12, 12])
    table = calibration_by_phase(y, p, minutes, n_bins=1)
    early = table[table["phase"] == "2-5"].iloc[0]
    assert early["predit"] == pytest.approx(0.8) and early["observe"] == 0.5 and early["n"] == 4
    assert early["ecart"] == pytest.approx(0.3)
    late = table[table["phase"] == "11-15"].iloc[0]
    assert late["observe"] == 1.0 and late["n"] == 2
    assert set(table["phase"]) == {"2-5", "11-15"}  # les phases sans ligne n'apparaissent pas


def test_curve_calibration_covers_each_phase_with_every_row():
    from src.winprob import curve_calibration

    df = training_frame(n_games=100, n_frames=26)  # minutes 2 à 25
    table = curve_calibration(df, n_splits=3, n_bins=3)
    assert set(table["phase"]) == {"2-5", "6-10", "11-15", "16-25"}
    assert table["n"].sum() == len(df)


# ----------------------------------------------------------------------------- explication des sauts
def test_jump_contributions_add_up_to_the_jump_and_name_the_family():
    model = fit_curve_model(training_frame(), kind="monotone")
    # seule la famille « ressources » bouge à la minute 10
    match, timeline = fake_game([0] * 10 + [5000] * 10)
    tp = analyze_match(model, match, timeline, puuid="b1")["turning_points"][0]
    parts = tp["contributions"]
    assert set(parts) == {"ressources", "combats", "objectifs", "vision"}
    jump = (tp["win_prob_after"] - tp["win_prob_before"]) * 100
    assert sum(parts.values()) == pytest.approx(jump, abs=0.5)
    assert parts["ressources"] == pytest.approx(jump, abs=0.5)
    assert parts["objectifs"] == 0 and parts["combats"] == 0


def test_jump_contributions_change_sign_with_the_perspective():
    model = fit_curve_model(training_frame(), kind="monotone")
    match, timeline = fake_game([0] * 10 + [5000] * 10)
    blue = analyze_match(model, match, timeline, puuid="b1")["turning_points"][0]["contributions"]
    red = analyze_match(model, match, timeline, puuid="r1")["turning_points"][0]["contributions"]
    assert blue["ressources"] > 0 > red["ressources"]


def test_jump_contributions_credit_a_baron_to_objectives():
    baron = {10: [{"type": "ELITE_MONSTER_KILL", "killerTeamId": 200, "monsterType": "BARON_NASHOR",
                   "timestamp": 590_000}]}
    match, timeline = fake_game([0] * 20, events_by_minute=baron)
    # le modèle d'entraînement ne voit aucun Nashor : on vérifie sur un modèle qui lui donne du poids
    df = training_frame()
    df["barons_diff"] = np.where(df["blue_win"] == 1, 1, -1)
    objective_model = fit_curve_model(df, kind="monotone")
    tp = next(t for t in analyze_match(objective_model, match, timeline, puuid="b1", min_delta=0.01)["turning_points"]
              if t["minute"] == 10)
    assert tp["contributions"]["objectifs"] < 0
    assert tp["contributions"]["ressources"] == 0


def test_free_model_with_a_negative_weight_gives_no_contributions():
    df = training_frame()
    df["kills_diff"] = np.where(df["blue_win"] == 1, -1, 1)  # à or égal, plus de kills = défaite
    match, timeline = fake_game([0] * 10 + [5000] * 10)
    tp = analyze_match(fit_curve_model(df), match, timeline, puuid="b1")["turning_points"][0]
    assert "contributions" not in tp


def test_turning_point_lists_the_gaps_that_moved_from_the_player_side():
    model = fit_curve_model(training_frame())
    baron = {10: [{"type": "ELITE_MONSTER_KILL", "killerTeamId": 100, "monsterType": "BARON_NASHOR",
                   "timestamp": 590_000}]}
    match, timeline = fake_game([0] * 10 + [5000] * 10, events_by_minute=baron)
    blue = analyze_match(model, match, timeline, puuid="b1")["turning_points"][0]
    assert blue["ecarts"] == {"or": 5000, "XP": 5000, "Nashor": 1}
    red = analyze_match(model, match, timeline, puuid="r1")["turning_points"][0]
    assert red["ecarts"] == {"or": -5000, "XP": -5000, "Nashor": -1}


def test_format_gaps_shows_signed_values_in_a_stable_order():
    from src.winprob import format_gaps

    assert format_gaps({"or": 2150, "kills": -1}) == "or +2150, kills -1"
    assert format_gaps({}) == ""


def test_moment_label_tells_how_many_events_the_chart_leaves_out():
    from src.winprob import NO_EVENT, moment_label

    assert moment_label({"events": ["Sivir tue Draven", "Udyr tue Katarina", "Udyr tue Shaco"]}) \
        == "Sivir tue Draven, +2 autres"
    assert moment_label({"events": ["Sivir tue Draven"]}) == "Sivir tue Draven"
    assert moment_label({"events": []}) == NO_EVENT


# ----------------------------------------------------------------------------- nouvelles parties
def _write_match(raw_dir, match_id, created):
    import json
    (raw_dir / "matches").mkdir(parents=True, exist_ok=True)
    (raw_dir / "matches" / f"{match_id}.json").write_text(
        json.dumps({"info": {"gameCreation": created}}), encoding="utf-8")


def test_select_ids_takes_the_most_recent_games(tmp_path):
    from src.winprob import select_ids

    for i, created in enumerate([300, 100, 200]):
        _write_match(tmp_path, f"EUW1_{i}", created)
    assert select_ids(tmp_path, tmp_path / "curves", limit=2) == ["EUW1_0", "EUW1_2"]


def test_select_ids_skip_existing_keeps_only_games_without_a_curve(tmp_path):
    from src.winprob import select_ids

    for i, created in enumerate([300, 100, 200]):
        _write_match(tmp_path, f"EUW1_{i}", created)
    curves = tmp_path / "curves"
    curves.mkdir()
    (curves / "EUW1_0.json").write_text("{}", encoding="utf-8")  # la plus récente est déjà analysée
    assert select_ids(tmp_path, curves, limit=5, skip_existing=True) == ["EUW1_2", "EUW1_1"]
    # le plafond s'applique après le filtre, pas avant
    assert select_ids(tmp_path, curves, limit=1, skip_existing=True) == ["EUW1_2"]


# ----------------------------------------------------------------------------- test dans le temps
def test_time_split_trains_on_old_games_and_tests_on_the_newest():
    import numpy as np
    import pandas as pd
    from src.train import BASE_FEATURES
    from src.winprob import REPORT_MINUTES, evaluate_time_split

    rng = np.random.default_rng(0)
    rows = []
    for g in range(200):
        gold = rng.normal(0, 1500)
        win = int(gold + rng.normal(0, 1500) > 0)
        for m in REPORT_MINUTES:
            row = {"match_id": f"EUW1_{g}", "game_creation": 1_000 + g, "minute": m, "blue_win": win}
            row.update({f: (gold if f == "gold_diff" else 0.0) for f in BASE_FEATURES})
            rows.append(row)
    ts = evaluate_time_split(pd.DataFrame(rows), test_fraction=0.25)
    assert ts.attrs["n_test"] == 50 and ts.attrs["n_train"] == 150
    assert list(ts["minute"]) == list(REPORT_MINUTES)
    assert (ts["auc"] > 0.7).all()  # le signal appris sur les anciennes parties vaut sur les récentes


def test_patch_holdout_tests_each_big_patch_on_the_others():
    import numpy as np
    import pandas as pd
    from src.train import BASE_FEATURES
    from src.winprob import REPORT_MINUTES, evaluate_patch_holdout

    rng = np.random.default_rng(1)
    rows = []
    for g in range(380):
        patch = "16.19" if g < 20 else ("16.20" if g < 200 else "16.21")
        gold = rng.normal(0, 1500)
        win = int(gold + rng.normal(0, 1500) > 0)
        for m in REPORT_MINUTES:
            row = {"match_id": f"EUW1_{g}", "patch": patch, "minute": m, "blue_win": win}
            row.update({f: (gold if f == "gold_diff" else 0.0) for f in BASE_FEATURES})
            rows.append(row)
    ph = evaluate_patch_holdout(pd.DataFrame(rows), min_games=100)
    assert sorted(ph["patch"].unique()) == ["16.20", "16.21"]  # 16.19 (20 parties) ne sert qu'à l'entraînement
    first = ph[ph["patch"] == "16.20"].iloc[0]
    assert first["n_train"] == 200  # 380 - 180 parties du patch testé
    assert (ph["auc"] > 0.7).all()


def test_patches_16_1_and_16_10_stay_distinct_after_a_csv_round_trip(tmp_path):
    import numpy as np
    import pandas as pd
    from src.train import BASE_FEATURES
    from src.winprob import REPORT_MINUTES, evaluate_patch_holdout, load_dataset

    rng = np.random.default_rng(2)
    rows = []
    for g in range(200):
        gold = rng.normal(0, 1500)
        win = int(gold + rng.normal(0, 1500) > 0)
        for m in REPORT_MINUTES:
            row = {"match_id": f"EUW1_{g}", "patch": "16.1" if g < 100 else "16.10",
                   "minute": m, "blue_win": win}
            row.update({f: (gold if f == "gold_diff" else 0.0) for f in BASE_FEATURES})
            rows.append(row)
    path = tmp_path / "dataset.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    ph = evaluate_patch_holdout(load_dataset(path), min_games=100)
    assert sorted(ph["patch"].unique()) == ["16.1", "16.10"]
    assert (ph["n_train"] == 100).all()
