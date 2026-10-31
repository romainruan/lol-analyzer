import numpy as np
import pandas as pd

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
