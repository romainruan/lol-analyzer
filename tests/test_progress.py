import json

import numpy as np
import pandas as pd
import pytest
from synthetic import make_game

from src.benchmarks import STAT_COLS
from src.progress import champion_table, main_role, player_history, priorities, trend_table


def history_of(values, role="MIDDLE", champion="Ahri", win=1):
    """Un historique dont toutes les statistiques valent `values`, de la plus ancienne partie à la plus récente."""
    return pd.DataFrame({**{c: list(values) for c in STAT_COLS}, "role": role, "champion": champion, "win": win,
                         "game_creation": range(len(values))})


def _write(raw_dir, game):
    match, timeline = game
    name = match["metadata"]["matchId"] + ".json"
    (raw_dir / "matches").mkdir(parents=True, exist_ok=True)
    (raw_dir / "timelines").mkdir(parents=True, exist_ok=True)
    (raw_dir / "matches" / name).write_text(json.dumps(match), encoding="utf-8")
    (raw_dir / "timelines" / name).write_text(json.dumps(timeline), encoding="utf-8")


def test_history_keeps_only_the_player_from_oldest_to_newest(tmp_path):
    for index in (5, 1, 3):
        _write(tmp_path, make_game(index, me="moi", champion="Ahri"))
    history = player_history(tmp_path, "moi")
    assert len(history) == 3
    assert list(history["game_creation"]) == sorted(history["game_creation"])
    assert set(history["puuid"]) == {"moi"}


def test_history_is_empty_without_any_game(tmp_path):
    assert player_history(tmp_path, "moi").empty


def test_main_role_is_the_most_played_recently():
    history = pd.concat([history_of([1.0] * 30, role="TOP"), history_of([1.0] * 6, role="JUNGLE")])
    assert main_role(history, window=3) == "JUNGLE"  # 6 dernières parties : toutes en jungle
    assert main_role(history, window=30) == "TOP"
    assert main_role(pd.DataFrame()) is None


def test_trend_reports_a_clear_improvement():
    table = trend_table(history_of([5.0, 5.2, 4.8, 5.1, 4.9] + [9.0, 9.2, 8.8, 9.1, 8.9]), window=5).set_index("stat")
    row = table.loc["cs_per_min"]
    assert row["precedent"] == pytest.approx(5.0)
    assert row["recent"] == pytest.approx(9.0)
    assert row["evolution"] == pytest.approx(4.0)
    assert row["verdict"] == "en progrès"


def test_trend_reports_a_clear_decline():
    table = trend_table(history_of([9.0, 9.2, 8.8, 9.1, 8.9] + [5.0, 5.2, 4.8, 5.1, 4.9]), window=5).set_index("stat")
    assert table.loc["kda", "verdict"] == "en baisse"


def test_trend_does_not_conclude_from_noise():
    rng = np.random.default_rng(0)
    table = trend_table(history_of(rng.normal(5, 2, 40)), window=20)
    assert set(table["verdict"]) == {"pas de conclusion"}


def test_trend_needs_enough_games_on_both_sides():
    table = trend_table(history_of([5.0] * 7), window=5).set_index("stat")  # 5 récentes, 2 précédentes seulement
    assert table.loc["kda", "verdict"] == "pas assez de parties"
    assert np.isnan(table.loc["kda", "evolution"])
    assert trend_table(pd.DataFrame(), window=5).empty


def test_champion_table_sorts_by_games_and_counts_wins():
    history = pd.concat([history_of([2.0] * 3, champion="Ahri", win=1),
                         history_of([4.0, 4.0, 4.0, 4.0], champion="Zed", win=0)])
    table = champion_table(history)
    assert list(table["champion"]) == ["Zed", "Ahri"]
    assert list(table["parties"]) == [4, 3]
    assert list(table["victoires_pct"]) == [0, 100]
    assert list(table["kda"]) == [4.0, 2.0]


def _bench(role="MIDDLE"):
    return pd.DataFrame({**{c: np.linspace(1, 10, 100) for c in STAT_COLS}, "role": role, "champion": "Zed"})


def test_priorities_are_the_lowest_stats_against_the_reference():
    history = history_of([5.5] * 10)
    history["vision_per_min"] = 1.5
    history["kda"] = 2.0
    history["cs_per_min"] = 9.0
    result = priorities(history, _bench(), "MIDDLE", window=10)
    assert [p["stat"] for p in result] == ["vision_per_min", "kda"]
    assert result[0]["percentile"] == 6
    assert result[0]["reference"] == "mid Master"
    assert "Conseil : poser et nettoyer plus de balises" in result[0]["phrase"]
    assert "sur vos 10 dernières parties" in result[0]["phrase"]


def test_priorities_only_use_recent_games_of_the_role():
    old = history_of([9.0] * 10)
    recent = history_of([1.5] * 3)
    other_role = history_of([9.0] * 3, role="TOP")
    result = priorities(pd.concat([old, recent, other_role]), _bench(), "MIDDLE", window=3)
    assert result[0]["moyenne"] == pytest.approx(1.5)


def test_support_priorities_ignore_farm_and_damage():
    history = history_of([5.5] * 10, role="UTILITY")
    history["cs_per_min"] = 1.0
    history["cs_at_10"] = 1.0
    history["damage_share"] = 1.0
    result = priorities(history, _bench("UTILITY"), "UTILITY", window=10, k=8)
    assert {"cs_per_min", "cs_at_10", "damage_share"}.isdisjoint(p["stat"] for p in result)


def test_priorities_skip_a_stat_without_any_reference_value():
    result = priorities(history_of([5.5] * 10), _bench().assign(cs_at_10=np.nan), "MIDDLE", window=10, k=8)
    assert "cs_at_10" not in {p["stat"] for p in result} and len(result) == 7


def test_priorities_are_empty_without_references():
    assert priorities(history_of([5.0] * 10), None, "MIDDLE") == []
    assert priorities(history_of([5.0] * 10), _bench("TOP"), "MIDDLE") == []
