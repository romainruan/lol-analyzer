import pandas as pd

from src.benchmarks import (
    STAT_COLS,
    compare,
    load_benchmarks,
    reference_for,
    role_rows,
)


def _player(pid, team, role, **kw):
    base = {
        "participantId": pid, "teamId": team, "teamPosition": role, "puuid": f"p{pid}",
        "championName": "Ahri", "win": team == 100, "kills": 2, "deaths": 1, "assists": 4,
        "totalMinionsKilled": 200, "neutralMinionsKilled": 20, "visionScore": 30,
        "totalDamageDealtToChampions": 10000,
    }
    base.update(kw)
    return base


def _match(duration_s=1800):
    players = [_player(i, 100 if i <= 5 else 200, "TOP") for i in range(1, 11)]
    players[2] = _player(3, 100, "MIDDLE", kills=6, deaths=2, assists=6)  # mid bleu
    players[7] = _player(8, 200, "MIDDLE")  # mid rouge
    return {
        "metadata": {"matchId": "EUW1_3"},
        "info": {"gameDuration": duration_s, "participants": players},
    }


def _timeline():
    pf = {
        str(i): {"totalGold": 3500, "xp": 4000, "minionsKilled": 80, "jungleMinionsKilled": 4}
        for i in range(1, 11)
    }
    return {"info": {"frames": [{"timestamp": m * 60_000, "participantFrames": pf} for m in range(12)]}}


def test_mid_rows_stats():
    rows = role_rows(_match(), _timeline())
    assert len(rows) == 2
    blue = next(r for r in rows if r["puuid"] == "p3")
    assert blue["cs_per_min"] == 220 / 30
    assert blue["kda"] == 12 / 2
    assert blue["vision_per_min"] == 30 / 30
    assert blue["kill_participation"] == 12 / (6 + 2 * 4)  # 4 autres joueurs à 2 kills
    assert blue["cs_at_10"] == 84
    assert blue["win"] == 1


def test_short_game_is_ignored():
    assert role_rows(_match(duration_s=600), _timeline()) == []


def test_compare_percentile():
    bench = pd.DataFrame({c: [1.0, 2.0, 3.0, 4.0] for c in STAT_COLS})
    me = pd.DataFrame({c: [3.5] for c in STAT_COLS})
    result = compare(me, bench).set_index("stat")
    assert result.loc["cs_per_min", "percentile"] == 75
    assert result.loc["cs_per_min", "mediane_ref"] == 2.5


# ----------------------------------------------------------------------------- tous les postes
def _full_match():
    positions = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
    players = [_player(i + 1, 100, r) for i, r in enumerate(positions)]
    players += [_player(i + 6, 200, r) for i, r in enumerate(positions)]
    return {"metadata": {"matchId": "EUW1_4"}, "info": {"gameDuration": 1800, "participants": players}}


def test_role_rows_gives_every_player_with_their_role():
    rows = role_rows(_full_match(), _timeline())
    assert len(rows) == 10
    assert sorted(r["role"] for r in rows if r["puuid"] in {"p1", "p2", "p3", "p4", "p5"}) == [
        "BOTTOM", "JUNGLE", "MIDDLE", "TOP", "UTILITY"]


def test_role_rows_skips_a_role_that_is_ambiguous_in_one_team():
    match = _full_match()
    match["info"]["participants"][0]["teamPosition"] = "JUNGLE"  # deux jungle côté bleu, plus de top
    rows = role_rows(match, _timeline())
    assert {"p1", "p2"}.isdisjoint(r["puuid"] for r in rows)
    assert "p7" in {r["puuid"] for r in rows}  # le jungle rouge n'est pas ambigu


def _bench(champions):
    return pd.DataFrame({"role": "MIDDLE", "champion": champions, "cs_per_min": 1.0})


def test_reference_is_the_champion_only_with_enough_games():
    bench = _bench(["Ahri"] * 30 + ["Zed"] * 29)
    rows, label = reference_for(bench, "MIDDLE", "Ahri")
    assert (len(rows), label) == (30, "Ahri mid Master")
    rows, label = reference_for(bench, "MIDDLE", "Zed")
    assert (len(rows), label) == (59, "mid Master")


def test_reference_keeps_the_name_of_the_sample_and_filters_the_role():
    bench = pd.concat([_bench(["Ahri"] * 3), pd.DataFrame({"role": ["BOTTOM"], "champion": ["Ashe"], "cs_per_min": [2.0]})])
    bench["reference"] = "Gold"
    rows, label = reference_for(bench, "BOTTOM")
    assert (len(rows), label) == (1, "ADC Gold")
    assert reference_for(bench, "TOP")[0].empty


def test_old_benchmark_file_counts_as_mid_only(tmp_path):
    path = tmp_path / "ancien.csv"
    pd.DataFrame({"champion": ["Ahri"], "cs_per_min": [7.0]}).to_csv(path, index=False)
    assert list(load_benchmarks(path)["role"]) == ["MIDDLE"]
    assert load_benchmarks(tmp_path / "absent.csv") is None


def test_default_benchmark_falls_back_on_the_old_file_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_benchmarks() is None
    (tmp_path / "data").mkdir()
    pd.DataFrame({"champion": ["Ahri"], "cs_per_min": [7.0]}).to_csv("data/mid_benchmarks.csv", index=False)
    assert len(load_benchmarks()) == 1
