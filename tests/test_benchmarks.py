import pandas as pd

from src.benchmarks import STAT_COLS, compare, mid_rows


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
    rows = mid_rows(_match(), _timeline())
    assert len(rows) == 2
    blue = next(r for r in rows if r["puuid"] == "p3")
    assert blue["cs_per_min"] == 220 / 30
    assert blue["kda"] == 12 / 2
    assert blue["vision_per_min"] == 30 / 30
    assert blue["kill_participation"] == 12 / (6 + 2 * 4)  # 4 autres joueurs à 2 kills
    assert blue["cs_at_10"] == 84
    assert blue["win"] == 1


def test_short_game_is_ignored():
    assert mid_rows(_match(duration_s=600), _timeline()) == []


def test_compare_percentile():
    bench = pd.DataFrame({c: [1.0, 2.0, 3.0, 4.0] for c in STAT_COLS})
    me = pd.DataFrame({c: [3.5] for c in STAT_COLS})
    result = compare(me, bench).set_index("stat")
    assert result.loc["cs_per_min", "percentile"] == 75
    assert result.loc["cs_per_min", "mediane_ref"] == 2.5
