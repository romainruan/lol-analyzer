from src.features import build_rows


def _frame(minute: int, blue_gold: int, red_gold: int, events=None) -> dict:
    pf = {}
    for pid in range(1, 11):
        gold = blue_gold if pid <= 5 else red_gold
        pf[str(pid)] = {
            "totalGold": gold,
            "xp": 1000,
            "level": 6,
            "minionsKilled": 50,
            "jungleMinionsKilled": 0,
        }
    return {"timestamp": minute * 60_000, "participantFrames": pf, "events": events or []}


def _match(blue_wins: bool) -> dict:
    return {
        "metadata": {"matchId": "EUW1_1"},
        "info": {
            "gameVersion": "15.20.123.456",
            "teams": [
                {"teamId": 100, "win": blue_wins},
                {"teamId": 200, "win": not blue_wins},
            ],
        },
    }


def test_basic_diffs_and_events():
    events_min3 = [
        {"type": "CHAMPION_KILL", "timestamp": 150_000, "killerId": 2},  # kill bleu
        {"type": "CHAMPION_KILL", "timestamp": 160_000, "killerId": 7},  # kill rouge
        {"type": "CHAMPION_KILL", "timestamp": 170_000, "killerId": 8},  # kill rouge
        {"type": "ELITE_MONSTER_KILL", "timestamp": 170_000, "killerTeamId": 100,
         "monsterType": "DRAGON"},
        {"type": "BUILDING_KILL", "timestamp": 175_000, "teamId": 200,
         "buildingType": "TOWER_BUILDING"},  # tour rouge détruite => point bleu
        {"type": "WARD_PLACED", "timestamp": 100_000, "creatorId": 1},
    ]
    timeline = {
        "info": {
            "frames": [
                _frame(0, 500, 500),
                _frame(1, 600, 600),
                _frame(2, 700, 700),
                _frame(3, 900, 800, events_min3),
                _frame(4, 1000, 900),
                _frame(5, 1200, 1000),
            ]
        }
    }
    rows = build_rows(_match(True), timeline, minutes=(5,))
    assert len(rows) == 1
    r = rows[0]
    assert r["gold_diff"] == 5 * 1200 - 5 * 1000
    assert r["xp_diff"] == 0
    assert r["kills_diff"] == 1 - 2
    assert r["dragons_diff"] == 1
    assert r["towers_diff"] == 1
    assert r["wards_diff"] == 1
    assert r["patch"] == "15.20"
    assert r["blue_win"] == 1


def test_no_leak_events_after_cutoff():
    late_kill = {"type": "CHAMPION_KILL", "timestamp": 400_000, "killerId": 1}
    timeline = {
        "info": {"frames": [_frame(i, 500, 500) for i in range(5)]
                 + [_frame(5, 500, 500, [late_kill])]}
    }
    rows = build_rows(_match(False), timeline, minutes=(4, 5))
    by_minute = {r["minute"]: r for r in rows}
    assert by_minute[4]["kills_diff"] == 0  # le kill de 6:40 n'existe pas encore à 4 min
    assert by_minute[5]["blue_win"] == 0


def test_skips_minutes_beyond_game_length():
    timeline = {"info": {"frames": [_frame(i, 500, 500) for i in range(8)]}}
    rows = build_rows(_match(True), timeline, minutes=(5, 10, 15))
    assert [r["minute"] for r in rows] == [5]


def test_skips_partial_last_frame():
    frames = [_frame(i, 500, 500) for i in range(10)]
    frames.append({**_frame(10, 500, 500), "timestamp": 10 * 60_000 - 30_000})  # fin de partie
    rows = build_rows(_match(True), {"info": {"frames": frames}}, minutes=(10,))
    assert rows == []


def test_monster_without_a_playing_team_counts_for_nobody():
    events = [
        {"type": "ELITE_MONSTER_KILL", "timestamp": 100_000, "killerTeamId": 300, "monsterType": "DRAGON"},
        {"type": "ELITE_MONSTER_KILL", "timestamp": 110_000, "monsterType": "DRAGON"},
        {"type": "ELITE_MONSTER_KILL", "timestamp": 120_000, "killerTeamId": 200, "monsterType": "DRAGON"},
    ]
    timeline = {"info": {"frames": [_frame(0, 500, 500), _frame(1, 500, 500), _frame(2, 500, 500, events),
                                    _frame(3, 500, 500)]}}
    assert build_rows(_match(True), timeline, minutes=(3,))[0]["dragons_diff"] == -1


def test_skips_remakes_only():
    timeline = {"info": {"frames": [_frame(i, 500, 500) for i in range(6)]}}
    match = _match(True)
    match["info"]["gameEndedInEarlySurrender"] = True
    assert build_rows(match, timeline, minutes=(2, 3)) == []
    match["info"]["gameEndedInEarlySurrender"] = False
    assert [r["minute"] for r in build_rows(match, timeline, minutes=(2, 3))] == [2, 3]
