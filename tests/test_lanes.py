from src.features import build_rows

ROLE_ORDER = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


def _participants(duplicate_blue_mid: bool = False) -> list[dict]:
    ps = []
    for i, role in enumerate(ROLE_ORDER):
        ps.append({"participantId": i + 1, "teamId": 100, "teamPosition": role})
        ps.append({"participantId": i + 6, "teamId": 200, "teamPosition": role})
    if duplicate_blue_mid:  # détection de poste ambiguë : deux mid côté bleu
        ps[0]["teamPosition"] = "MIDDLE"
    return ps


def _frame(minute: int, overrides: dict | None = None) -> dict:
    pf = {
        str(pid): {"totalGold": 500, "xp": 1000, "level": 6, "minionsKilled": 50,
                   "jungleMinionsKilled": 0}
        for pid in range(1, 11)
    }
    for pid, values in (overrides or {}).items():
        pf[str(pid)].update(values)
    return {"timestamp": minute * 60_000, "participantFrames": pf, "events": []}


def _match(participants: list[dict]) -> dict:
    return {
        "metadata": {"matchId": "EUW1_2"},
        "info": {
            "gameVersion": "15.20.1",
            "teams": [{"teamId": 100, "win": True}, {"teamId": 200, "win": False}],
            "participants": participants,
        },
    }


def test_mid_lane_diffs():
    # participant 3 = mid bleu, participant 8 = mid rouge
    overrides = {
        3: {"totalGold": 900, "xp": 1500, "level": 8, "minionsKilled": 70},
        8: {"totalGold": 600, "xp": 1300, "level": 7, "minionsKilled": 63},
    }
    frames = [_frame(i) for i in range(5)] + [_frame(5, overrides)]
    rows = build_rows(_match(_participants()), {"info": {"frames": frames}}, minutes=(5,))
    r = rows[0]
    assert r["mid_gold_diff"] == 300
    assert r["mid_xp_diff"] == 200
    assert r["mid_level_diff"] == 1
    assert r["mid_cs_diff"] == 7
    assert r["top_gold_diff"] == 0


def test_ambiguous_role_gives_missing_values():
    frames = [_frame(i) for i in range(6)]
    rows = build_rows(
        _match(_participants(duplicate_blue_mid=True)), {"info": {"frames": frames}}, minutes=(5,)
    )
    assert rows[0]["mid_gold_diff"] is None
    assert rows[0]["jng_gold_diff"] == 0  # les autres rôles restent calculés


def test_missing_participants_do_not_break_rows():
    match = _match([])
    frames = [_frame(i) for i in range(6)]
    rows = build_rows(match, {"info": {"frames": frames}}, minutes=(5,))
    assert rows[0]["mid_gold_diff"] is None
