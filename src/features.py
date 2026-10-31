"""Construction des features à partir d'une partie (match-v5) et de sa timeline.

Chaque ligne = une partie observée à une minute donnée. La cible `blue_win`
vaut 1 si l'équipe bleue gagne. Seules les informations disponibles AVANT la
minute observée sont utilisées (pas de fuite de données).
"""
from __future__ import annotations

BLUE, RED = 100, 200
DEFAULT_MINUTES = (5, 10, 15, 20, 25)
FRAME_TOLERANCE_MS = 5000

# teamPosition (match-v5) -> préfixe de colonne
ROLES = {"TOP": "top", "JUNGLE": "jng", "MIDDLE": "mid", "BOTTOM": "bot", "UTILITY": "sup"}
LANE_COLS = [f"{prefix}_gold_diff" for prefix in ROLES.values()] + [
    "mid_xp_diff",
    "mid_cs_diff",
    "mid_level_diff",
]


def _role_map(info: dict) -> dict[str, tuple[int, int]]:
    """Pour chaque rôle, (participantId bleu, participantId rouge).

    Un rôle est ignoré si la détection de poste est ambiguë ou manquante
    (déconnexions, remakes...), pour ne pas fabriquer de fausses comparaisons.
    """
    found: dict[tuple[str, int], list[int]] = {}
    for p in info.get("participants", []):
        role = p.get("teamPosition")
        if role in ROLES:
            found.setdefault((role, p["teamId"]), []).append(p["participantId"])
    out = {}
    for role in ROLES:
        blue, red = found.get((role, BLUE), []), found.get((role, RED), [])
        if len(blue) == 1 and len(red) == 1:
            out[role] = (blue[0], red[0])
    return out


def team_of(participant_id: int) -> int:
    """Dans match-v5, les participants 1-5 sont l'équipe bleue, 6-10 la rouge."""
    return BLUE if participant_id <= 5 else RED


def _sign(team_id: int) -> int:
    return 1 if team_id == BLUE else -1


def _patch(game_version: str) -> str:
    return ".".join(game_version.split(".")[:2])


def build_rows(match: dict, timeline: dict, minutes=DEFAULT_MINUTES) -> list[dict]:
    info = match["info"]
    if info.get("gameEndedInEarlySurrender"):  # remake : la partie n'a pas vraiment été jouée
        return []
    blue_win = int(next(t for t in info["teams"] if t["teamId"] == BLUE)["win"])
    frames = timeline["info"]["frames"]
    events = [e for f in frames for e in f.get("events", [])]
    match_id = match["metadata"]["matchId"]
    roles = _role_map(info)

    rows: list[dict] = []
    for m in minutes:
        if m >= len(frames):
            continue
        cutoff = frames[m]["timestamp"]
        if cutoff < m * 60_000 - FRAME_TOLERANCE_MS:  # dernière frame partielle
            continue

        # --- état des deux équipes à la minute m (différences bleu - rouge)
        diff = {"gold": 0, "xp": 0, "level": 0, "cs": 0}
        for pid, pf in frames[m]["participantFrames"].items():
            s = _sign(team_of(int(pid)))
            diff["gold"] += s * pf["totalGold"]
            diff["xp"] += s * pf["xp"]
            diff["level"] += s * pf["level"]
            diff["cs"] += s * (pf["minionsKilled"] + pf["jungleMinionsKilled"])

        # --- écarts lane contre lane (bleu - rouge), None si le rôle est ambigu
        lane: dict[str, int | None] = {c: None for c in LANE_COLS}
        for role, (blue_id, red_id) in roles.items():
            b = frames[m]["participantFrames"][str(blue_id)]
            r = frames[m]["participantFrames"][str(red_id)]
            prefix = ROLES[role]
            lane[f"{prefix}_gold_diff"] = b["totalGold"] - r["totalGold"]
            if role == "MIDDLE":
                lane["mid_xp_diff"] = b["xp"] - r["xp"]
                lane["mid_level_diff"] = b["level"] - r["level"]
                lane["mid_cs_diff"] = (b["minionsKilled"] + b["jungleMinionsKilled"]) - (
                    r["minionsKilled"] + r["jungleMinionsKilled"]
                )

        # --- événements jusqu'à la minute m
        kills = towers = dragons = heralds = grubs = barons = wards = 0
        for e in events:
            if e["timestamp"] > cutoff:
                continue
            t = e["type"]
            if t == "CHAMPION_KILL" and e.get("killerId", 0) > 0:
                kills += _sign(team_of(e["killerId"]))
            elif t == "BUILDING_KILL" and e.get("buildingType") == "TOWER_BUILDING":
                towers += _sign(RED if e["teamId"] == BLUE else BLUE)  # équipe qui détruit
            elif t == "ELITE_MONSTER_KILL":
                s = _sign(e.get("killerTeamId", 0))
                kind = e.get("monsterType")
                if kind == "DRAGON":
                    dragons += s
                elif kind == "RIFTHERALD":
                    heralds += s
                elif kind == "HORDE":
                    grubs += s
                elif kind == "BARON_NASHOR":
                    barons += s
            elif t == "WARD_PLACED" and e.get("creatorId", 0) > 0:
                wards += _sign(team_of(e["creatorId"]))

        rows.append(
            {
                "match_id": match_id,
                "patch": _patch(info.get("gameVersion", "0.0")),
                "game_creation": info.get("gameCreation", 0),  # date de la partie (ms), pour un test dans le temps
                "minute": m,
                "gold_diff": diff["gold"],
                "xp_diff": diff["xp"],
                "level_diff": diff["level"],
                "cs_diff": diff["cs"],
                "kills_diff": kills,
                "towers_diff": towers,
                "dragons_diff": dragons,
                "heralds_diff": heralds,
                "grubs_diff": grubs,
                "barons_diff": barons,
                "wards_diff": wards,
                **lane,
                "blue_win": blue_win,
            }
        )
    return rows
