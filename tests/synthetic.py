"""Parties synthétiques au format match-v5, pour tester toute la chaîne sans clé Riot ni réseau."""
from __future__ import annotations

import numpy as np

POSITIONS = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
CHAMPIONS = {
    "TOP": ("Garen", "Darius", "Ornn"),
    "JUNGLE": ("LeeSin", "Elise", "Hecarim"),
    "MIDDLE": ("Ahri", "Syndra", "Zed"),
    "BOTTOM": ("Jinx", "Caitlyn", "Draven"),
    "UTILITY": ("Nami", "Leona", "Thresh"),
}
FIRST_GAME_MS = 1_790_000_000_000
HOUR_MS = 3_600_000


def make_game(index: int, me: str | None = None, role: str = "MIDDLE", champion: str | None = None,
              skill: float = 0.0) -> tuple[dict, dict]:
    """Une partie reproductible. `me` place ce puuid au poste `role` ; `skill` améliore ses statistiques."""
    rng = np.random.default_rng(1000 + index)
    duration = int(rng.integers(20, 36))
    strength = float(rng.normal(0, 1))
    blue_wins = bool(strength + rng.logistic(0, 0.5) > 0)
    match_id = f"EUW1_{8_000_000 + index}"
    my_pid = None
    if me is not None:
        my_pid = POSITIONS.index(role) + (1 if index % 2 == 0 else 6)  # bleu une partie sur deux

    stats = {pid: {"kills": 0, "deaths": 0, "assists": 0} for pid in range(1, 11)}
    frames = []
    for m in range(duration + 2):
        last = m == duration + 1  # dernière frame : la fin de partie, moins d'une minute après
        minute = duration + 25 / 60 if last else m
        frame_pf = {}
        for pid in range(1, 11):
            side = 1 if pid <= 5 else -1
            position = POSITIONS[(pid - 1) % 5]
            boost = skill if pid == my_pid else 0.0
            farm = {"JUNGLE": 0.0, "UTILITY": 1.0}.get(position, 7.0) + boost
            frame_pf[str(pid)] = {
                "totalGold": int(500 + minute * (380 + side * strength * 18 + boost * 20) + rng.normal(0, 30)),
                "xp": int(minute * (420 + side * strength * 15) + rng.normal(0, 30)),
                "level": int(min(18, 1 + minute // 2)),
                "minionsKilled": int(minute * farm),
                "jungleMinionsKilled": int(minute * 5) if position == "JUNGLE" else 0,
            }
        frames.append({"timestamp": int(minute * 60_000), "participantFrames": frame_pf,
                       "events": _events(rng, m, strength, stats) if 0 < m <= duration else []})

    win_blue = {100: blue_wins, 200: not blue_wins}
    participants = []
    for pid in range(1, 11):
        team = 100 if pid <= 5 else 200
        position = POSITIONS[(pid - 1) % 5]
        boost = skill if pid == my_pid else 0.0
        kda = stats[pid]
        participants.append({
            "participantId": pid, "teamId": team, "teamPosition": position,
            "puuid": me if pid == my_pid else f"{match_id}-{pid}",
            "championName": champion if pid == my_pid and champion else CHAMPIONS[position][index % 3],
            "win": win_blue[team], **kda,
            "totalMinionsKilled": frames[-1]["participantFrames"][str(pid)]["minionsKilled"],
            "neutralMinionsKilled": frames[-1]["participantFrames"][str(pid)]["jungleMinionsKilled"],
            "visionScore": int(rng.integers(10, 40) + boost),
            "totalDamageDealtToChampions": int(rng.integers(5_000, 30_000)),
        })
    match = {
        "metadata": {"matchId": match_id},
        "info": {
            "gameCreation": FIRST_GAME_MS + index * HOUR_MS, "gameDuration": duration * 60 + 25,
            "gameVersion": "16.20.700.1", "queueId": 420, "participants": participants,
            "gameEndedInEarlySurrender": False,
            "teams": [{"teamId": 100, "win": blue_wins}, {"teamId": 200, "win": not blue_wins}],
        },
    }
    return match, {"metadata": {"matchId": match_id}, "info": {"frames": frames}}


def _events(rng: np.random.Generator, minute: int, strength: float, stats: dict) -> list[dict]:
    t = (minute - 1) * 60_000
    blue_edge = 0.5 + 0.25 * np.tanh(strength)
    events = [{"type": "WARD_PLACED", "timestamp": t + 1_000, "creatorId": int(rng.integers(1, 11))}]

    def team() -> int:
        return 100 if rng.random() < blue_edge else 200

    for _ in range(rng.poisson(0.8)):
        killer_team = team()
        offset = 0 if killer_team == 100 else 5
        killer, assist = rng.choice(5, 2, replace=False) + 1 + offset
        victim = int(rng.integers(6, 11)) if killer_team == 100 else int(rng.integers(1, 6))
        stats[int(killer)]["kills"] += 1
        stats[int(assist)]["assists"] += 1
        stats[victim]["deaths"] += 1
        events.append({"type": "CHAMPION_KILL", "timestamp": t + 20_000, "killerId": int(killer),
                       "victimId": victim})
    if minute % 6 == 0:
        events.append({"type": "ELITE_MONSTER_KILL", "timestamp": t + 30_000, "killerTeamId": team(),
                       "monsterType": "DRAGON", "monsterSubType": "FIRE_DRAGON"})
    if minute == 6:
        events.append({"type": "ELITE_MONSTER_KILL", "timestamp": t + 31_000, "killerTeamId": team(),
                       "monsterType": "HORDE"})
    if minute == 15:
        events.append({"type": "ELITE_MONSTER_KILL", "timestamp": t + 32_000, "killerTeamId": team(),
                       "monsterType": "RIFTHERALD"})
    if minute == 22:
        events.append({"type": "ELITE_MONSTER_KILL", "timestamp": t + 33_000, "killerTeamId": team(),
                       "monsterType": "BARON_NASHOR"})
    if minute > 10 and rng.random() < 0.3:
        loser = 300 - team()
        events.append({"type": "BUILDING_KILL", "timestamp": t + 40_000, "teamId": loser,
                       "buildingType": "TOWER_BUILDING"})
    return events


class FakeRiotClient:
    """Même interface que `RiotClient`, sur des parties synthétiques : `n_sample` joueurs et vos `n_mine` parties."""

    def __init__(self, n_sample: int = 0, n_mine: int = 0, my_puuid: str = "puuid-secret-123", role: str = "MIDDLE"):
        self.my_puuid = my_puuid
        self.games: dict[str, tuple[dict, dict]] = {}
        self.sample_ids: dict[str, list[str]] = {}
        for i in range(n_sample):
            game = make_game(i)
            self.games[game[0]["metadata"]["matchId"]] = game
            self.sample_ids[f"joueur-{i}"] = [game[0]["metadata"]["matchId"]]
        self.my_ids = []
        for i in range(n_mine):
            game = make_game(100_000 + i, me=my_puuid, role=role, champion=("Ahri", "Syndra")[i % 2])
            self.games[game[0]["metadata"]["matchId"]] = game
            self.my_ids.insert(0, game[0]["metadata"]["matchId"])  # du plus récent au plus ancien, comme Riot

    def get_puuid(self, game_name: str, tag_line: str) -> str:
        return self.my_puuid

    def get_apex_entries(self, tier: str) -> list[dict]:
        return [{"puuid": puuid} for puuid in self.sample_ids]

    def get_match_ids(self, puuid: str, start: int = 0, count: int = 20, **_) -> list[str]:
        ids = self.my_ids if puuid == self.my_puuid else self.sample_ids[puuid]
        return ids[start:start + count]

    def get_match(self, match_id: str) -> dict:
        return self.games[match_id][0]

    def get_timeline(self, match_id: str) -> dict:
        return self.games[match_id][1]
