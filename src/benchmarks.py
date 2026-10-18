"""Repères de référence pour le rôle mid (une ligne = un mid laner dans une partie).

    python -m src.benchmarks --raw-dir data/raw/sample --me-dir data/raw/me

Si `data/raw/me/puuid.txt` existe (créé par `collect --riot-id`), vos parties
jouées en mid sont comparées à la distribution des mid laners de l'échantillon.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

MIN_DURATION_S = 900  # on ignore les remakes et parties très courtes
STAT_COLS = [
    "cs_per_min",
    "vision_per_min",
    "kda",
    "kill_participation",
    "damage_share",
    "cs_at_10",
    "gold_at_10",
    "xp_at_10",
]


def mid_rows(match: dict, timeline: dict | None = None) -> list[dict]:
    """Statistiques des mid laners d'une partie (vide si le rôle est ambigu)."""
    info = match["info"]
    duration_s = info.get("gameDuration", 0)  # en secondes depuis le patch 11.20
    if duration_s < MIN_DURATION_S:
        return []

    team_kills: dict[int, int] = defaultdict(int)
    team_damage: dict[int, int] = defaultdict(int)
    mids: dict[int, list[dict]] = defaultdict(list)
    for p in info["participants"]:
        team_kills[p["teamId"]] += p["kills"]
        team_damage[p["teamId"]] += p["totalDamageDealtToChampions"]
        if p.get("teamPosition") == "MIDDLE":
            mids[p["teamId"]].append(p)

    frames = timeline["info"]["frames"] if timeline else []
    minutes = duration_s / 60
    rows = []
    for team_id, players in mids.items():
        if len(players) != 1:
            continue
        p = players[0]
        cs = p["totalMinionsKilled"] + p.get("neutralMinionsKilled", 0)
        row = {
            "match_id": match["metadata"]["matchId"],
            "puuid": p.get("puuid"),
            "champion": p.get("championName"),
            "win": int(p["win"]),
            "cs_per_min": cs / minutes,
            "vision_per_min": p["visionScore"] / minutes,
            "kda": (p["kills"] + p["assists"]) / max(1, p["deaths"]),
            "kill_participation": (p["kills"] + p["assists"]) / max(1, team_kills[team_id]),
            "damage_share": p["totalDamageDealtToChampions"] / max(1, team_damage[team_id]),
            "cs_at_10": None,
            "gold_at_10": None,
            "xp_at_10": None,
        }
        if len(frames) > 10:
            pf = frames[10]["participantFrames"][str(p["participantId"])]
            row["cs_at_10"] = pf["minionsKilled"] + pf["jungleMinionsKilled"]
            row["gold_at_10"] = pf["totalGold"]
            row["xp_at_10"] = pf["xp"]
        rows.append(row)
    return rows


def build_benchmark(raw_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for match_path in sorted((raw_dir / "matches").glob("*.json")):
        match = json.loads(match_path.read_text(encoding="utf-8"))
        tl_path = raw_dir / "timelines" / match_path.name
        timeline = json.loads(tl_path.read_text(encoding="utf-8")) if tl_path.exists() else None
        rows.extend(mid_rows(match, timeline))
    return pd.DataFrame(rows)


def percentile_table(bench: pd.DataFrame) -> pd.DataFrame:
    return bench[STAT_COLS].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).T.rename(
        columns=lambda q: f"p{int(q * 100)}"
    )


def compare(me: pd.DataFrame, bench: pd.DataFrame) -> pd.DataFrame:
    """Votre moyenne sur chaque statistique et son rang dans la distribution de référence."""
    out = []
    for col in STAT_COLS:
        values = me[col].dropna()
        ref = bench[col].dropna()
        if values.empty or ref.empty:
            continue
        mean = values.mean()
        out.append(
            {
                "stat": col,
                "vous": round(mean, 3),
                "mediane_ref": round(ref.median(), 3),
                "percentile": round(100 * (ref < mean).mean()),
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default="data/raw/sample")
    p.add_argument("--me-dir", default="data/raw/me")
    p.add_argument("--out", default="data/mid_benchmarks.csv")
    args = p.parse_args()

    bench = build_benchmark(Path(args.raw_dir))
    if bench.empty:
        print("Aucun mid laner trouvé (parties trop courtes ou rôles non renseignés).")
        return
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bench.to_csv(out, index=False)
    print(f"{len(bench)} mid laners ({bench['match_id'].nunique()} parties) -> {out}\n")
    print("Repères mid (percentiles de la population de référence) :")
    print(percentile_table(bench).round(2).to_string(), "\n")

    puuid_file = Path(args.me_dir) / "puuid.txt"
    if puuid_file.exists():
        me_all = build_benchmark(Path(args.me_dir))
        me = me_all[me_all["puuid"] == puuid_file.read_text(encoding="utf-8").strip()]
        if me.empty:
            print("Aucune de vos parties jouée en mid dans data/raw/me.")
        else:
            print(f"Vos {len(me)} parties en mid comparées à la référence :")
            print(compare(me, bench).to_string(index=False))
    else:
        print("(Pas de data/raw/me/puuid.txt : relancez collect --riot-id pour comparer vos parties.)")


if __name__ == "__main__":
    main()
