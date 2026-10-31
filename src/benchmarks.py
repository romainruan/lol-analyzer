"""Repères de référence par poste (une ligne = un joueur dans une partie).

    python -m src.benchmarks --raw-dir data/raw/sample --me-dir data/raw/me --label Master

Si `data/raw/me/puuid.txt` existe (créé par `collect --riot-id`), vos parties sont comparées, poste par poste,
à la distribution des joueurs de l'échantillon. `--label` donne le nom de cet échantillon dans les phrases.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

MIN_DURATION_S = 900  # on ignore les remakes et parties très courtes
MIN_REFERENCE_ROWS = 30  # en dessous, un percentile est trop instable : pas de comparaison à ce champion ou ce poste
DEFAULT_BENCH = Path("data/benchmarks.csv")
LEGACY_BENCH = Path("data/mid_benchmarks.csv")  # anciens repères : mid uniquement, échantillon Master
DEFAULT_REFERENCE = "Master"
ROLE_NAMES = {"TOP": "top", "JUNGLE": "jungle", "MIDDLE": "mid", "BOTTOM": "ADC", "UTILITY": "support"}
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


def role_rows(match: dict, timeline: dict | None = None) -> list[dict]:
    """Statistiques de chaque joueur d'une partie (un poste ambigu dans une équipe est ignoré)."""
    info = match["info"]
    duration_s = info.get("gameDuration", 0)  # en secondes depuis le patch 11.20
    if duration_s < MIN_DURATION_S:
        return []

    team_kills: dict[int, int] = defaultdict(int)
    team_damage: dict[int, int] = defaultdict(int)
    by_post: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for p in info["participants"]:
        team_kills[p["teamId"]] += p["kills"]
        team_damage[p["teamId"]] += p["totalDamageDealtToChampions"]
        if p.get("teamPosition") in ROLE_NAMES:
            by_post[p["teamId"], p["teamPosition"]].append(p)

    frames = timeline["info"]["frames"] if timeline else []
    minutes = duration_s / 60
    rows = []
    for (team_id, role), players in by_post.items():
        if len(players) != 1:
            continue
        p = players[0]
        cs = p["totalMinionsKilled"] + p.get("neutralMinionsKilled", 0)
        row = {
            "match_id": match["metadata"]["matchId"],
            "game_creation": info.get("gameCreation", 0),
            "puuid": p.get("puuid"),
            "role": role,
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
        rows.extend(role_rows(match, timeline))
    return pd.DataFrame(rows)


def load_benchmarks(path: str | Path = DEFAULT_BENCH) -> pd.DataFrame | None:
    """Les repères enregistrés, ou None. Un ancien fichier (mid uniquement) reste utilisable."""
    path = Path(path)
    if not path.exists() and path == DEFAULT_BENCH and LEGACY_BENCH.exists():
        path = LEGACY_BENCH
    if not path.exists():
        return None
    bench = pd.read_csv(path)
    if "role" not in bench.columns:
        bench["role"] = "MIDDLE"
    return bench


def reference_for(bench: pd.DataFrame, role: str, champion: str | None = None) -> tuple[pd.DataFrame, str]:
    """Les lignes de comparaison et leur nom dans les phrases : le champion s'il a assez de parties, sinon le
    poste. Un CS ou une part de dégâts dépendent beaucoup du champion : autant comparer ce qui est comparable."""
    rows = bench[bench["role"] == role]
    tier = rows["reference"].iloc[0] if "reference" in rows.columns and len(rows) else DEFAULT_REFERENCE
    label = f"{ROLE_NAMES.get(role, role)} {tier}"
    if champion and "champion" in rows.columns:
        same = rows[rows["champion"] == champion]
        if len(same) >= MIN_REFERENCE_ROWS:
            return same, f"{champion} {label}"
    return rows, label


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


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--raw-dir", default="data/raw/sample")
    p.add_argument("--me-dir", default="data/raw/me")
    p.add_argument("--out", default=str(DEFAULT_BENCH))
    p.add_argument("--label", default=DEFAULT_REFERENCE, help="nom de l'échantillon dans les phrases, ex. Gold")
    args = p.parse_args(argv)

    bench = build_benchmark(Path(args.raw_dir))
    if bench.empty:
        print("Aucun joueur trouvé (parties trop courtes ou rôles non renseignés).")
        return
    bench["reference"] = args.label
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bench.to_csv(out, index=False)
    print(f"{len(bench)} lignes ({bench['match_id'].nunique()} parties) -> {out}\n")
    for role, rows in bench.groupby("role"):
        print(f"Repères {ROLE_NAMES.get(role, role)} (percentiles de la population de référence) :")
        print(percentile_table(rows).round(2).to_string(), "\n")

    puuid_file = Path(args.me_dir) / "puuid.txt"
    if puuid_file.exists():
        me_all = build_benchmark(Path(args.me_dir))
        me = me_all[me_all["puuid"] == puuid_file.read_text(encoding="utf-8").strip()] if not me_all.empty else me_all
        if me.empty:
            print("Aucune de vos parties dans data/raw/me.")
        for role, rows in me.groupby("role") if not me.empty else ():
            reference, label = reference_for(bench, role)
            print(f"Vos {len(rows)} parties au poste {ROLE_NAMES.get(role, role)} comparées aux {label} :")
            print(compare(rows, reference).to_string(index=False), "\n")
    else:
        print("(Pas de data/raw/me/puuid.txt : relancez collect --riot-id pour comparer vos parties.)")


if __name__ == "__main__":
    main()
