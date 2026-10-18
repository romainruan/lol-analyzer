"""Téléchargement des parties (JSON bruts) depuis l'API Riot.

Exemples :
    # vos propres parties classées
    python -m src.collect --riot-id "VotrePseudo#EUW" --count 50

    # échantillon de joueurs Gold II (5 parties par joueur, 2 pages de la ligue)
    python -m src.collect --tier GOLD --division II --pages 2 --per-player 5

    # échantillon de joueurs Master (200 joueurs tirés au sort, 5 parties chacun)
    python -m src.collect --tier MASTER --max-players 200 --per-player 5

Chaque partie est sauvegardée immédiatement : on peut interrompre (Ctrl+C) et
relancer sans rien retélécharger.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from dotenv import load_dotenv

from .riot_client import NotFound, RiotAPIError, RiotClient

APEX_TIERS = {"MASTER", "GRANDMASTER", "CHALLENGER"}


def resolve_puuid(client: RiotClient, entry: dict) -> str | None:
    """Récupère le puuid d'une entrée de ligue (directement, sinon via summoner-v4)."""
    if entry.get("puuid"):
        return entry["puuid"]
    summoner_id = entry.get("summonerId")
    if summoner_id:
        try:
            return client.get_puuid_by_summoner_id(summoner_id)
        except (NotFound, RiotAPIError):
            return None
    return None


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def collect_match(client: RiotClient, match_id: str, raw_dir: Path) -> bool:
    """Télécharge la partie et sa timeline. Renvoie True si téléchargée."""
    match_path = raw_dir / "matches" / f"{match_id}.json"
    tl_path = raw_dir / "timelines" / f"{match_id}.json"
    if match_path.exists() and tl_path.exists():
        return False
    try:
        match = client.get_match(match_id)
        timeline = client.get_timeline(match_id)
    except NotFound:
        return False
    save_json(match_path, match)
    save_json(tl_path, timeline)
    return True


def collect_for_puuid(client: RiotClient, puuid: str, raw_dir: Path, count: int) -> int:
    new = 0
    for match_id in client.get_match_ids(puuid, count=count):
        if collect_match(client, match_id, raw_dir):
            new += 1
            print(f"  + {match_id}")
    return new


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--riot-id", help='Votre Riot ID, ex. "Pseudo#EUW"')
    p.add_argument("--count", type=int, default=20, help="Nb de parties (mode --riot-id)")
    p.add_argument("--tier", help="Tier à échantillonner (IRON..DIAMOND, MASTER, GRANDMASTER, CHALLENGER)")
    p.add_argument("--division", default="II", help="Division (I..IV), ignorée pour Master+")
    p.add_argument("--pages", type=int, default=1, help="Pages de la ligue (IRON..DIAMOND)")
    p.add_argument("--max-players", type=int, default=200, help="Joueurs tirés au sort (Master+)")
    p.add_argument("--per-player", type=int, default=5, help="Parties par joueur échantillonné")
    p.add_argument("--out", help="Dossier de sortie (défaut : data/raw/me ou data/raw/sample)")
    args = p.parse_args()

    if not args.riot_id and not args.tier:
        p.error("Donnez --riot-id ou --tier.")

    client = RiotClient()

    if args.riot_id:
        raw_dir = Path(args.out or "data/raw/me")
        name, _, tag = args.riot_id.partition("#")
        puuid = client.get_puuid(name, tag)
        raw_dir.mkdir(parents=True, exist_ok=True)
        # utilisé plus tard pour retrouver vos parties (data/ est ignoré par git)
        (raw_dir / "puuid.txt").write_text(puuid, encoding="utf-8")
        total = collect_for_puuid(client, puuid, raw_dir, args.count)
        print(f"{total} nouvelles parties dans {raw_dir}")
        return

    raw_dir = Path(args.out or "data/raw/sample")
    if args.tier.upper() in APEX_TIERS:
        # Master+ : une seule liste (souvent des milliers de joueurs), on en tire un échantillon
        entries = client.get_apex_entries(args.tier)
        random.Random(42).shuffle(entries)
        entries = entries[: args.max_players]
    else:
        entries = []
        for page in range(1, args.pages + 1):
            entries.extend(client.get_league_entries(args.tier, args.division, page))

    total = 0
    for i, entry in enumerate(entries, 1):
        puuid = resolve_puuid(client, entry)
        if not puuid:
            print(f"[{i}/{len(entries)}] entrée sans identifiant exploitable, ignorée")
            continue
        print(f"[{i}/{len(entries)}] joueur suivant")
        total += collect_for_puuid(client, puuid, raw_dir, args.per_player)
    print(f"{total} nouvelles parties dans {raw_dir}")


if __name__ == "__main__":
    main()
