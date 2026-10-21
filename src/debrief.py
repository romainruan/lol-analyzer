"""Débrief d'une partie rédigé par un LLM local (Ollama).

Le modèle ne lit jamais la timeline brute (trop longue, il inventerait). On lui
donne un petit résumé chiffré (« faits ») construit à partir de :
  - la courbe de probabilité et les moments clés (results/curves/*.json),
  - vos statistiques de la partie,
  - leur position parmi les mid laners de référence (data/mid_benchmarks.csv).

Ensuite on vérifie automatiquement que le texte ne contient aucun nombre absent
des faits : un modèle de 7 milliards de paramètres en invente facilement.

    python -m src.debrief --limit 3                 # les 3 dernières parties analysées
    python -m src.debrief --match-id EUW1_123456    # une partie précise
    python -m src.debrief --limit 1 --dry-run       # affiche les faits et le prompt, sans LLM
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
import requests

from .benchmarks import STAT_COLS, build_benchmark, compare, mid_rows
from .winprob import collapse_duplicates

DEFAULT_MODEL = "qwen2.5:7b"
MIN_BENCH_ROWS = 50
MIN_PROFILE_GAMES = 10

ROLES_FR = {"TOP": "top", "JUNGLE": "jungle", "MIDDLE": "mid", "BOTTOM": "bot (ADC)",
            "UTILITY": "support"}
# nom de la statistique dans benchmarks.py -> (nom dans les faits, facteur, décimales)
STAT_FORMAT = {
    "cs_per_min": ("cs_par_min", 1, 1),
    "vision_per_min": ("vision_par_min", 1, 2),
    "kda": ("kda_ratio", 1, 1),
    "kill_participation": ("participation_kills_pct", 100, 0),
    "damage_share": ("part_degats_pct", 100, 0),
    "cs_at_10": ("cs_a_10_min", 1, 0),
    "gold_at_10": ("or_a_10_min", 1, 0),
    "xp_at_10": ("xp_a_10_min", 1, 0),
}

SYSTEM_PROMPT = """Tu es un coach de League of Legends. Tu rédiges, uniquement en français, le \
débrief d'une partie à partir de DONNÉES au format JSON.

Règles strictes :
- N'utilise que les chiffres, champions et événements présents dans les DONNÉES. \
N'invente rien : ni chiffre, ni objet, ni événement.
- Un saut de probabilité n'est pas une cause prouvée : écris « probablement lié à » ou \
« coïncide avec », jamais « a causé ».
- Ne mentionne pas ce qui n'est pas dans les DONNÉES, même si tu le sais du jeu.
- Les percentiles comparent le joueur à des mid laners Master : un percentile de 20 \
signifie mieux que 20 % d'entre eux.
- « profil_habituel » décrit sa tendance sur plusieurs parties, pas cette partie seule.

Format : un paragraphe « Résumé » de 2 phrases, puis « Points forts » (2 puces), puis \
« Axes de progrès » (2 puces). Chaque puce s'appuie sur un chiffre fourni. 12 lignes maximum."""


class OllamaError(RuntimeError):
    """Ollama est injoignable ou le modèle est absent."""


# ----------------------------------------------------------------------------- faits
def _round(value: float, decimals: int) -> float | int:
    return int(round(value)) if decimals == 0 else round(float(value), decimals)


def _percentile(value: float, reference: pd.Series) -> int:
    ref = reference.dropna()
    return int(round(100 * (ref < value).mean())) if len(ref) else 0


def _format(col: str, value: float) -> float | int:
    _, factor, decimals = STAT_FORMAT[col]
    return _round(value * factor, decimals)


def _basic_stats(me: dict, minutes: float) -> dict:
    cs = me.get("totalMinionsKilled", 0) + me.get("neutralMinionsKilled", 0)
    return {
        "cs_par_min": _round(cs / minutes, 1),
        "vision_par_min": _round(me.get("visionScore", 0) / minutes, 2),
    }


def build_facts(
    analysis: dict,
    match: dict,
    timeline: dict | None,
    puuid: str,
    bench: pd.DataFrame | None = None,
    profile: tuple[int, pd.DataFrame] | None = None,
) -> dict | None:
    """Résumé chiffré d'une partie du point de vue du joueur. None si le joueur n'y figure pas."""
    info = match["info"]
    me = next((p for p in info["participants"] if p.get("puuid") == puuid), None)
    if me is None:
        return None
    minutes = max(info.get("gameDuration", 0), 1) / 60
    role = me.get("teamPosition")

    facts: dict = {
        "partie": {
            "champion": me.get("championName"),
            "role": ROLES_FR.get(role, "inconnu"),
            "resultat": analysis["perspective"]["result"],
            "duree_min": int(round(minutes)),
        },
        "stats": {"kda": f"{me.get('kills', 0)}/{me.get('deaths', 0)}/{me.get('assists', 0)}"},
    }

    row = None
    if role == "MIDDLE":
        row = next((r for r in mid_rows(match, timeline) if r["puuid"] == puuid), None)
    if row is not None:
        for col, (name, _, _) in STAT_FORMAT.items():
            if row.get(col) is not None and not pd.isna(row[col]):
                facts["stats"][name] = _format(col, row[col])
        if bench is not None and len(bench) >= MIN_BENCH_ROWS:
            facts["comparaison_mid_master"] = {
                STAT_FORMAT[col][0]: {
                    "valeur": _format(col, row[col]),
                    "mediane": _format(col, bench[col].median()),
                    "percentile": _percentile(row[col], bench[col]),
                }
                for col in STAT_COLS
                if row.get(col) is not None and not pd.isna(row[col])
            }
    else:
        facts["stats"].update(_basic_stats(me, minutes))

    if profile is not None:
        n_games, table = profile
        facts["profil_habituel"] = {
            "nb_parties_mid": n_games,
            "percentiles": {STAT_FORMAT[r["stat"]][0]: int(r["percentile"])
                            for _, r in table.iterrows() if r["stat"] in STAT_FORMAT},
        }

    probs = [c["win_prob"] for c in analysis["curve"]]
    if probs:
        facts["courbe"] = {
            "probabilite_depart_pct": round(probs[0] * 100),
            "plus_haut_pct": round(max(probs) * 100),
            "plus_bas_pct": round(min(probs) * 100),
        }
    facts["moments_cles"] = [
        {
            "minute": tp["minute"],
            "avant_pct": round(tp["win_prob_before"] * 100),
            "apres_pct": round(tp["win_prob_after"] * 100),
            "evenements": collapse_duplicates(tp["events"])[:4],
        }
        for tp in analysis["turning_points"]
    ]
    return facts


# ----------------------------------------------------------------------------- LLM
def build_messages(facts: dict) -> list[dict]:
    data = json.dumps(facts, ensure_ascii=False, indent=1)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"DONNÉES :\n```json\n{data}\n```\n\nRédige le débrief."},
    ]


def chat(messages: list[dict], model: str = DEFAULT_MODEL, host: str | None = None,
         timeout: int = 600) -> str:
    host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
    host = host.replace("0.0.0.0", "localhost")
    if not host.startswith("http"):
        host = "http://" + host
    session = requests.Session()
    session.trust_env = False  # Ollama est local : on ignore les proxys éventuels
    payload = {
        "model": model, "messages": messages, "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 4096, "seed": 42},
    }
    try:
        r = session.post(f"{host}/api/chat", json=payload, timeout=timeout)
    except requests.ConnectionError as e:
        raise OllamaError(
            f"Ollama ne répond pas sur {host}. Lancez `ollama serve` dans un autre terminal."
        ) from e
    if r.status_code == 404:
        raise OllamaError(f"Modèle introuvable : lancez `ollama pull {model}`.")
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


# ----------------------------------------------------------------------------- vérification
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def _numbers(text: str) -> list[float]:
    return [float(x.replace(",", ".")) for x in _NUMBER.findall(text)]


def check_numbers(text: str, facts: dict) -> list[float]:
    """Nombres du texte absents des faits (à un arrondi près). Liste vide = rien d'inventé."""
    allowed = _numbers(json.dumps(facts, ensure_ascii=False))
    body = re.sub(r"(?m)^\s*\d+[.)]\s+", "", text)  # numérotation éventuelle des listes
    unknown = []
    for x in _numbers(body):
        if not any(abs(x - a) < 0.051 or (x == int(x) and abs(x - a) <= 0.5) for a in allowed):
            unknown.append(x)
    return sorted(set(unknown))


def _fmt_number(x: float) -> str:
    return str(int(x)) if x == int(x) else str(x)


def render_markdown(match_id: str, facts: dict, text: str, unknown: list[float], model: str) -> str:
    if unknown:
        check = ("nombres absents des données, à vérifier à la main : "
                 + ", ".join(_fmt_number(x) for x in unknown))
    else:
        check = "aucun nombre inventé détecté (ce contrôle ne vérifie ni le sens ni les noms)"
    data = json.dumps(facts, ensure_ascii=False, indent=1)
    return (f"# Débrief {match_id}\n\n{text}\n\n---\n"
            f"*Modèle : {model}. Vérification automatique : {check}.*\n\n"
            f"Données fournies au modèle :\n\n```json\n{data}\n```\n")


# ----------------------------------------------------------------------------- CLI
def _select_ids(curves_dir: Path, match_id: str | None, limit: int) -> list[str]:
    if match_id:
        return [match_id]
    files = sorted(curves_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    return [f.stem for f in files[:limit]]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--raw-dir", default="data/raw/me", help="parties brutes + puuid.txt")
    p.add_argument("--curves", default="results/curves")
    p.add_argument("--bench", default="data/mid_benchmarks.csv")
    p.add_argument("--out", default="results/debriefs")
    p.add_argument("--match-id")
    p.add_argument("--limit", type=int, default=3)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dry-run", action="store_true", help="affiche les faits et le prompt sans appeler Ollama")
    args = p.parse_args()

    raw_dir, curves_dir = Path(args.raw_dir), Path(args.curves)
    puuid_file = raw_dir / "puuid.txt"
    if not puuid_file.exists():
        raise SystemExit(f"{puuid_file} introuvable : lancez d'abord `python -m src.collect --riot-id ...`.")
    puuid = puuid_file.read_text(encoding="utf-8").strip()

    bench = pd.read_csv(args.bench) if Path(args.bench).exists() else None
    if bench is None:
        print("(Pas de data/mid_benchmarks.csv : lancez `python -m src.benchmarks` pour comparer aux mid Master.)")
    profile = None
    if bench is not None:
        mine = build_benchmark(raw_dir)
        mine = mine[mine["puuid"] == puuid] if not mine.empty else mine
        if len(mine) >= MIN_PROFILE_GAMES:
            profile = (len(mine), compare(mine, bench))

    ids = _select_ids(curves_dir, args.match_id, args.limit)
    if not ids:
        raise SystemExit(f"Aucune analyse dans {curves_dir} : lancez `python -m src.winprob analyze`.")

    out_dir = Path(args.out)
    for match_id in ids:
        analysis_path = curves_dir / f"{match_id}.json"
        match_path = raw_dir / "matches" / f"{match_id}.json"
        if not analysis_path.exists() or not match_path.exists():
            print(f"{match_id} : analyse ou partie introuvable, ignorée.")
            continue
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        match = json.loads(match_path.read_text(encoding="utf-8"))
        tl_path = raw_dir / "timelines" / f"{match_id}.json"
        timeline = json.loads(tl_path.read_text(encoding="utf-8")) if tl_path.exists() else None

        facts = build_facts(analysis, match, timeline, puuid, bench, profile)
        if facts is None:
            print(f"{match_id} : vous n'apparaissez pas dans cette partie, ignorée.")
            continue

        messages = build_messages(facts)
        if args.dry_run:
            print(f"\n===== {match_id} =====\n{messages[1]['content']}")
            continue

        try:
            text = chat(messages, args.model)
        except OllamaError as e:
            raise SystemExit(str(e))
        unknown = check_numbers(text, facts)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{match_id}.md").write_text(
            render_markdown(match_id, facts, text, unknown, args.model), encoding="utf-8")
        print(f"\n===== {match_id} : {facts['partie']['champion']}, {facts['partie']['resultat']} =====")
        print(text)
        if unknown:
            print(f"\n[!] Nombres absents des données : {', '.join(_fmt_number(x) for x in unknown)}")
        else:
            print("\n[ok] Aucun nombre inventé détecté.")
    if not args.dry_run:
        print(f"\nDébriefs écrits dans {out_dir}/")


if __name__ == "__main__":
    main()
