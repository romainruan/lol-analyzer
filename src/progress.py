"""Suivi de vos parties dans le temps : ce qui évolue, et ce qu'il faut travailler en priorité.

    python -m src.progress --raw-dir data/raw/me --window 20

Compare vos `window` dernières parties aux `window` précédentes, au poste que vous jouez le plus, et situe votre
moyenne récente par rapport aux repères (`python -m src.benchmarks`). Une moyenne varie moins qu'une partie : son
percentile sous-estime l'écart réel avec les autres joueurs (voir Limites dans le README).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .benchmarks import (
    DEFAULT_BENCH,
    MIN_REFERENCE_ROWS,
    ROLE_NAMES,
    STAT_COLS,
    build_benchmark,
    load_benchmarks,
    reference_for,
)
from .debrief import ADVICE, ADVICE_BY_ROLE, IGNORED_STATS, LABELS, STAT_FORMAT, _format, _fr, _percentile, _rating

MIN_GAMES = 5  # en dessous, comparer deux moyennes n'a pas de sens
# 95 % pour les 8 statistiques prises ensemble (Bonferroni), sinon il y aurait trop souvent une fausse alerte
Z_ALL_STATS = 2.73


def player_history(raw_dir: Path, puuid: str | None) -> pd.DataFrame:
    """Une ligne par partie du joueur, de la plus ancienne à la plus récente."""
    rows = build_benchmark(raw_dir)
    if rows.empty:
        return rows
    return rows[rows["puuid"] == puuid].sort_values("game_creation").reset_index(drop=True)


def main_role(history: pd.DataFrame, window: int = 20) -> str | None:
    """Le poste le plus joué récemment."""
    if history.empty:
        return None
    return history.tail(2 * window)["role"].mode().iloc[0]


def trend_table(history: pd.DataFrame, window: int = 20, role: str | None = None) -> pd.DataFrame:
    """Vos `window` dernières parties contre les `window` précédentes, statistique par statistique."""
    if history.empty:
        return pd.DataFrame()
    rows = history if role is None else history[history["role"] == role]
    recent, previous = rows.tail(window), rows.iloc[-2 * window:-window]
    out = []
    for col in STAT_COLS:
        new, old = recent[col].dropna().astype(float), previous[col].dropna().astype(float)
        row = {"stat": col, "recent": new.mean() if len(new) else float("nan"),
               "precedent": old.mean() if len(old) else float("nan"), "evolution": float("nan")}
        if len(new) < MIN_GAMES or len(old) < MIN_GAMES:
            row["verdict"] = "pas assez de parties"
        else:
            row["evolution"] = new.mean() - old.mean()
            margin = Z_ALL_STATS * (new.var() / len(new) + old.var() / len(old)) ** 0.5
            row["verdict"] = ("en progrès" if row["evolution"] > margin
                              else "en baisse" if row["evolution"] < -margin else "pas de conclusion")
        out.append(row)
    return pd.DataFrame(out)


def champion_table(history: pd.DataFrame) -> pd.DataFrame:
    """Parties, victoires et KDA moyen par champion, du plus joué au moins joué."""
    if history.empty:
        return pd.DataFrame()
    table = history.groupby("champion", sort=False).agg(
        parties=("win", "size"), victoires_pct=("win", "mean"), kda=("kda", "mean")).reset_index()
    table["victoires_pct"] = (100 * table["victoires_pct"]).round().astype(int)
    table["kda"] = table["kda"].round(1)
    return table.sort_values("parties", ascending=False, kind="stable").reset_index(drop=True)


def priorities(history: pd.DataFrame, bench: pd.DataFrame | None, role: str | None, window: int = 20,
               k: int = 2) -> list[dict]:
    """Les `k` statistiques les plus basses face aux joueurs du même poste, sur vos parties récentes."""
    if bench is None or role is None or history.empty:
        return []
    reference, label = reference_for(bench, role)
    recent = history[history["role"] == role].tail(window)
    if len(reference) < MIN_REFERENCE_ROWS or recent.empty:
        return []
    advice = {**ADVICE, **ADVICE_BY_ROLE.get(role, {})}
    ignored = IGNORED_STATS.get(role, set())
    games = "votre dernière partie" if len(recent) == 1 else f"vos {len(recent)} dernières parties"
    out = []
    for col in STAT_COLS:
        name = STAT_FORMAT[col][0]
        values = recent[col].dropna().astype(float)
        if name in ignored or values.empty or not reference[col].notna().any():
            continue
        mean = values.mean()
        percentile = _percentile(mean, reference[col])
        title, unit = LABELS[name]
        out.append({
            "stat": col, "nom": title, "moyenne": mean, "percentile": percentile, "reference": label,
            "valeur": f"{_fr(_format(col, mean))}{unit}", "mediane": f"{_fr(_format(col, reference[col].median()))}{unit}",
            "phrase": (f"{title} : {_fr(_format(col, mean))}{unit} en moyenne sur {games}, mieux que {percentile} % "
                       f"des {label} ({_rating(percentile)}). Conseil : {advice[name]}"),
        })
    return sorted(out, key=lambda row: row["percentile"])[:k]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--raw-dir", default="data/raw/me")
    p.add_argument("--bench", default=str(DEFAULT_BENCH))
    p.add_argument("--window", type=int, default=20, help="nombre de parties récentes à comparer")
    args = p.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    puuid_file = raw_dir / "puuid.txt"
    if not puuid_file.exists():
        raise SystemExit(f"{puuid_file} introuvable : lancez d'abord `python -m src.collect --riot-id ...`.")
    history = player_history(raw_dir, puuid_file.read_text(encoding="utf-8").strip())
    if history.empty:
        raise SystemExit(f"Aucune partie de plus de 15 minutes dans {raw_dir}.")
    role = main_role(history, args.window)
    print(f"{len(history)} parties, poste principal : {ROLE_NAMES.get(role, role)}\n")
    print(f"Vos {args.window} dernières parties contre les {args.window} précédentes :")
    print(trend_table(history, args.window, role).round(2).to_string(index=False))
    print("\nVos champions :")
    print(champion_table(history).head(10).to_string(index=False))
    found = priorities(history, load_benchmarks(args.bench), role, args.window)
    if found:
        print("\nÀ travailler en priorité :")
        for row in found:
            print(f"- {row['phrase']}")
    else:
        print(f"\n(Pas de repères pour ce poste dans {args.bench} : lancez `python -m src.benchmarks`.)")


if __name__ == "__main__":
    main()
