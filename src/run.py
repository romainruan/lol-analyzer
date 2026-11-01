"""Toute la chaîne en deux commandes, avec une clé Riot dans .env.

    # une fois : échantillon de référence, modèle de probabilité et repères (500 joueurs : environ 2 h)
    python -m src.run prepare --tier MASTER --max-players 500 --per-player 5

    # à chaque fois que vous voulez faire le point : télécharge, analyse, écrit le rapport HTML
    python -m src.run report --riot-id "Pseudo#EUW" --count 20

Le débrief est écrit par le code. Avec `--llm`, un modèle local (Ollama) rédige le résumé de chaque partie.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import benchmarks, build_dataset, collect, debrief, report, winprob
from .benchmarks import DEFAULT_BENCH

SAMPLE_DIR = "data/raw/sample"
ME_DIR = "data/raw/me"
DATASET = "data/dataset_all.csv"
MODEL = "results/curve_model.joblib"
CURVES = "results/curves"
DEBRIEFS = "results/debriefs"


def _riot_options(args: argparse.Namespace) -> list[str]:
    options = []
    if args.platform:
        options += ["--platform", args.platform]
    if args.min_interval is not None:
        options += ["--min-interval", str(args.min_interval)]
    return options


def prepare(args: argparse.Namespace) -> None:
    if not args.no_collect:
        collect.main(["--tier", args.tier, "--division", args.division, "--pages", str(args.pages),
                      "--max-players", str(args.max_players), "--per-player", str(args.per_player),
                      "--out", SAMPLE_DIR, *_riot_options(args)])
    build_dataset.main(["--raw-dir", SAMPLE_DIR, "--all-minutes", "--out", DATASET])
    winprob.main(["train", "--data", DATASET, "--model", MODEL, "--kind", args.kind, "--features", args.features,
                  "--calibration"])
    benchmarks.main(["--raw-dir", SAMPLE_DIR, "--me-dir", ME_DIR, "--out", args.bench,
                     "--label", args.label or args.tier.capitalize()])


def _write_debriefs(args: argparse.Namespace) -> None:
    common = ["--raw-dir", ME_DIR, "--curves", CURVES, "--bench", args.bench, "--out", DEBRIEFS,
              "--limit", str(args.count), "--only-missing"]
    if args.llm:
        try:
            debrief.main([*common, "--mode", "hybrid", "--model", args.model])
            return
        except SystemExit as stop:  # Ollama injoignable ou modèle absent
            print(f"{stop.code}\nLes débriefs restants sont écrits par le code, sans LLM.")
    debrief.main([*common, "--mode", "template"])


def make_report(args: argparse.Namespace) -> None:
    if not args.no_collect:
        if not args.riot_id:
            raise SystemExit('Donnez --riot-id "Pseudo#EUW", ou --no-collect pour utiliser les parties déjà téléchargées.')
        collect.main(["--riot-id", args.riot_id, "--count", str(args.count), "--out", ME_DIR, *_riot_options(args)])
    if Path(MODEL).exists():
        winprob.main(["analyze", "--raw-dir", ME_DIR, "--me-dir", ME_DIR, "--limit", str(args.count),
                      "--skip-existing", "--model", MODEL, "--out", CURVES])
    else:
        print(f"(Pas de {MODEL} : le rapport se limite à vos tendances. "
              "Pour les courbes et les repères, lancez `python -m src.run prepare`.)")
    if any(Path(CURVES).glob("*.json")):
        _write_debriefs(args)
    report.main(["--raw-dir", ME_DIR, "--curves", CURVES, "--debriefs", DEBRIEFS, "--bench", args.bench,
                 "--limit", str(args.count), "--window", str(args.window), "--out", args.out])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    riot = argparse.ArgumentParser(add_help=False)
    riot.add_argument("--no-collect", action="store_true", help="n'appelle pas Riot : utilise les parties déjà téléchargées")
    riot.add_argument("--platform", help="serveur de jeu, ex. euw1, na1, kr (défaut : RIOT_PLATFORM dans .env)")
    riot.add_argument("--min-interval", type=float, help="secondes entre deux requêtes (défaut : 1,25, pour une clé de dev)")
    riot.add_argument("--bench", default=str(DEFAULT_BENCH), help="fichier des repères, ex. un par rang")

    a = sub.add_parser("prepare", parents=[riot], help="échantillon de référence, modèle et repères")
    a.add_argument("--tier", default="MASTER", help="IRON..DIAMOND, MASTER, GRANDMASTER ou CHALLENGER")
    a.add_argument("--division", default="II", help="I..IV, ignorée pour Master et au-delà")
    a.add_argument("--pages", type=int, default=1, help="pages de la ligue (IRON..DIAMOND)")
    a.add_argument("--max-players", type=int, default=500, help="joueurs tirés au sort (Master et au-delà)")
    a.add_argument("--per-player", type=int, default=5, help="parties par joueur")
    a.add_argument("--label", help="nom de l'échantillon dans les phrases (défaut : le tier)")
    a.add_argument("--kind", choices=["lineaire", "phases", "monotone"], default=winprob.DEFAULT_KIND,
                   help="type de modèle")
    a.add_argument("--features", choices=list(winprob.FEATURE_SETS), default=winprob.DEFAULT_FEATURES,
                   help="variables du modèle")
    a.set_defaults(func=prepare)

    b = sub.add_parser("report", parents=[riot], help="vos parties : analyse, débriefs et rapport HTML")
    b.add_argument("--riot-id", help='votre Riot ID, ex. "Pseudo#EUW"')
    b.add_argument("--count", type=int, default=20, help="nombre de parties à télécharger et à détailler")
    b.add_argument("--window", type=int, default=20, help="nombre de parties récentes pour les tendances")
    b.add_argument("--llm", action="store_true", help="le résumé de chaque partie est rédigé par un LLM local (Ollama)")
    b.add_argument("--model", default=debrief.DEFAULT_MODEL, help="avec --llm : modèle Ollama")
    b.add_argument("--out", default="results/report.html")
    b.set_defaults(func=make_report)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
