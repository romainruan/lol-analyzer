"""Courbe de probabilité de victoire d'une partie et moments clés.

Trois sous-commandes :

    # 1. entraîner un modèle valable à TOUTES les minutes (jeu de données --all-minutes)
    python -m src.build_dataset --all-minutes --out data/dataset_all.csv
    python -m src.winprob train --data data/dataset_all.csv

    # 2. analyser vos parties (data/raw/me) : un JSON et un graphique par partie
    python -m src.winprob analyze --raw-dir data/raw/me --limit 5

    # 3. ou une partie précise
    python -m src.winprob analyze --raw-dir data/raw/me --match-id EUW1_1234567890
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import BLUE, RED, build_rows
from .train import BASE_FEATURES

TIME_SCALE = 10.0
NO_EVENT = "pas d'événement majeur, l'écart d'or ou d'XP évolue"
MAX_MINUTE = 40  # limite du jeu d'entraînement (--all-minutes) : au-delà on n'extrapole pas
REPORT_MINUTES = (5, 10, 15, 20, 25)
FIRST_MINUTE = 2  # avant, tous les écarts sont quasi nuls
PHASES = ((2, 5), (6, 10), (11, 15), (16, 25), (26, MAX_MINUTE))
# les variables d'une même famille bougent ensemble : les détailler une par une serait trompeur
FAMILIES = {
    "gold_diff": "ressources", "xp_diff": "ressources", "level_diff": "ressources", "cs_diff": "ressources",
    "kills_diff": "combats",
    "towers_diff": "objectifs", "dragons_diff": "objectifs", "heralds_diff": "objectifs",
    "grubs_diff": "objectifs", "barons_diff": "objectifs",
    "wards_diff": "vision",
}
MONSTERS = {
    "DRAGON": "le Dragon",
    "RIFTHERALD": "le Héraut",
    "BARON_NASHOR": "Nashor",
    "HORDE": "des Larves du Néant",
    "ATAKHAN": "Atakhan",
}


# ----------------------------------------------------------------------------- modèle
def design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Variables d'équipe + leur produit avec la minute.

    Un avantage de 500 d'or ne pèse pas pareil à 5 et à 25 minutes. Plutôt que
    d'entraîner un modèle par minute, on laisse le poids de chaque variable
    évoluer linéairement avec le temps (terme `variable * minute/10`).
    """
    X = df[BASE_FEATURES].astype(float).copy()
    t = df["minute"].astype(float).clip(upper=MAX_MINUTE) / TIME_SCALE
    for f in BASE_FEATURES:
        X[f"{f}*t"] = X[f] * t
    return X


def load_dataset(path: str | Path) -> pd.DataFrame:
    # patch en texte : relu comme un nombre, 16.1 et 16.10 seraient confondus
    return pd.read_csv(path, dtype={"patch": str})


def make_curve_model() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])


def fit_curve_model(df: pd.DataFrame) -> Pipeline:
    model = make_curve_model()
    model.fit(design_matrix(df), df["blue_win"])
    return model


def _oof_curve_proba(df: pd.DataFrame, n_splits: int = 5) -> np.ndarray:
    """Probabilités hors échantillon : chaque partie est prédite par un modèle qui ne l'a pas vue."""
    cv = GroupKFold(n_splits=min(n_splits, df["match_id"].nunique()))
    return cross_val_predict(
        make_curve_model(), design_matrix(df), df["blue_win"],
        groups=df["match_id"], cv=cv, method="predict_proba",
    )[:, 1]


def calibration_by_phase(y: np.ndarray, p: np.ndarray, minutes: np.ndarray, n_bins: int = 5) -> pd.DataFrame:
    """Quand le modèle annonce X %, quelle part des parties est gagnée ? Une table par phase de jeu,
    car un bon score global peut cacher un modèle trop sûr de lui en début de partie."""
    rows = []
    for first, last in PHASES:
        mask = (minutes >= first) & (minutes <= last)
        if not mask.any():
            continue
        y_phase, p_phase = y[mask], p[mask]
        for chunk in np.array_split(np.argsort(p_phase, kind="stable"), n_bins):  # effectifs égaux
            if len(chunk):
                predicted, observed = p_phase[chunk].mean(), y_phase[chunk].mean()
                rows.append({"phase": f"{first}-{last}", "n": len(chunk), "predit": predicted,
                             "observe": observed, "ecart": abs(predicted - observed)})
    return pd.DataFrame(rows)


def curve_calibration(df: pd.DataFrame, n_splits: int = 5, n_bins: int = 5) -> pd.DataFrame:
    """Calibration du modèle de courbe, mesurée hors échantillon (validation croisée par partie)."""
    p = _oof_curve_proba(df, n_splits)
    return calibration_by_phase(df["blue_win"].to_numpy(), p, df["minute"].to_numpy(), n_bins)


def evaluate_curve_model(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """AUC hors échantillon (validation croisée par partie) à quelques minutes repères."""
    p = _oof_curve_proba(df, n_splits)
    rows = []
    for m in REPORT_MINUTES:
        mask = (df["minute"] == m).to_numpy()
        if mask.sum() >= 50 and df.loc[mask, "blue_win"].nunique() == 2:
            rows.append(
                {"minute": m, "n_games": int(mask.sum()),
                 "auc": roc_auc_score(df.loc[mask, "blue_win"], p[mask])}
            )
    out = pd.DataFrame(rows)
    out.attrs["log_loss"] = log_loss(df["blue_win"], p)
    return out


def _split_auc(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Entraîne sur `train`, mesure l'AUC sur `test` aux minutes repères."""
    proba = fit_curve_model(train).predict_proba(design_matrix(test))[:, 1]
    rows = []
    for m in REPORT_MINUTES:
        mask = (test["minute"] == m).to_numpy()
        if mask.sum() >= 30 and test.loc[mask, "blue_win"].nunique() == 2:
            rows.append({"minute": m, "n_games": int(mask.sum()),
                         "auc": roc_auc_score(test.loc[mask, "blue_win"], proba[mask])})
    out = pd.DataFrame(rows)
    out.attrs.update(n_train=int(train["match_id"].nunique()), n_test=int(test["match_id"].nunique()),
                     log_loss=log_loss(test["blue_win"], proba))
    return out


def evaluate_time_split(df: pd.DataFrame, test_fraction: float = 0.2) -> pd.DataFrame:
    """Test dans le temps : on entraîne sur les parties les plus anciennes et on mesure l'AUC sur
    les plus récentes, ce qu'une validation croisée ne fait pas (elle mélange les dates)."""
    starts = df.groupby("match_id")["game_creation"].first().sort_values()
    n_test = max(1, int(round(len(starts) * test_fraction)))
    is_test = df["match_id"].isin(set(starts.index[-n_test:]))
    return _split_auc(df[~is_test], df[is_test])


def evaluate_patch_holdout(df: pd.DataFrame, min_games: int = 300) -> pd.DataFrame:
    """Un patch à la fois est mis de côté pour le test, le modèle est entraîné sur tous les autres :
    on mesure si le modèle tient d'un patch à l'autre. Les patchs de moins de `min_games` parties
    ne servent qu'à l'entraînement."""
    games = df.groupby("patch")["match_id"].nunique()
    parts = []
    for patch in games[games >= min_games].index:
        is_test = df["patch"] == patch
        r = _split_auc(df[~is_test], df[is_test])
        r.insert(0, "patch", patch)
        r["n_train"] = r.attrs["n_train"]
        parts.append(r)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ----------------------------------------------------------------------------- analyse
def find_turning_points(
    minutes: list[int], probs: list[float], k: int = 3, min_delta: float = 0.03
) -> list[dict]:
    """Les k plus gros sauts de probabilité entre deux minutes consécutives."""
    candidates = []
    for i in range(1, len(probs)):
        if minutes[i] - minutes[i - 1] != 1:
            continue
        delta = probs[i] - probs[i - 1]
        if abs(delta) >= min_delta:
            candidates.append(
                {"minute": minutes[i], "before": probs[i - 1], "after": probs[i], "delta": delta}
            )
    top = sorted(candidates, key=lambda c: abs(c["delta"]), reverse=True)[:k]
    return sorted(top, key=lambda c: c["minute"])


def jump_contributions(model: Pipeline, before: pd.Series, after: pd.Series, jump: float) -> dict[str, float]:
    """Répartit un saut de probabilité (en points) entre les familles de variables.

    Le modèle est linéaire en logit : la variation de chaque variable a un effet exact sur le logit,
    et `jump` est réparti au prorata. Dit ce qui a bougé la prédiction, pas ce qui a causé la victoire."""
    coefs = model.named_steps["clf"].coef_[0] / model.named_steps["scale"].scale_
    effects = dict.fromkeys(FAMILIES.values(), 0.0)
    for column, coef, change in zip(before.index, coefs, after - before, strict=True):
        effects[FAMILIES[column.split("*")[0]]] += coef * change
    total = sum(effects.values())
    if total == 0:
        return effects
    return {family: round(jump * effect / total, 1) + 0.0 for family, effect in effects.items()}


def collapse_duplicates(items: list[str]) -> list[str]:
    """Regroupe les phrases identiques : trois fois « prennent des Larves » -> « ... (×3) »."""
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return [f"{text} (×{n})" if n > 1 else text for text, n in counts.items()]


def describe_events(events: list[dict], participants: list[dict], my_team: int | None) -> list[str]:
    """Traduit les événements bruts de la timeline en phrases courtes."""
    champ = {p["participantId"]: p.get("championName", "?") for p in participants}
    team = {p["participantId"]: p["teamId"] for p in participants}

    def side(team_id: int) -> str:
        if my_team is None:
            return "bleus" if team_id == BLUE else "rouges"
        return "alliés" if team_id == my_team else "ennemis"

    def who(pid: int) -> str:
        if pid not in champ:
            return "un sbire ou une tourelle"
        return f"{champ[pid]} ({side(team[pid])})"

    objectives, kills = [], []
    for e in events:
        t = e.get("type")
        if t == "ELITE_MONSTER_KILL":
            if e.get("killerTeamId") not in (BLUE, RED):  # monstre tué sans équipe en jeu
                continue
            name = MONSTERS.get(e.get("monsterType"), str(e.get("monsterType")))
            objectives.append(f"Les {side(e.get('killerTeamId', 0))} prennent {name}")
        elif t == "BUILDING_KILL":
            destroyer = RED if e.get("teamId") == BLUE else BLUE
            kind = "un inhibiteur" if e.get("buildingType") == "INHIBITOR_BUILDING" else "une tour"
            objectives.append(f"Les {side(destroyer)} détruisent {kind}")
        elif t == "CHAMPION_KILL":
            victim = who(e.get("victimId", 0))
            killer = e.get("killerId", 0)
            kills.append(f"{who(killer)} tue {victim}" if killer else f"{victim} meurt")
    return collapse_duplicates(objectives + kills)[:6]


def analyze_match(
    model: Pipeline, match: dict, timeline: dict, puuid: str | None = None,
    k: int = 3, min_delta: float = 0.03,
) -> dict | None:
    info = match["info"]
    frames = timeline["info"]["frames"]
    df = pd.DataFrame(build_rows(match, timeline, minutes=range(FIRST_MINUTE, len(frames))))
    if df.empty:
        return None
    X = design_matrix(df)
    p_blue = model.predict_proba(X)[:, 1]

    me = next((p for p in info["participants"] if puuid and p.get("puuid") == puuid), None)
    my_team = me["teamId"] if me else None
    perspective = my_team or BLUE
    probs = [float(x if perspective == BLUE else 1 - x) for x in p_blue]
    minutes = [int(m) for m in df["minute"]]

    turning = []
    for tp in find_turning_points(minutes, probs, k, min_delta):
        i = minutes.index(tp["minute"])
        turning.append(
            {
                "minute": tp["minute"],
                "win_prob_before": round(tp["before"], 3),
                "win_prob_after": round(tp["after"], 3),
                "contributions": jump_contributions(model, X.iloc[i - 1], X.iloc[i], tp["delta"] * 100),
                "events": describe_events(
                    frames[tp["minute"]].get("events", []), info["participants"], my_team
                ),
            }
        )

    won = next(t for t in info["teams"] if t["teamId"] == perspective)["win"]
    return {
        "match_id": match["metadata"]["matchId"],
        "perspective": {
            "team": "bleue" if perspective == BLUE else "rouge",
            "champion": me.get("championName") if me else None,
            "role": me.get("teamPosition") if me else None,
            "result": "victoire" if won else "défaite",
        },
        "curve": [{"minute": m, "win_prob": round(p, 3)} for m, p in zip(minutes, probs, strict=True)],
        "turning_points": turning,
    }


# ----------------------------------------------------------------------------- graphique
def plot_curve(analysis: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink_soft, grid, series = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6"
    minutes = [c["minute"] for c in analysis["curve"]]
    probs = [c["win_prob"] * 100 for c in analysis["curve"]]
    tps = analysis["turning_points"]

    # Mise en page en pouces : une ligne de légende de 0,3 po par moment clé sous le graphique
    n = len(tps)
    height = 4.2 + 0.3 * n
    bottom = (0.85 + 0.3 * n) / height
    fig = plt.figure(figsize=(8, height), facecolor=surface)
    ax = fig.add_axes([0.08, bottom, 0.89, 1 - bottom - 0.12], facecolor=surface)
    ax.axhline(50, color=ink_soft, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.plot(minutes, probs, color=series, linewidth=2, clip_on=False)
    for i, tp in enumerate(tps, 1):
        y = tp["win_prob_after"] * 100
        ax.plot(tp["minute"], y, "o", color=series, markersize=9,
                markeredgecolor=surface, markeredgewidth=2, zorder=3, clip_on=False)
        offset = -17 if y > 85 else 11  # sous le point s'il est près du bord haut
        ax.annotate(str(i), (tp["minute"], y), xytext=(0, offset), textcoords="offset points",
                    ha="center", fontsize=9, color=ink, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel("Minute de la partie", color=ink_soft)
    ax.set_ylabel("Probabilité de victoire (%)", color=ink_soft)
    persp = analysis["perspective"]
    who = persp["champion"] or f"équipe {persp['team']}"
    ax.set_title(f"{who} : {persp['result']}", color=ink, loc="left")
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.tick_params(colors=ink_soft, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(grid)

    for i, tp in enumerate(tps, 1):
        label = tp["events"][0] if tp["events"] else NO_EVENT
        line = (f"{i}. {tp['minute']} min : {tp['win_prob_before'] * 100:.0f} % → "
                f"{tp['win_prob_after'] * 100:.0f} %  ({label})")
        fig.text(0.08, (0.15 + 0.3 * (n - i)) / height, line, color=ink_soft, fontsize=8.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


# ----------------------------------------------------------------------------- CLI
def _load(raw_dir: Path, match_id: str) -> tuple[dict, dict]:
    match = json.loads((raw_dir / "matches" / f"{match_id}.json").read_text(encoding="utf-8"))
    timeline = json.loads((raw_dir / "timelines" / f"{match_id}.json").read_text(encoding="utf-8"))
    return match, timeline


def cmd_train(args: argparse.Namespace) -> None:
    df = load_dataset(args.data).dropna(subset=BASE_FEATURES)
    if df["match_id"].nunique() < 50:
        raise SystemExit("Moins de 50 parties : collectez davantage avant d'entraîner.")
    scores = evaluate_curve_model(df)
    print("AUC hors échantillon du modèle « toutes minutes » (validation croisée par partie) :")
    print(scores.round(3).to_string(index=False))
    print(f"Log-loss global : {scores.attrs['log_loss']:.3f}")
    if args.time_split > 0:
        if "game_creation" not in df.columns:
            print("\n(Pas de colonne game_creation : reconstruisez le jeu de données avec build_dataset.)")
        else:
            ts = evaluate_time_split(df, args.time_split)
            print(f"\nTest dans le temps : entraîné sur {ts.attrs['n_train']} parties anciennes, "
                  f"testé sur les {ts.attrs['n_test']} plus récentes")
            print(ts.round(3).to_string(index=False))
            print(f"Log-loss : {ts.attrs['log_loss']:.3f}")
    if args.patch_holdout:
        ph = evaluate_patch_holdout(df)
        if ph.empty:
            print("\n(Aucun patch avec assez de parties pour un test par patch.)")
        else:
            print("\nTest par patch (entraîné sur tous les autres patchs) :")
            print(ph.round(3).to_string(index=False))
    model = fit_curve_model(df)
    out = Path(args.model)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.calibration:
        calib = curve_calibration(df)
        calib.to_csv(out.parent / "curve_calibration.csv", index=False)
        print("\nCalibration du modèle de courbe, hors échantillon (prédit vs observé) :")
        print(calib.round(3).to_string(index=False))
        print(f"Détail : {out.parent / 'curve_calibration.csv'}")
    joblib.dump(model, out)
    print(f"\nModèle enregistré dans {out}")


def select_ids(raw_dir: Path, out_dir: Path, limit: int, skip_existing: bool = False) -> list[str]:
    """Les `limit` parties les plus récentes. Avec `skip_existing`, celles qui ont déjà une courbe
    dans `out_dir` sont écartées avant de compter : on ne traite que les nouvelles."""
    files = sorted((raw_dir / "matches").glob("*.json"),
                   key=lambda f: json.loads(f.read_text(encoding="utf-8"))["info"]
                   .get("gameCreation", 0), reverse=True)
    ids = [f.stem for f in files]
    if skip_existing:
        ids = [i for i in ids if not (out_dir / f"{i}.json").exists()]
    return ids[:limit]


def cmd_analyze(args: argparse.Namespace) -> None:
    model = joblib.load(args.model)
    raw_dir = Path(args.raw_dir)
    puuid_file = Path(args.me_dir) / "puuid.txt"
    puuid = puuid_file.read_text(encoding="utf-8").strip() if puuid_file.exists() else None
    if puuid is None:
        print("(Pas de puuid.txt : la courbe est tracée du point de vue de l'équipe bleue.)")

    if args.match_id:
        ids = [args.match_id]
    else:
        ids = select_ids(raw_dir, Path(args.out), args.limit, args.skip_existing)

    out_dir = Path(args.out)
    for match_id in ids:
        match, timeline = _load(raw_dir, match_id)
        analysis = analyze_match(model, match, timeline, puuid, k=args.top)
        if analysis is None:
            print(f"{match_id} : partie trop courte, ignorée.")
            continue
        (out_dir).mkdir(parents=True, exist_ok=True)
        (out_dir / f"{match_id}.json").write_text(
            json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
        plot_curve(analysis, out_dir / f"{match_id}.png")
        persp = analysis["perspective"]
        print(f"\n{match_id} : {persp['champion'] or 'équipe ' + persp['team']}, {persp['result']}")
        for tp in analysis["turning_points"]:
            print(f"  {tp['minute']:>2} min : {tp['win_prob_before']:.0%} -> {tp['win_prob_after']:.0%}"
                  f"  | {' ; '.join(tp['events'][:3]) or NO_EVENT}")
            parts = [f"{family} {points:+.1f}" for family, points in tp["contributions"].items() if points]
            print(f"           points de probabilité : {', '.join(parts) or 'aucun'}")
    print(f"\nFichiers écrits dans {out_dir}/")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="entraîne le modèle toutes minutes")
    t.add_argument("--data", default="data/dataset_all.csv")
    t.add_argument("--model", default="results/curve_model.joblib")
    t.add_argument("--time-split", type=float, default=0.0, metavar="FRACTION",
                   help="ex. 0.2 : teste aussi sur les 20 %% de parties les plus récentes, "
                        "entraînement sur les autres")
    t.add_argument("--patch-holdout", action="store_true",
                   help="teste aussi chaque patch (300 parties au moins) en entraînant sur les autres")
    t.add_argument("--calibration", action="store_true",
                   help="mesure aussi la calibration par phase de jeu (curve_calibration.csv)")
    t.set_defaults(func=cmd_train)

    a = sub.add_parser("analyze", help="courbe et moments clés de vos parties")
    a.add_argument("--raw-dir", default="data/raw/me")
    a.add_argument("--me-dir", default="data/raw/me")
    a.add_argument("--match-id")
    a.add_argument("--limit", type=int, default=5)
    a.add_argument("--skip-existing", action="store_true",
                   help="ignore les parties qui ont déjà une courbe (ne traite que les nouvelles)")
    a.add_argument("--top", type=int, default=3, help="nombre de moments clés")
    a.add_argument("--model", default="results/curve_model.joblib")
    a.add_argument("--out", default="results/curves")
    a.set_defaults(func=cmd_analyze)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
