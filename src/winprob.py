"""Courbe de probabilité de victoire d'une partie et moments clés.

Sous-commandes :

    # 1. entraîner un modèle valable à TOUTES les minutes (jeu de données --all-minutes)
    python -m src.build_dataset --all-minutes --out data/dataset_all.csv
    python -m src.winprob train --data data/dataset_all.csv

    # 2. analyser vos parties (data/raw/me) : un JSON et un graphique par partie
    python -m src.winprob analyze --raw-dir data/raw/me --limit 5

    # 3. ou une partie précise
    python -m src.winprob analyze --raw-dir data/raw/me --match-id EUW1_1234567890

    # 4. choisir le modèle : comparaison statistique, et utilité de collecter plus de parties
    python -m src.winprob compare --data data/dataset_all.csv --time-split 0.2
    python -m src.winprob learning-curve --data data/dataset_all.csv
"""
from __future__ import annotations

import argparse
import json
import re
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
from .monotone import MonotoneLogit
from .train import BASE_FEATURES, EXTENDED_FEATURES, bootstrap_auc_ci, load_dataset

TIME_SCALE = 10.0
NO_EVENT = "pas d'événement majeur, l'écart d'or ou d'XP évolue"
MAX_MINUTE = 40  # limite du jeu d'entraînement (--all-minutes) : au-delà on n'extrapole pas
REPORT_MINUTES = (5, 10, 15, 20, 25)
FIRST_MINUTE = 2  # avant, tous les écarts sont quasi nuls
EARLY_MINUTE = 5  # avant, le modèle prédit mal (AUC de 0,65 à 0,68 à 5 minutes, voir README)
PHASES = ((2, 5), (6, 10), (11, 15), (16, 25), (26, MAX_MINUTE))
KNOTS = (5, 10, 15, 25, MAX_MINUTE)  # repères des modèles par phases : un poids par variable à chaque repère
FEATURE_SETS = {"base": BASE_FEATURES, "etendu": EXTENDED_FEATURES}
# (type de modèle, variables) comparés par `winprob compare` ; le premier sert de référence
CANDIDATES = {
    "lineaire": ("lineaire", BASE_FEATURES),
    "phases": ("phases", BASE_FEATURES),
    "monotone": ("monotone", BASE_FEATURES),
    "monotone_etendu": ("monotone", EXTENDED_FEATURES),
}
# les variables d'une même famille bougent ensemble : les détailler une par une serait trompeur
FAMILIES = {
    "gold_diff": "ressources", "xp_diff": "ressources", "level_diff": "ressources", "cs_diff": "ressources",
    "gold_share_diff": "ressources",
    "kills_diff": "combats", "recent_kills_diff": "combats",
    "towers_diff": "objectifs", "dragons_diff": "objectifs", "heralds_diff": "objectifs",
    "grubs_diff": "objectifs", "barons_diff": "objectifs", "inhibitors_diff": "objectifs",
    "elders_diff": "objectifs", "souls_diff": "objectifs", "atakhans_diff": "objectifs",
    "wards_diff": "vision",
}
# écarts affichés pour un moment clé : ce qui a réellement bougé dans la partie, sans aucun modèle
GAP_LABELS = {
    "gold_diff": "or", "xp_diff": "XP", "cs_diff": "CS", "kills_diff": "kills", "towers_diff": "tours",
    "dragons_diff": "Dragons", "heralds_diff": "Hérauts", "grubs_diff": "Larves", "barons_diff": "Nashor",
}
MONSTERS = {
    "DRAGON": "le Dragon",
    "RIFTHERALD": "le Héraut",
    "BARON_NASHOR": "Nashor",
    "HORDE": "des Larves du Néant",
    "ATAKHAN": "Atakhan",
}


# ----------------------------------------------------------------------------- modèle
def design_matrix(df: pd.DataFrame, kind: str = "lineaire", features=BASE_FEATURES) -> pd.DataFrame:
    """Variables d'équipe, dont le poids évolue avec la minute.

    Un avantage de 500 d'or ne pèse pas pareil à 5 et à 25 minutes. Plutôt que d'entraîner un modèle par
    minute, on laisse le poids de chaque variable évoluer :
      - `lineaire` : en ligne droite (termes `variable` et `variable * minute/10`) ;
      - sinon, par phases : un poids par repère de `KNOTS` (termes `variable@minute`), interpolé entre deux
        repères. Une phase ne dépend que des parties à ces minutes : le début de partie n'hérite pas de la
        tendance de la fin.
    """
    X = df[list(features)].astype(float)
    minutes = df["minute"].astype(float).clip(upper=MAX_MINUTE)
    if kind == "lineaire":
        X = X.copy()
        for f in features:
            X[f"{f}*t"] = X[f] * (minutes / TIME_SCALE)
        return X
    weights = {knot: np.interp(minutes, KNOTS, one_hot)
               for knot, one_hot in zip(KNOTS, np.eye(len(KNOTS)), strict=True)}
    return pd.DataFrame({f"{f}@{knot}": X[f].to_numpy() * w for f in features for knot, w in weights.items()},
                        index=df.index)


def _variable(column: str) -> str:
    """`gold_diff*t` ou `gold_diff@10` -> `gold_diff`."""
    return re.split(r"[*@]", column)[0]


def design_for(model: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """Les colonnes attendues par un modèle enregistré, quel que soit son type."""
    columns = list(model.feature_names_in_)
    kind = "phases" if any("@" in c for c in columns) else "lineaire"
    return design_matrix(df, kind, list(dict.fromkeys(_variable(c) for c in columns)))[columns]


def make_curve_model(kind: str = "lineaire") -> Pipeline:
    clf = MonotoneLogit() if kind == "monotone" else LogisticRegression(max_iter=2000)
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def fit_curve_model(df: pd.DataFrame, kind: str = "lineaire", features=BASE_FEATURES) -> Pipeline:
    model = make_curve_model(kind)
    model.fit(design_matrix(df, kind, features), df["blue_win"])
    return model


def has_positive_effects(model: Pipeline) -> bool:
    """Aucun avantage ne fait baisser la probabilité : on peut lire le poids de chaque famille de variables."""
    return bool((model.named_steps["clf"].coef_ >= 0).all())


def _oof_curve_proba(df: pd.DataFrame, n_splits: int = 5, kind: str = "lineaire",
                     features=BASE_FEATURES) -> np.ndarray:
    """Probabilités hors échantillon : chaque partie est prédite par un modèle qui ne l'a pas vue."""
    cv = GroupKFold(n_splits=min(n_splits, df["match_id"].nunique()))
    return cross_val_predict(
        make_curve_model(kind), design_matrix(df, kind, features), df["blue_win"],
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


def curve_calibration(df: pd.DataFrame, n_splits: int = 5, n_bins: int = 5, kind: str = "lineaire",
                      features=BASE_FEATURES) -> pd.DataFrame:
    """Calibration du modèle de courbe, mesurée hors échantillon (validation croisée par partie)."""
    p = _oof_curve_proba(df, n_splits, kind, features)
    return calibration_by_phase(df["blue_win"].to_numpy(), p, df["minute"].to_numpy(), n_bins)


def _auc_row(y: np.ndarray, p: np.ndarray, minute: int) -> dict:
    """Une ligne par partie à une minute donnée : l'intervalle de confiance se rééchantillonne par partie."""
    lo, hi = bootstrap_auc_ci(y, p)
    return {"minute": minute, "n_games": len(y), "auc": roc_auc_score(y, p), "auc_lo": lo, "auc_hi": hi}


def evaluate_curve_model(df: pd.DataFrame, n_splits: int = 5, kind: str = "lineaire",
                         features=BASE_FEATURES) -> pd.DataFrame:
    """AUC hors échantillon (validation croisée par partie) à quelques minutes repères."""
    p = _oof_curve_proba(df, n_splits, kind, features)
    y = df["blue_win"].to_numpy()
    rows = []
    for m in REPORT_MINUTES:
        mask = (df["minute"] == m).to_numpy()
        if mask.sum() >= 50 and len(set(y[mask])) == 2:
            rows.append(_auc_row(y[mask], p[mask], m))
    out = pd.DataFrame(rows)
    out.attrs["log_loss"] = log_loss(y, p)
    return out


def _split_auc(train: pd.DataFrame, test: pd.DataFrame, kind: str = "lineaire",
               features=BASE_FEATURES) -> pd.DataFrame:
    """Entraîne sur `train`, mesure l'AUC sur `test` aux minutes repères."""
    proba = fit_curve_model(train, kind, features).predict_proba(design_matrix(test, kind, features))[:, 1]
    y = test["blue_win"].to_numpy()
    rows = []
    for m in REPORT_MINUTES:
        mask = (test["minute"] == m).to_numpy()
        if mask.sum() >= 30 and len(set(y[mask])) == 2:
            rows.append(_auc_row(y[mask], proba[mask], m))
    out = pd.DataFrame(rows)
    out.attrs.update(n_train=int(train["match_id"].nunique()), n_test=int(test["match_id"].nunique()),
                     log_loss=log_loss(y, proba))
    return out


def _most_recent_games(df: pd.DataFrame, fraction: float) -> pd.Series:
    """Les lignes des parties les plus récentes (`fraction` des parties)."""
    starts = df.groupby("match_id")["game_creation"].first().sort_values()
    n_test = max(1, int(round(len(starts) * fraction)))
    return df["match_id"].isin(set(starts.index[-n_test:]))


def evaluate_time_split(df: pd.DataFrame, test_fraction: float = 0.2, kind: str = "lineaire",
                        features=BASE_FEATURES) -> pd.DataFrame:
    """Test dans le temps : on entraîne sur les parties les plus anciennes et on mesure l'AUC sur
    les plus récentes, ce qu'une validation croisée ne fait pas (elle mélange les dates)."""
    is_test = _most_recent_games(df, test_fraction)
    return _split_auc(df[~is_test], df[is_test], kind, features)


def evaluate_patch_holdout(df: pd.DataFrame, min_games: int = 300, kind: str = "lineaire",
                           features=BASE_FEATURES) -> pd.DataFrame:
    """Un patch à la fois est mis de côté pour le test, le modèle est entraîné sur tous les autres :
    on mesure si le modèle tient d'un patch à l'autre. Les patchs de moins de `min_games` parties
    ne servent qu'à l'entraînement."""
    games = df.groupby("patch")["match_id"].nunique()
    parts = []
    for patch in games[games >= min_games].index:
        is_test = df["patch"] == patch
        r = _split_auc(df[~is_test], df[is_test], kind, features)
        r.insert(0, "patch", patch)
        r["n_train"] = r.attrs["n_train"]
        parts.append(r)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ----------------------------------------------------------------------------- comparaison de modèles
def paired_logloss_gain(y: np.ndarray, p_reference: np.ndarray, p_other: np.ndarray, groups: np.ndarray,
                        n: int = 300, seed: int = 0) -> tuple[float, float, float]:
    """Log-loss gagnée par un modèle sur la référence, avec son intervalle à 95 %.

    Les deux modèles sont mesurés sur les mêmes lignes et le rééchantillonnage se fait par partie, car les
    minutes d'une même partie ne sont pas indépendantes. Si l'intervalle contient 0, rien ne les sépare.
    Comparer deux intervalles qui se chevauchent serait moins puissant que ce test apparié."""
    def row_loss(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    gains = row_loss(p_reference) - row_loss(p_other)
    codes = pd.factorize(groups)[0]
    by_game, rows_by_game = np.bincount(codes, gains), np.bincount(codes)
    picks = np.random.default_rng(seed).integers(0, len(by_game), (n, len(by_game)))
    resampled = by_game[picks].sum(axis=1) / rows_by_game[picks].sum(axis=1)
    lo, hi = np.percentile(resampled, [2.5, 97.5])
    return float(gains.mean()), float(lo), float(hi)


def calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Écart moyen entre la probabilité annoncée et le taux de victoire réel, en tranches d'effectifs égaux."""
    chunks = [c for c in np.array_split(np.argsort(p, kind="stable"), n_bins) if len(c)]
    return float(sum(len(c) * abs(p[c].mean() - y[c].mean()) for c in chunks) / len(y))


def compare_models(df: pd.DataFrame, n_splits: int = 5, time_split: float = 0.0, n_boot: int = 300) -> pd.DataFrame:
    """Compare les modèles de `CANDIDATES` à la référence (le premier), au global et par phase de jeu.

    Mesure hors échantillon : validation croisée par partie, ou avec `time_split`, sur les parties les plus
    récentes après un entraînement sur les autres. Les modèles dont les variables manquent sont ignorés."""
    names = [name for name, (_, features) in CANDIDATES.items() if set(features) <= set(df.columns)]
    test = _most_recent_games(df, time_split) if time_split else pd.Series(True, index=df.index)
    scored = df[test]
    y, minutes, groups = scored["blue_win"].to_numpy(), scored["minute"].to_numpy(), scored["match_id"].to_numpy()

    probas = {}
    for name in names:
        kind, features = CANDIDATES[name]
        probas[name] = (
            fit_curve_model(df[~test], kind, features).predict_proba(design_matrix(scored, kind, features))[:, 1]
            if time_split else _oof_curve_proba(df, n_splits, kind, features))

    masks = {"global": np.ones(len(y), dtype=bool)}
    masks.update({f"{first}-{last}": (minutes >= first) & (minutes <= last) for first, last in PHASES})
    rows = []
    for name in names:
        for phase, mask in masks.items():
            if not mask.any():
                continue
            y_phase, p_phase = y[mask], probas[name][mask]
            gain, lo, hi = paired_logloss_gain(y_phase, probas[names[0]][mask], p_phase, groups[mask], n_boot)
            rows.append({
                "modele": name, "phase": phase, "n_parties": len(set(groups[mask])), "n_lignes": int(mask.sum()),
                "auc": roc_auc_score(y_phase, p_phase) if len(set(y_phase)) == 2 else np.nan,
                "log_loss": log_loss(y_phase, p_phase, labels=[0, 1]),
                "ecart_calibration": calibration_error(y_phase, p_phase),
                "gain": gain, "gain_lo": lo, "gain_hi": hi,
            })
    return pd.DataFrame(rows)


def learning_curve(df: pd.DataFrame, sizes=(250, 500, 1000, 2000, 10**9), test_fraction: float = 0.2,
                   kind: str = "lineaire", features=BASE_FEATURES, seed: int = 0) -> pd.DataFrame:
    """Score sur un même jeu de test (les parties récentes) selon le nombre de parties d'entraînement.
    Si le score ne bouge plus, collecter davantage de parties du même type ne servira à rien."""
    is_test = _most_recent_games(df, test_fraction)
    test = df[is_test]
    games = list(df.loc[~is_test, "match_id"].unique())
    order = np.random.default_rng(seed).permutation(len(games))
    y = test["blue_win"].to_numpy()
    rows = []
    for size in sorted({min(s, len(games)) for s in sizes}):
        train = df[df["match_id"].isin({games[i] for i in order[:size]})]
        p = fit_curve_model(train, kind, features).predict_proba(design_matrix(test, kind, features))[:, 1]
        rows.append({"n_train": size, "n_test": test["match_id"].nunique(),
                     "log_loss": log_loss(y, p), "auc": roc_auc_score(y, p)})
    return pd.DataFrame(rows)


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
    et `jump` est réparti au prorata. Dit ce qui a bougé la prédiction, pas ce qui a causé la victoire.
    À réserver à un modèle sans poids négatif (`has_positive_effects`) : sinon, à cause des corrélations,
    prendre Nashor pourrait s'afficher comme une perte de points sur les objectifs."""
    coefs = model.named_steps["clf"].coef_[0] / model.named_steps["scale"].scale_
    effects = dict.fromkeys(FAMILIES.values(), 0.0)
    for column, coef, change in zip(before.index, coefs, after - before, strict=True):
        effects[FAMILIES[_variable(column)]] += coef * change
    total = sum(effects.values())
    if total == 0:
        return effects
    return {family: round(float(jump * effect / total), 1) + 0.0 for family, effect in effects.items()}


def gap_changes(before: pd.Series, after: pd.Series, sign: int) -> dict[str, int]:
    """Ce qui a bougé entre deux minutes, du point de vue du joueur (`sign` = -1 pour l'équipe rouge)."""
    changes = {label: int(round(sign * (after[column] - before[column]))) for column, label in GAP_LABELS.items()}
    return {label: change for label, change in changes.items() if change}


def format_gaps(gaps: dict[str, int]) -> str:
    return ", ".join(f"{label} {change:+d}" for label, change in gaps.items())


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
    X = design_for(model, df)
    p_blue = model.predict_proba(X)[:, 1]

    me = next((p for p in info["participants"] if puuid and p.get("puuid") == puuid), None)
    my_team = me["teamId"] if me else None
    perspective = my_team or BLUE
    probs = [float(x if perspective == BLUE else 1 - x) for x in p_blue]
    minutes = [int(m) for m in df["minute"]]

    sign = 1 if perspective == BLUE else -1
    explainable = has_positive_effects(model)
    turning = []
    for tp in find_turning_points(minutes, probs, k, min_delta):
        i = minutes.index(tp["minute"])
        moment = {
            "minute": tp["minute"],
            "win_prob_before": round(tp["before"], 3),
            "win_prob_after": round(tp["after"], 3),
            "ecarts": gap_changes(df.iloc[i - 1], df.iloc[i], sign),
            "events": describe_events(
                frames[tp["minute"]].get("events", []), info["participants"], my_team
            ),
        }
        if explainable:
            moment["contributions"] = jump_contributions(model, X.iloc[i - 1], X.iloc[i], tp["delta"] * 100)
        turning.append(moment)

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
def moment_label(tp: dict) -> str:
    """Le premier événement d'un moment clé, et combien d'autres : une partie peut basculer sur plusieurs."""
    events = tp["events"]
    if not events:
        return NO_EVENT
    return events[0] if len(events) == 1 else f"{events[0]}, +{len(events) - 1} autres"


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
    if minutes and minutes[0] < EARLY_MINUTE:
        ax.axvspan(minutes[0], EARLY_MINUTE, color=grid, alpha=0.5, linewidth=0, zorder=0)
        ax.text(minutes[0] + 0.15, 3, "peu fiable", color=ink_soft, fontsize=7.5)
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
        line = (f"{i}. {tp['minute']} min : {tp['win_prob_before'] * 100:.0f} % → "
                f"{tp['win_prob_after'] * 100:.0f} %  ({moment_label(tp)})")
        fig.text(0.08, (0.15 + 0.3 * (n - i)) / height, line, color=ink_soft, fontsize=8.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


# ----------------------------------------------------------------------------- CLI
def _load(raw_dir: Path, match_id: str) -> tuple[dict, dict]:
    match = json.loads((raw_dir / "matches" / f"{match_id}.json").read_text(encoding="utf-8"))
    timeline = json.loads((raw_dir / "timelines" / f"{match_id}.json").read_text(encoding="utf-8"))
    return match, timeline


def _load_columns(path: str, columns: list[str]) -> pd.DataFrame:
    df = load_dataset(path)
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise SystemExit(f"Colonnes absentes de {path} ({', '.join(missing)}) : "
                         "reconstruisez le jeu de données avec `python -m src.build_dataset`.")
    return df


def cmd_train(args: argparse.Namespace) -> None:
    kind, features = args.kind, FEATURE_SETS[args.features]
    df = _load_columns(args.data, features).dropna(subset=features)
    if df["match_id"].nunique() < 50:
        raise SystemExit("Moins de 50 parties : collectez davantage avant d'entraîner.")
    scores = evaluate_curve_model(df, kind=kind, features=features)
    print(f"Modèle « {kind} », variables « {args.features} »")
    print("AUC hors échantillon du modèle « toutes minutes » (validation croisée par partie) :")
    print(scores.round(3).to_string(index=False))
    print(f"Log-loss global : {scores.attrs['log_loss']:.3f}")
    if args.time_split > 0:
        if "game_creation" not in df.columns:
            print("\n(Pas de colonne game_creation : reconstruisez le jeu de données avec build_dataset.)")
        else:
            ts = evaluate_time_split(df, args.time_split, kind, features)
            print(f"\nTest dans le temps : entraîné sur {ts.attrs['n_train']} parties anciennes, "
                  f"testé sur les {ts.attrs['n_test']} plus récentes")
            print(ts.round(3).to_string(index=False))
            print(f"Log-loss : {ts.attrs['log_loss']:.3f}")
    if args.patch_holdout:
        ph = evaluate_patch_holdout(df, kind=kind, features=features)
        if ph.empty:
            print("\n(Aucun patch avec assez de parties pour un test par patch.)")
        else:
            print("\nTest par patch (entraîné sur tous les autres patchs) :")
            print(ph.round(3).to_string(index=False))
    model = fit_curve_model(df, kind, features)
    out = Path(args.model)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.calibration:
        calib = curve_calibration(df, kind=kind, features=features)
        calib.to_csv(out.parent / "curve_calibration.csv", index=False)
        print("\nCalibration du modèle de courbe, hors échantillon (prédit vs observé) :")
        print(calib.round(3).to_string(index=False))
        print(f"Détail : {out.parent / 'curve_calibration.csv'}")
    joblib.dump(model, out)
    print(f"\nModèle enregistré dans {out}")
    if not has_positive_effects(model):
        print("(Certains poids sont négatifs à cause des corrélations entre variables : les moments clés "
              "n'indiqueront pas la part de chaque famille. `--kind monotone` l'évite.)")


def cmd_compare(args: argparse.Namespace) -> None:
    needed = BASE_FEATURES + (["game_creation"] if args.time_split else [])
    df = _load_columns(args.data, needed).dropna(subset=BASE_FEATURES)
    table = compare_models(df, args.folds, args.time_split, args.boot)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print("Mesuré sur les parties les plus récentes" if args.time_split else "Mesuré par validation croisée")
    print("gain = log-loss gagnée sur le modèle « lineaire » (positif = meilleur), intervalle à 95 % par partie")
    print(table.drop(columns="n_lignes").round(4).to_string(index=False))
    print()
    reference = next(iter(CANDIDATES))
    for r in table[(table["phase"] == "global") & (table["modele"] != reference)].itertuples():
        verdict = ("meilleur" if r.gain_lo > 0 else "moins bon" if r.gain_hi < 0
                   else "pas de différence significative")
        print(f"{r.modele} : {verdict} (gain {r.gain:+.4f}, de {r.gain_lo:+.4f} à {r.gain_hi:+.4f})")
    print(f"\nDétail : {out}")


def cmd_learning_curve(args: argparse.Namespace) -> None:
    features = FEATURE_SETS[args.features]
    df = _load_columns(args.data, [*features, "game_creation"]).dropna(subset=features)
    table = learning_curve(df, kind=args.kind, features=features)
    print("Score sur les 20 % de parties les plus récentes, selon le nombre de parties d'entraînement :")
    print(table.round(4).to_string(index=False))


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
    if not Path(args.model).exists():
        raise SystemExit(f"{args.model} introuvable : lancez d'abord `python -m src.winprob train`.")
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
            if tp["ecarts"]:
                print(f"           écarts sur la minute : {format_gaps(tp['ecarts'])}")
            if "contributions" in tp:
                parts = [f"{family} {points:+.1f}" for family, points in tp["contributions"].items() if points]
                print(f"           points de probabilité : {', '.join(parts) or 'aucun'}")
    print(f"\nFichiers écrits dans {out_dir}/")


def main(argv: list[str] | None = None) -> None:
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
    t.add_argument("--kind", choices=["lineaire", "phases", "monotone"], default="lineaire",
                   help="lineaire : poids en ligne droite avec le temps ; phases : un poids par phase de jeu ; "
                        "monotone : par phases, et aucun avantage ne fait baisser la probabilité")
    t.add_argument("--features", choices=list(FEATURE_SETS), default="base",
                   help="etendu ajoute inhibiteurs, Dragon Ancien, âme, Atakhan, kills récents, part de l'or")
    t.set_defaults(func=cmd_train)

    c = sub.add_parser("compare", help="compare les modèles : gain, calibration, intervalle apparié par partie")
    c.add_argument("--data", default="data/dataset_all.csv")
    c.add_argument("--folds", type=int, default=5)
    c.add_argument("--time-split", type=float, default=0.0, metavar="FRACTION",
                   help="ex. 0.2 : mesure sur les 20 %% de parties les plus récentes, entraînement sur les autres")
    c.add_argument("--boot", type=int, default=300, help="tirages du bootstrap")
    c.add_argument("--out", default="results/curve_comparison.csv")
    c.set_defaults(func=cmd_compare)

    lc = sub.add_parser("learning-curve", help="le score progresse-t-il encore avec plus de parties ?")
    lc.add_argument("--data", default="data/dataset_all.csv")
    lc.add_argument("--kind", choices=["lineaire", "phases", "monotone"], default="lineaire")
    lc.add_argument("--features", choices=list(FEATURE_SETS), default="base")
    lc.set_defaults(func=cmd_learning_curve)

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

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
