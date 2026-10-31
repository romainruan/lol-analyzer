"""Entraînement et évaluation du modèle de probabilité de victoire.

    python -m src.train --data data/dataset.csv --out results

Un modèle est entraîné par minute (5, 10, 15...), avec une validation croisée
**par partie** : les prédictions sont toujours faites sur des parties que le
modèle n'a pas vues. Trois modèles sont comparés :
  - logreg_base   : régression logistique sur les écarts d'équipe
  - gbm_base      : gradient boosting sur les mêmes variables
  - logreg_lanes  : régression logistique avec l'écart d'or de chaque lane
                    (top / jungle / mid / bot / support) à la place de l'or total
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import DEFAULT_MINUTES

BASE_FEATURES = [
    "gold_diff", "xp_diff", "level_diff", "cs_diff", "kills_diff", "towers_diff",
    "dragons_diff", "heralds_diff", "grubs_diff", "barons_diff", "wards_diff",
]
# variables en plus, à évaluer avec `winprob compare` avant d'en faire le choix par défaut
EXTENDED_FEATURES = BASE_FEATURES + [
    "inhibitors_diff", "elders_diff", "souls_diff", "atakhans_diff", "recent_kills_diff", "gold_share_diff",
]
LANE_FEATURES = [f for f in BASE_FEATURES if f != "gold_diff"] + [
    "top_gold_diff", "jng_gold_diff", "mid_gold_diff", "bot_gold_diff", "sup_gold_diff",
    "mid_cs_diff",
]
MIN_GAMES = 50

MODEL_LABELS = {
    "logreg_base": "Régression logistique (équipes)",
    "gbm_base": "Gradient boosting (équipes)",
    "logreg_lanes": "Régression logistique (lanes)",
}
# Palette de référence, slots 1 à 3 (bleu, orange, aqua), surface claire
MODEL_COLORS = {"logreg_base": "#2a78d6", "gbm_base": "#eb6834", "logreg_lanes": "#1baf7a"}


def make_models() -> dict[str, tuple[list[str], Pipeline]]:
    def logreg() -> Pipeline:
        return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])

    gbm = Pipeline(
        [("clf", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                max_iter=200, random_state=0))]
    )
    return {
        "logreg_base": (BASE_FEATURES, logreg()),
        "gbm_base": (BASE_FEATURES, gbm),
        "logreg_lanes": (LANE_FEATURES, logreg()),
    }


def load_dataset(path: str | Path) -> pd.DataFrame:
    if not Path(path).exists():
        raise SystemExit(f"{path} introuvable : lancez d'abord `python -m src.build_dataset`.")
    # patch en texte : relu comme un nombre, 16.1 et 16.10 seraient confondus
    return pd.read_csv(path, dtype={"patch": str})


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Garde les lignes où toutes les variables sont connues (même base pour tous les modèles)."""
    needed = sorted(set(BASE_FEATURES + LANE_FEATURES))
    clean = df.dropna(subset=needed)
    dropped = len(df) - len(clean)
    if dropped:
        print(f"{dropped} lignes ignorées (rôles non détectés).")
    return clean


def oof_proba(d: pd.DataFrame, features: list[str], model: Pipeline, n_splits: int) -> np.ndarray:
    """Probabilités « hors échantillon » : chaque partie est prédite par un modèle qui ne l'a pas vue."""
    cv = GroupKFold(n_splits=min(n_splits, d["match_id"].nunique()))
    return cross_val_predict(
        model, d[features], d["blue_win"], groups=d["match_id"], cv=cv, method="predict_proba"
    )[:, 1]


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, n: int = 300, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        scores.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def evaluate(df: pd.DataFrame, minutes=DEFAULT_MINUTES, n_splits: int = 5) -> pd.DataFrame:
    rows = []
    for m in minutes:
        d = df[df["minute"] == m]
        if d["match_id"].nunique() < MIN_GAMES:
            print(f"Minute {m} : moins de {MIN_GAMES} parties, ignorée.")
            continue
        y = d["blue_win"].to_numpy()
        for name, (features, model) in make_models().items():
            p = oof_proba(d, features, model, n_splits)
            lo, hi = bootstrap_auc_ci(y, p)
            rows.append(
                {
                    "minute": m,
                    "model": name,
                    "n_games": len(d),
                    "auc": roc_auc_score(y, p),
                    "auc_lo": lo,
                    "auc_hi": hi,
                    "accuracy": accuracy_score(y, p > 0.5),
                    "log_loss": log_loss(y, p),
                    "brier": brier_score_loss(y, p),
                }
            )
    return pd.DataFrame(rows)


def lane_coefficients(df: pd.DataFrame, minute: int) -> pd.DataFrame:
    """Poids de chaque variable (standardisée) dans le modèle « lanes » à une minute donnée."""
    d = df[df["minute"] == minute]
    _, model = make_models()["logreg_lanes"]
    model.fit(d[LANE_FEATURES], d["blue_win"])
    coefs = model.named_steps["clf"].coef_[0]
    out = pd.DataFrame({"feature": LANE_FEATURES, "coef_std": coefs})
    out["minute"] = minute
    return out.reindex(out["coef_std"].abs().sort_values(ascending=False).index)


def calibration_table(df: pd.DataFrame, minute: int, n_splits: int = 5, n_bins: int = 5) -> pd.DataFrame:
    """Quand le modèle annonce X %, quelle part des parties est réellement gagnée ?"""
    d = df[df["minute"] == minute]
    features, model = make_models()["logreg_base"]
    p = oof_proba(d, features, model, n_splits)
    observed, predicted = calibration_curve(d["blue_win"], p, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"minute": minute, "predit": predicted, "observe": observed})


def plot_auc(metrics: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink_soft, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=surface)
    ax.set_facecolor(surface)
    offsets = {"logreg_base": -0.3, "gbm_base": 0.0, "logreg_lanes": 0.3}  # évite le recouvrement
    for name, g in metrics.groupby("model"):
        g = g.sort_values("minute")
        x = g["minute"] + offsets.get(name, 0)
        ax.errorbar(
            x, g["auc"],
            yerr=[g["auc"] - g["auc_lo"], g["auc_hi"] - g["auc"]],
            color=MODEL_COLORS[name], label=MODEL_LABELS[name],
            linewidth=2, marker="o", markersize=6, markeredgecolor=surface, markeredgewidth=1.5,
            elinewidth=1, capsize=0,
        )
    ax.set_xlabel("Minute de la partie", color=ink_soft)
    ax.set_ylabel("AUC (hors échantillon)", color=ink_soft)
    ax.set_title("Prédire la victoire : AUC par minute (IC 95 % bootstrap)", color=ink, loc="left")
    ax.set_xticks(sorted(metrics["minute"].unique()))
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.tick_params(colors=ink_soft, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.legend(frameon=False, labelcolor=ink_soft, loc="upper center",
              bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=8, handlelength=1.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/dataset.csv")
    p.add_argument("--out", default="results")
    p.add_argument("--minutes", type=int, nargs="+", default=list(DEFAULT_MINUTES))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--calib-minute", type=int, default=15)
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = prepare(load_dataset(args.data))

    metrics = evaluate(df, args.minutes, args.folds)
    if metrics.empty:
        print("Pas assez de parties pour évaluer. Collectez-en davantage.")
        return
    metrics.to_csv(out / "metrics_by_minute.csv", index=False)
    plot_auc(metrics, out / "auc_by_minute.png")

    print("\nAUC par minute (validation croisée par partie) :")
    print(metrics.pivot(index="minute", columns="model", values="auc").round(3).to_string())
    print("\nPrécision par minute :")
    print(metrics.pivot(index="minute", columns="model", values="accuracy").round(3).to_string())

    coefs = pd.concat([lane_coefficients(df, m) for m in metrics["minute"].unique()])
    coefs.to_csv(out / "lane_coefficients.csv", index=False)
    mid = int(metrics["minute"].unique()[len(metrics["minute"].unique()) // 2])
    print(f"\nPoids des variables à {mid} min (modèle lanes, variables standardisées) :")
    print(coefs[coefs["minute"] == mid][["feature", "coef_std"]].round(3).to_string(index=False))

    if args.calib_minute in set(metrics["minute"]):
        calib = calibration_table(df, args.calib_minute, args.folds)
        calib.to_csv(out / "calibration.csv", index=False)
        print(f"\nCalibration à {args.calib_minute} min (prédit vs observé) :")
        print(calib[["predit", "observe"]].round(3).to_string(index=False))

    print(f"\nRésultats enregistrés dans {out}/")


if __name__ == "__main__":
    main()
