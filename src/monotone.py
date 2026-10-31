"""Régression logistique dont les coefficients sont positifs ou nuls.

Toutes les variables sont des avantages (bleu - rouge) : un avantage ne devrait jamais faire baisser la
probabilité de victoire. Une régression libre donne pourtant des poids négatifs aux kills ou à Nashor, car
ils sont très corrélés à l'or : à or égal, avoir plus de kills signifie avoir moins farmé. Forcer le signe
rend l'explication d'un saut lisible, et un écart de score éventuel se mesure avec `winprob compare`.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin


class MonotoneLogit(ClassifierMixin, BaseEstimator):
    """Même objectif que `LogisticRegression` (pénalité L2 de force 1/C), avec des coefficients >= 0."""

    def __init__(self, C: float = 1.0, max_iter: int = 2000) -> None:
        self.C = C
        self.max_iter = max_iter

    def fit(self, X, y) -> MonotoneLogit:
        X = np.asarray(X, dtype=float)
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        sign = 2.0 * encoded - 1

        def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
            coef, intercept = w[:-1], w[-1]
            margin = sign * (X @ coef + intercept)
            slope = -sign * expit(-margin)
            value = self.C * np.logaddexp(0, -margin).sum() + 0.5 * coef @ coef
            return value, np.append(self.C * (X.T @ slope) + coef, self.C * slope.sum())

        result = minimize(
            objective, np.zeros(self.n_features_in_ + 1), jac=True, method="L-BFGS-B",
            bounds=[(0, None)] * self.n_features_in_ + [(None, None)],
            options={"maxiter": self.max_iter, "maxfun": 10 * self.max_iter, "ftol": 1e-10},
        )
        self.coef_ = result.x[:-1].reshape(1, -1)
        self.intercept_ = result.x[-1:]
        return self

    def decision_function(self, X) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_[0] + self.intercept_[0]

    def predict_proba(self, X) -> np.ndarray:
        p = expit(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X) -> np.ndarray:
        return self.classes_[(self.decision_function(X) > 0).astype(int)]
