"""Métriques de classification (scikit-learn).

Toutes les expériences utilisent **exactement** ces métriques, afin que les
résultats soient comparables. Tâche IMDB = classification binaire
(0 = négatif, 1 = positif).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    average: str = "binary",
) -> Dict[str, float]:
    """Calcule accuracy, precision, recall et F1.

    Paramètres
    ----------
    y_true : étiquettes de référence
    y_pred : étiquettes prédites
    average : ``"binary"`` (classe positive = 1) ou ``"macro"`` / ``"weighted"``

    Retourne un dictionnaire arrondi à 4 décimales.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    y_true = list(y_true)
    y_pred = list(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true et y_pred n'ont pas la même longueur.")
    if not y_true:
        raise ValueError("Impossible de calculer des métriques sur une liste vide.")

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(
            float(precision_score(y_true, y_pred, average=average, zero_division=0)), 4
        ),
        "recall": round(
            float(recall_score(y_true, y_pred, average=average, zero_division=0)), 4
        ),
        "f1": round(
            float(f1_score(y_true, y_pred, average=average, zero_division=0)), 4
        ),
    }


def classification_report_text(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    target_names: Iterable[str] = ("négatif", "positif"),
) -> str:
    """Retourne le rapport de classification scikit-learn sous forme de texte."""
    from sklearn.metrics import classification_report

    return classification_report(
        list(y_true),
        list(y_pred),
        target_names=list(target_names),
        digits=4,
        zero_division=0,
    )


def confusion_matrix_values(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> List[List[int]]:
    """Matrice de confusion 2x2 sous forme de listes (sérialisable en JSON)."""
    from sklearn.metrics import confusion_matrix

    matrice = confusion_matrix(list(y_true), list(y_pred), labels=[0, 1])
    return matrice.astype(int).tolist()


def format_metrics(metrics: Dict[str, float]) -> str:
    """Formate les métriques pour l'affichage console."""
    ordre = ("accuracy", "precision", "recall", "f1")
    morceaux = [f"{cle}={metrics[cle]:.4f}" for cle in ordre if cle in metrics]
    return " | ".join(morceaux)


__all__ = [
    "compute_metrics",
    "classification_report_text",
    "confusion_matrix_values",
    "format_metrics",
]
