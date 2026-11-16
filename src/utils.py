"""Utilitaires transverses : seed, mémoire GPU et journalisation.

Ces fonctions sont volontairement défensives : elles doivent fonctionner sur
Colab (GPU), sur un poste sans GPU et dans un simple test d'import.
Les imports lourds (``torch``, ``numpy``) sont donc faits à l'intérieur des
fonctions concernées.
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .config import ALL_EXPERIMENTS_CSV, EXPERIMENT_FIELDS

# ---------------------------------------------------------------------------
# Reproductibilité
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Fixe la seed de toutes les bibliothèques utilisées.

    On fixe ``random``, ``numpy`` et ``torch`` (CPU + CUDA). ``torch`` et
    ``numpy`` sont optionnels : si l'un n'est pas installé, on l'ignore.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy est une dépendance du projet
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Déterminisme des opérations CUDA (léger coût en performance).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# Mémoire GPU
# ---------------------------------------------------------------------------


def gpu_memory_mb() -> Dict[str, float]:
    """Retourne la mémoire GPU du pic, en Mo.

    Clés retournées :
        - ``allocated`` : pic de mémoire allouée par PyTorch (Mo)
        - ``reserved``  : pic de mémoire réservée par le cache CUDA (Mo)

    Sans GPU (ou sans torch), les valeurs valent ``0.0``.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return {"allocated": 0.0, "reserved": 0.0}
        # On lit les pics : appeler reset_peak_memory_stats() avant l'entraînement.
        return {
            "allocated": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1),
            "reserved": round(torch.cuda.max_memory_reserved() / 1024 ** 2, 1),
        }
    except Exception:  # pragma: no cover - environnement sans GPU
        return {"allocated": 0.0, "reserved": 0.0}


def reset_gpu_peak_memory() -> None:
    """Remet à zéro le compteur de pic mémoire CUDA avant un entraînement."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover
        pass


def gpu_name() -> str:
    """Nom du GPU utilisé (``cpu`` si aucun GPU n'est disponible)."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:  # pragma: no cover
        pass
    return "cpu"


def count_parameters_millions(model: Any) -> float:
    """Nombre de paramètres du modèle, en millions (arrondi à 2 décimales)."""
    try:
        total = sum(p.numel() for p in model.parameters())
        return round(total / 1e6, 2)
    except Exception:  # pragma: no cover
        return 0.0


# ---------------------------------------------------------------------------
# Journalisation des expériences
# ---------------------------------------------------------------------------


def log_experiment(
    row: Dict[str, Any],
    csv_path: Optional[Path] = None,
    field_order: Optional[list] = None,
) -> Path:
    """Ajoute **une ligne** au journal cumulé ``results/all_experiments.csv``.

    - Le fichier et ses dossiers parents sont créés au besoin.
    - L'en-tête est écrit uniquement si le fichier est nouveau.
    - Les clés manquantes sont remplies par une chaîne vide.
    - Les valeurs ``None`` deviennent des chaînes vides (pas de « None » en CSV).

    Retourne le chemin du fichier CSV utilisé.
    """
    chemin = Path(csv_path) if csv_path is not None else ALL_EXPERIMENTS_CSV
    chemin.parent.mkdir(parents=True, exist_ok=True)

    colonnes = field_order or EXPERIMENT_FIELDS
    # On conserve d'éventuelles colonnes supplémentaires passées par l'appelant.
    for cle in row:
        if cle not in colonnes:
            colonnes.append(cle)

    nouveau_fichier = not chemin.exists() or chemin.stat().st_size == 0
    valeurs = {c: ("" if row.get(c) is None else row.get(c)) for c in colonnes}

    # ``newline=""`` évite les lignes vides sous Windows.
    with chemin.open("a", newline="", encoding="utf-8") as fichier:
        writer = csv.DictWriter(fichier, fieldnames=colonnes)
        if nouveau_fichier:
            writer.writeheader()
        writer.writerow(valeurs)

    return chemin


def write_json(path: Path, payload: Dict[str, Any]) -> Path:
    """Écrit un dictionnaire au format JSON lisible (UTF-8, indenté)."""
    chemin = Path(path)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with chemin.open("w", encoding="utf-8") as fichier:
        json.dump(payload, fichier, ensure_ascii=False, indent=2)
    return chemin


class Chronometre:
    """Chronomètre simple basé sur ``time.perf_counter``.

    Utilisation :
        >>> with Chronometre() as c:
        ...     ...  # code à mesurer
        >>> c.secondes
    """

    def __init__(self) -> None:
        self._debut: float = 0.0
        self.secondes: float = 0.0

    def __enter__(self) -> "Chronometre":
        self._debut = time.perf_counter()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.secondes = time.perf_counter() - self._debut

    @property
    def minutes(self) -> float:
        return self.secondes / 60.0


def format_duree(secondes: float) -> str:
    """Formate une durée en ``Xm Ys`` (ou ``Ys`` si moins d'une minute)."""
    secondes = float(secondes)
    if secondes < 60:
        return f"{secondes:.1f}s"
    minutes, reste = divmod(int(secondes), 60)
    return f"{minutes}m {reste:02d}s"


__all__ = [
    "set_seed",
    "gpu_memory_mb",
    "reset_gpu_peak_memory",
    "gpu_name",
    "count_parameters_millions",
    "log_experiment",
    "write_json",
    "Chronometre",
    "format_duree",
]
