"""Configuration de référence du projet.

Toutes les expériences partent de la dataclass :class:`Config` définie ici.

Règle d'équipe : une expérience ne modifie **qu'un seul** champ par rapport à la
baseline (sauf les expériences d'interaction explicitement documentées).

Exemple d'utilisation :

    >>> from src.config import Config
    >>> cfg = Config()
    >>> cfg_batch16 = Config.from_dict({"batch_size": 16, "name": "batch_16"})
    >>> cfg_batch16.changed_fields(Config())
    ['batch_size', 'name']

Ce module ne dépend que de la bibliothèque standard : il s'importe même sans
``torch`` installé, ce qui permet de valider la structure du dépôt rapidement.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Chemins de référence (calculés depuis l'emplacement du fichier)
# src/config.py -> parent = src/ -> parent = racine du dépôt
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

EXPERIMENTS_CSV: Path = PROJECT_ROOT / "experiments" / "experiments_plan.csv"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
ALL_EXPERIMENTS_CSV: Path = RESULTS_DIR / "all_experiments.csv"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

# Colonnes du journal d'expériences (ordre stable pour le CSV cumulé).
EXPERIMENT_FIELDS: List[str] = [
    "id",
    "owner",
    "name",
    "learning_rate",
    "warmup_ratio",
    "scheduler",
    "batch_size",
    "optimizer",
    "train_time_s",
    "gpu_mem_alloc_mb",
    "gpu_mem_reserved_mb",
    "train_loss",
    "val_loss",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "num_params_m",
    "observations",
]

# Valeurs autorisées : validées dans ``Config.__post_init__``.
SCHEDULERS = ("linear", "cosine", "constant")
OPTIMIZERS = ("adamw", "sgd")


@dataclass
class Config:
    """Hyperparamètres de référence — expérience 1, dite « baseline ».

    Les valeurs par défaut correspondent au protocole validé par l'équipe.
    Elles ne doivent être modifiées qu'avec l'accord de tous.
    """

    # --- Identité de l'expérience -------------------------------------------
    experiment_id: int = 1
    owner: str = "Nada"
    name: str = "baseline"

    # --- Modèle et données ---------------------------------------------------
    model_name: str = "distilbert-base-uncased"
    dataset_name: str = "imdb"
    max_length: int = 128
    train_size: int = 4000
    val_size: int = 1000
    test_size: int = 1000

    # --- Optimisation --------------------------------------------------------
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    optimizer_type: str = "adamw"  # "adamw" ou "sgd"
    scheduler_type: str = "linear"  # "linear", "cosine" ou "constant"
    warmup_ratio: float = 0.0  # proportion des steps totaux passée en warmup
    batch_size: int = 32
    num_epochs: int = 2
    seed: int = 42

    # --- Exécution / journalisation -----------------------------------------
    output_dir: str = "outputs"  # ignoré par git
    save_model_dir: str = ""  # ex. /content/drive/MyDrive/... (vide = pas de sauvegarde)
    max_train_minutes: float = 10.0  # garde-fou du budget Colab

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        """Vérifie la cohérence des valeurs (échec rapide et explicite)."""
        if self.optimizer_type not in OPTIMIZERS:
            raise ValueError(
                f"optimizer_type={self.optimizer_type!r} invalide ; "
                f"valeurs autorisées : {OPTIMIZERS}"
            )
        if self.scheduler_type not in SCHEDULERS:
            raise ValueError(
                f"scheduler_type={self.scheduler_type!r} invalide ; "
                f"valeurs autorisées : {SCHEDULERS}"
            )
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio doit être dans [0, 1[.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate doit être strictement positif.")
        if self.batch_size <= 0 or self.num_epochs <= 0:
            raise ValueError("batch_size et num_epochs doivent être positifs.")
        if self.max_length <= 0:
            raise ValueError("max_length doit être positif.")
        if min(self.train_size, self.val_size, self.test_size) <= 0:
            raise ValueError("Les tailles train/val/test doivent être positives.")

    # ------------------------------------------------------------------
    # Conversions et utilitaires
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Retourne la configuration sous forme de dictionnaire sérialisable."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "Config":
        """Construit une configuration à partir d'un dictionnaire partiel.

        Les clés inconnues sont ignorées avec un avertissement, pour rester
        tolérant face à d'anciens fichiers de configuration.
        """
        champs = {f.name for f in dataclasses.fields(cls)}
        inconnues = set(values) - champs
        if inconnues:
            print(f"[config] champs ignorés (inconnus) : {sorted(inconnues)}")
        propres = {k: v for k, v in values.items() if k in champs}
        return cls(**propres)

    def changed_fields(self, reference: "Config") -> List[str]:
        """Liste les champs qui diffèrent de ``reference``.

        Sert à vérifier la règle « un seul hyperparamètre modifié » :
        les champs d'identité (``experiment_id``, ``owner``, ``name``) et les
        champs d'exécution ne comptent pas comme des hyperparamètres.
        """
        ignores = {"experiment_id", "owner", "name", "output_dir",
                   "save_model_dir", "max_train_minutes"}
        return [
            f.name
            for f in dataclasses.fields(self)
            if f.name not in ignores
            and getattr(self, f.name) != getattr(reference, f.name)
        ]

    def tag(self) -> str:
        """Identifiant court et lisible : ``exp_01_baseline``."""
        nom = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.name)
        return f"exp_{self.experiment_id:02d}_{nom}"

    def resume(self) -> str:
        """Résumé compact des hyperparamètres, utile pour les logs."""
        return (
            f"lr={self.learning_rate:g} | warmup={self.warmup_ratio:g} | "
            f"sched={self.scheduler_type} | batch={self.batch_size} | "
            f"opt={self.optimizer_type} | epochs={self.num_epochs} | "
            f"seed={self.seed}"
        )


# Alias pratique pour les notebooks : ``from src.config import BASELINE``.
BASELINE = Config()

__all__ = [
    "Config",
    "BASELINE",
    "EXPERIMENTS_CSV",
    "ALL_EXPERIMENTS_CSV",
    "RESULTS_DIR",
    "PROJECT_ROOT",
    "EXPERIMENT_FIELDS",
    "SCHEDULERS",
    "OPTIMIZERS",
]
