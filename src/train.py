"""Boucle d'entraînement générique pour l'étude des hyperparamètres.

Fonctionnalités :

- optimiseurs **AdamW** et **SGD** ;
- schedulers **linear**, **cosine** et **constant**, avec **warmup** optionnel ;
- mesure du **temps d'entraînement** et de la **mémoire GPU** (pic alloué/réservé) ;
- évaluation Validation puis Test (accuracy, precision, recall, F1) ;
- journalisation automatique dans ``results/all_experiments.csv`` ;
- arguments en ligne de commande : tout champ de :class:`src.config.Config`.

Exemples
--------
Baseline :

    python -m src.train --id 1 --owner Nada --name baseline

Learning rate 5e-5 :

    python -m src.train --id 3 --owner Nada --name lr_5e-5 --lr 5e-5

SGD avec warmup et cosine :

    python -m src.train --id 7 --owner Fatima --name warmup_0.1_cosine \
        --warmup 0.1 --scheduler cosine
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

try:  # exécution en module : python -m src.train
    from .config import Config, EXPERIMENT_FIELDS, RESULTS_DIR
    from .data import build_datasets, describe_datasets
    from .metrics import (
        classification_report_text,
        compute_metrics,
        confusion_matrix_values,
    )
    from .utils import (
        Chronometre,
        count_parameters_millions,
        format_duree,
        gpu_memory_mb,
        gpu_name,
        log_experiment,
        reset_gpu_peak_memory,
        set_seed,
        write_json,
    )
except ImportError:  # exécution directe : python src/train.py
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.config import Config, EXPERIMENT_FIELDS, RESULTS_DIR
    from src.data import build_datasets, describe_datasets
    from src.metrics import (
        classification_report_text,
        compute_metrics,
        confusion_matrix_values,
    )
    from src.utils import (
        Chronometre,
        count_parameters_millions,
        format_duree,
        gpu_memory_mb,
        gpu_name,
        log_experiment,
        reset_gpu_peak_memory,
        set_seed,
        write_json,
    )


# ---------------------------------------------------------------------------
# Construction de l'optimiseur et du scheduler
# ---------------------------------------------------------------------------


def build_optimizer(model: torch.nn.Module, config: Config) -> torch.optim.Optimizer:
    """Construit AdamW ou SGD.

    On n'applique pas de weight decay aux biais ni aux LayerNorm : c'est la
    pratique standard pour les Transformers et cela évite de dégrader
    l'apprentissage. Pour SGD, un momentum de 0,9 est utilisé.
    """
    sans_decroissance = ("bias", "LayerNorm.weight", "layer_norm.weight")
    parametres = list(model.named_parameters())

    groupes = [
        {
            "params": [p for n, p in parametres if not any(nd in n for nd in sans_decroissance)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in parametres if any(nd in n for nd in sans_decroissance)],
            "weight_decay": 0.0,
        },
    ]

    if config.optimizer_type == "adamw":
        # eps=1e-8 : valeur par défaut de Hugging Face pour les Transformers.
        return torch.optim.AdamW(groupes, lr=config.learning_rate, eps=1e-8)
    if config.optimizer_type == "sgd":
        return torch.optim.SGD(groupes, lr=config.learning_rate, momentum=0.9)
    raise ValueError(f"Optimiseur inconnu : {config.optimizer_type!r}")


def build_scheduler(optimizer: torch.optim.Optimizer, config: Config, total_steps: int):
    """Construit le scheduler avec warmup.

    ``warmup_ratio`` est la proportion des steps totaux consacrée à la montée
    linéaire du learning rate (0.0 = pas de warmup).
    """
    total_steps = max(1, total_steps)
    warmup_steps = int(config.warmup_ratio * total_steps)

    if config.scheduler_type == "linear":
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
    if config.scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
    if config.scheduler_type == "constant":
        # LR constant après le warmup : sert d'ablation (exp. 14).
        return get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps
        )
    raise ValueError(f"Scheduler inconnu : {config.scheduler_type!r}")


# ---------------------------------------------------------------------------
# Boucles d'entraînement et d'évaluation
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    scaler,
    use_amp: bool,
) -> float:
    """Entraîne le modèle pendant une époque et retourne la perte moyenne.

    La précision mixte (``torch.cuda.amp``) est activée uniquement sur GPU.
    """
    model.train()
    perte_cumulee = 0.0
    nb_lots = 0

    for lot in dataloader:
        input_ids = lot["input_ids"].to(device, non_blocking=True)
        attention_mask = lot["attention_mask"].to(device, non_blocking=True)
        labels = lot["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            sorties = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            perte = sorties.loss

        if use_amp:
            scaler.scale(perte).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            perte.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        perte_cumulee += perte.item()
        nb_lots += 1

    return perte_cumulee / max(1, nb_lots)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, List[int], List[int]]:
    """Évalue le modèle : perte moyenne, prédictions et étiquettes réelles."""
    model.eval()
    perte_cumulee = 0.0
    nb_lots = 0
    predictions: List[int] = []
    references: List[int] = []

    for lot in dataloader:
        input_ids = lot["input_ids"].to(device, non_blocking=True)
        attention_mask = lot["attention_mask"].to(device, non_blocking=True)
        labels = lot["labels"].to(device, non_blocking=True)

        sorties = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        perte_cumulee += sorties.loss.item()
        nb_lots += 1

        predictions.extend(torch.argmax(sorties.logits, dim=-1).cpu().tolist())
        references.extend(labels.cpu().tolist())

    return perte_cumulee / max(1, nb_lots), predictions, references


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------


def parse_args(argv: List[str] | None = None) -> Config:
    """Lit les arguments de la ligne de commande et construit une ``Config``.

    Les valeurs par défaut sont celles de la baseline ; un argument omis ne
    modifie donc rien.
    """
    base = Config()
    parser = argparse.ArgumentParser(
        description="Entraînement DistilBERT — étude des hyperparamètres (IMDB).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Identité
    parser.add_argument("--id", dest="experiment_id", type=int, default=base.experiment_id)
    parser.add_argument("--owner", default=base.owner)
    parser.add_argument("--name", default=base.name)

    # Modèle / données
    parser.add_argument("--model", dest="model_name", default=base.model_name)
    parser.add_argument("--dataset", dest="dataset_name", default=base.dataset_name)
    parser.add_argument("--max-length", dest="max_length", type=int, default=base.max_length)
    parser.add_argument("--train-size", dest="train_size", type=int, default=base.train_size)
    parser.add_argument("--val-size", dest="val_size", type=int, default=base.val_size)
    parser.add_argument("--test-size", dest="test_size", type=int, default=base.test_size)

    # Optimisation
    parser.add_argument("--lr", dest="learning_rate", type=float, default=base.learning_rate)
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=base.weight_decay)
    parser.add_argument(
        "--optimizer", dest="optimizer_type", choices=("adamw", "sgd"), default=base.optimizer_type
    )
    parser.add_argument(
        "--scheduler",
        dest="scheduler_type",
        choices=("linear", "cosine", "constant"),
        default=base.scheduler_type,
    )
    parser.add_argument("--warmup", dest="warmup_ratio", type=float, default=base.warmup_ratio)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=base.batch_size)
    parser.add_argument("--epochs", dest="num_epochs", type=int, default=base.num_epochs)
    parser.add_argument("--seed", type=int, default=base.seed)

    # Exécution
    parser.add_argument("--output-dir", dest="output_dir", default=base.output_dir)
    parser.add_argument("--save-model-dir", dest="save_model_dir", default=base.save_model_dir)
    parser.add_argument(
        "--max-minutes", dest="max_train_minutes", type=float, default=base.max_train_minutes
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Ne pas ajouter de ligne à results/all_experiments.csv",
    )
    parser.add_argument(
        "--num-workers", dest="num_workers", type=int, default=2,
        help="Processus de chargement des données (0 sur Windows si erreur).",
    )

    args = parser.parse_args(argv)
    valeurs = {k: v for k, v in vars(args).items() if k not in ("no_log", "num_workers")}
    config = Config.from_dict(valeurs)
    # On conserve les options non-Config sur l'objet pour la suite du programme.
    config.no_log = args.no_log  # type: ignore[attr-defined]
    config.num_workers = args.num_workers  # type: ignore[attr-defined]
    return config


def main(argv: List[str] | None = None) -> Dict[str, float]:
    """Exécute une expérience complète et retourne les métriques de test."""
    config = parse_args(argv)
    no_log = bool(getattr(config, "no_log", False))
    num_workers = int(getattr(config, "num_workers", 2))

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print("=" * 78)
    print(f"Expérience {config.experiment_id:02d} — {config.name} ({config.owner})")
    print(f"Matériel : {gpu_name()} | device={device} | AMP={use_amp}")
    print(f"Hyperparamètres : {config.resume()}")
    print("=" * 78)

    # --- Données ---------------------------------------------------------
    with Chronometre() as prep:
        dataset, tokenizer = build_datasets(config)
    print(describe_datasets(dataset))
    print(f"Préparation des données : {format_duree(prep.secondes)}")

    train_loader = DataLoader(
        dataset["train"],
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_amp,
    )
    val_loader = DataLoader(
        dataset["validation"],
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_amp,
    )
    test_loader = DataLoader(
        dataset["test"],
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_amp,
    )

    # --- Modèle ----------------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name, num_labels=2
    )
    model.to(device)
    nb_params = count_parameters_millions(model)

    total_steps = len(train_loader) * config.num_epochs
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # --- Entraînement ----------------------------------------------------
    reset_gpu_peak_memory()
    meilleur_etat = None
    meilleur_f1 = -1.0
    meilleure_epoque = 0
    historique: List[Dict[str, float]] = []

    with Chronometre() as entrainement:
        for epoque in range(1, config.num_epochs + 1):
            debut = time.perf_counter()
            perte_train = train_one_epoch(
                model, train_loader, optimizer, scheduler, device, scaler, use_amp
            )
            duree = time.perf_counter() - debut

            perte_val, pred_val, ref_val = evaluate(model, val_loader, device)
            metriques_val = compute_metrics(ref_val, pred_val)

            print(
                f"Époque {epoque}/{config.num_epochs} | "
                f"train_loss={perte_train:.4f} | val_loss={perte_val:.4f} | "
                f"val_f1={metriques_val['f1']:.4f} | {format_duree(duree)}"
            )
            historique.append(
                {
                    "epoch": epoque,
                    "train_loss": round(perte_train, 4),
                    "val_loss": round(perte_val, 4),
                    "val_f1": metriques_val["f1"],
                    "train_time_s": round(duree, 1),
                }
            )

            # On garde le meilleur modèle selon le F1 de validation.
            if metriques_val["f1"] > meilleur_f1:
                meilleur_f1 = metriques_val["f1"]
                meilleure_epoque = epoque
                meilleur_etat = {
                    cle: valeur.detach().cpu().clone()
                    for cle, valeur in model.state_dict().items()
                }

            # Garde-fou : respecter le budget Colab (< 10 min par défaut).
            if entrainement.secondes / 60.0 > config.max_train_minutes:
                print(
                    f"⚠️  Budget de {config.max_train_minutes:g} min dépassé : "
                    "arrêt anticipé de l'entraînement."
                )
                break

    temps_entrainement = entrainement.secondes
    memoire = gpu_memory_mb()

    # --- Évaluation finale sur le test -----------------------------------
    if meilleur_etat is not None:
        model.load_state_dict(meilleur_etat)
    perte_test, pred_test, ref_test = evaluate(model, test_loader, device)
    metriques_test = compute_metrics(ref_test, pred_test)

    print("-" * 78)
    print(
        f"TEST | loss={perte_test:.4f} | accuracy={metriques_test['accuracy']:.4f} | "
        f"precision={metriques_test['precision']:.4f} | "
        f"recall={metriques_test['recall']:.4f} | F1={metriques_test['f1']:.4f}"
    )
    print(
        f"Temps d'entraînement : {format_duree(temps_entrainement)} | "
        f"mémoire GPU : {memoire['allocated']} Mo alloués / {memoire['reserved']} Mo réservés"
    )
    print(f"Meilleure époque (val F1) : {meilleure_epoque} (F1={meilleur_f1:.4f})")
    print("-" * 78)

    # --- Sauvegarde des résultats ----------------------------------------
    dossier = RESULTS_DIR / config.owner.lower() / config.tag()
    dossier.mkdir(parents=True, exist_ok=True)

    rapport = classification_report_text(ref_test, pred_test)
    (dossier / "classification_report.txt").write_text(rapport, encoding="utf-8")

    resultats = {
        "experiment_id": config.experiment_id,
        "owner": config.owner,
        "name": config.name,
        "train_time_s": round(temps_entrainement, 1),
        "train_loss": historique[-1]["train_loss"] if historique else None,
        "val_loss": historique[-1]["val_loss"] if historique else None,
        "test_loss": round(perte_test, 4),
        "accuracy": metriques_test["accuracy"],
        "precision": metriques_test["precision"],
        "recall": metriques_test["recall"],
        "f1": metriques_test["f1"],
        "best_epoch": meilleure_epoque,
        "best_val_f1": round(meilleur_f1, 4),
        "gpu_mem_alloc_mb": memoire["allocated"],
        "gpu_mem_reserved_mb": memoire["reserved"],
        "num_params_m": nb_params,
        "device": str(device),
        "gpu": gpu_name(),
        "history": historique,
        "confusion_matrix": confusion_matrix_values(ref_test, pred_test),
        "config": config.to_dict(),
    }
    write_json(dossier / "metrics.json", resultats)
    write_json(dossier / "run_config.json", config.to_dict())

    # --- Journal cumulé ---------------------------------------------------
    if not no_log:
        ligne = {
            "id": config.experiment_id,
            "owner": config.owner,
            "name": config.name,
            "learning_rate": config.learning_rate,
            "warmup_ratio": config.warmup_ratio,
            "scheduler": config.scheduler_type,
            "batch_size": config.batch_size,
            "optimizer": config.optimizer_type,
            "train_time_s": resultats["train_time_s"],
            "gpu_mem_alloc_mb": memoire["allocated"],
            "gpu_mem_reserved_mb": memoire["reserved"],
            "train_loss": resultats["train_loss"],
            "val_loss": resultats["val_loss"],
            "accuracy": metriques_test["accuracy"],
            "precision": metriques_test["precision"],
            "recall": metriques_test["recall"],
            "f1": metriques_test["f1"],
            "num_params_m": nb_params,
            "observations": "",
        }
        chemin_csv = log_experiment(ligne, field_order=EXPERIMENT_FIELDS)
        print(f"Journal mis à jour : {chemin_csv}")

    print(f"Résultats écrits dans : {dossier}")

    # --- Sauvegarde optionnelle du modèle (Google Drive) ------------------
    if config.save_model_dir:
        cible = Path(config.save_model_dir) / config.tag()
        cible.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(cible)
        tokenizer.save_pretrained(cible)
        print(f"Modèle sauvegardé hors dépôt : {cible}")

    return metriques_test


if __name__ == "__main__":
    main()
