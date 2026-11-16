"""Chargement, nettoyage et tokenisation du jeu de données IMDB.

Pipeline :

1. Téléchargement du dataset Hugging Face (``imdb``) ;
2. conversion en DataFrame Pandas ;
3. nettoyage : balises HTML, espaces multiples, caractères de contrôle,
   valeurs manquantes, doublons ;
4. découpe déterministe Train (4 000) / Validation (1 000) / Test (1 000) ;
5. tokenisation (``max_length=128``) et conversion en tenseurs PyTorch.

Toutes les tailles et la seed proviennent de :class:`src.config.Config`.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .config import Config

# ---------------------------------------------------------------------------
# Expressions régulières de nettoyage
# ---------------------------------------------------------------------------
BALISE_HTML = re.compile(r"<[^>]+>")  # <br />, <b>, etc.
ESPACES_MULTIPLES = re.compile(r"\s+")
CARACTERES_CONTROLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
ECHAPPEMENTS_HTML = {
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&quot;": '"',
    "&#39;": "'",
    "&apos;": "'",
    "&nbsp;": " ",
}


def clean_text(texte: str) -> str:
    """Nettoie un commentaire IMDB.

    - supprime les balises HTML ;
    - décode les entités HTML les plus courantes ;
    - supprime les caractères de contrôle ;
    - normalise les espaces (y compris retours à la ligne et tabulations).
    """
    if not isinstance(texte, str):
        return ""
    texte = BALISE_HTML.sub(" ", texte)
    for entite, remplacement in ECHAPPEMENTS_HTML.items():
        texte = texte.replace(entite, remplacement)
    texte = CARACTERES_CONTROLE.sub(" ", texte)
    return ESPACES_MULTIPLES.sub(" ", texte).strip()


def load_raw_dataframe(config: Config) -> pd.DataFrame:
    """Charge les splits ``train`` et ``test`` d'IMDB dans un DataFrame unique.

    On ajoute une colonne ``origine`` pour pouvoir construire un test propre
    à partir du split officiel, sans mélanger avec les données d'entraînement.
    """
    dataset = load_dataset(config.dataset_name)
    morceaux = []
    for split in ("train", "test"):
        if split not in dataset:
            continue
        df = dataset[split].to_pandas()
        df["origine"] = split
        morceaux.append(df)
    if not morceaux:
        raise ValueError(f"Dataset {config.dataset_name!r} vide ou introuvable.")
    return pd.concat(morceaux, ignore_index=True)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie le DataFrame : texte, NaN, doublons.

    Étapes :
        1. suppression des lignes sans texte ou sans étiquette ;
        2. nettoyage du texte (HTML, espaces, caractères de contrôle) ;
        3. suppression des textes vides après nettoyage ;
        4. suppression des doublons exacts (même texte + même étiquette) ;
        5. conversion de l'étiquette en entier 0/1.
    """
    df = df.copy()
    df = df[["text", "label", "origine"]]
    df = df.dropna(subset=["text", "label"])

    df["text"] = df["text"].astype(str).map(clean_text)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df = df[df["label"].isin([0, 1])]
    df["label"] = df["label"].astype(int)

    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    return df


def _echantillonner(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Prend ``n`` lignes au hasard (seed fixe), sans dépasser la taille dispo."""
    n = min(n, len(df))
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def split_dataframe(df: pd.DataFrame, config: Config) -> DatasetDict:
    """Construit les splits Train / Validation / Test de façon déterministe.

    - Train et Validation proviennent du split officiel ``train`` ;
    - Test provient du split officiel ``test`` (jamais vu à l'entraînement) ;
    - les textes du test déjà présents dans le pool d'entraînement sont retirés
      pour limiter les fuites de données (doublons inter-splits).
    """
    pool_train = df[df["origine"] == "train"].reset_index(drop=True)
    pool_test = df[df["origine"] == "test"].reset_index(drop=True)

    pool_train = pool_train.sample(frac=1.0, random_state=config.seed).reset_index(drop=True)

    besoin = config.train_size + config.val_size
    if len(pool_train) < besoin:
        raise ValueError(
            f"Pas assez de données d'entraînement : {len(pool_train)} lignes "
            f"pour {besoin} demandées."
        )

    train_df = pool_train.iloc[: config.train_size].reset_index(drop=True)
    val_df = pool_train.iloc[config.train_size : besoin].reset_index(drop=True)

    # Anti-fuite : on retire du test les textes déjà vus dans train/val.
    textes_connus = set(train_df["text"]) | set(val_df["text"])
    pool_test = pool_test[~pool_test["text"].isin(textes_connus)].reset_index(drop=True)
    test_df = _echantillonner(pool_test, config.test_size, config.seed)

    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False),
            "validation": Dataset.from_pandas(val_df[["text", "label"]], preserve_index=False),
            "test": Dataset.from_pandas(test_df[["text", "label"]], preserve_index=False),
        }
    )


def tokenize_datasets(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    config: Config,
) -> DatasetDict:
    """Tokenise les trois splits et prépare les colonnes attendues par le modèle.

    On utilise ``padding="max_length"`` : toutes les séquences ont la même
    longueur, ce qui simplifie la mesure du temps et de la mémoire GPU.
    La colonne ``label`` est renommée en ``labels`` (nom attendu par
    ``AutoModelForSequenceClassification``).
    """

    def _tokeniser(lot: dict) -> dict:
        return tokenizer(
            lot["text"],
            truncation=True,
            padding="max_length",
            max_length=config.max_length,
        )

    dataset = dataset.map(
        _tokeniser,
        batched=True,
        remove_columns=["text"],
        desc="Tokenisation",
    )
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset


def build_datasets(
    config: Config,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> Tuple[DatasetDict, PreTrainedTokenizerBase]:
    """Pipeline complet : chargement → nettoyage → splits → tokenisation.

    Retourne ``(dataset_tokenise, tokenizer)``.
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    brut = load_raw_dataframe(config)
    propre = clean_dataframe(brut)
    decoupe = split_dataframe(propre, config)
    tokenise = tokenize_datasets(decoupe, tokenizer, config)
    return tokenise, tokenizer


def describe_datasets(dataset: DatasetDict) -> str:
    """Résumé lisible des splits (utile dans les notebooks)."""
    lignes = [f"{nom:>10} : {len(split):>5} exemples" for nom, split in dataset.items()]
    return "\n".join(lignes)


__all__ = [
    "clean_text",
    "clean_dataframe",
    "load_raw_dataframe",
    "split_dataframe",
    "tokenize_datasets",
    "build_datasets",
    "describe_datasets",
]
