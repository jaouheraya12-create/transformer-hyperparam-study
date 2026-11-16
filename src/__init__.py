"""Paquet `src` du projet d'étude des hyperparamètres de DistilBERT.

Modules :
    - ``config``  : dataclass ``Config`` (configuration de référence)
    - ``data``    : chargement / nettoyage / tokenisation du jeu IMDB
    - ``train``   : boucle d'entraînement générique (CLI)
    - ``metrics`` : accuracy, precision, recall, F1
    - ``utils``   : seed, mémoire GPU, journalisation des expériences
"""

__version__ = "0.1.0"

__all__ = ["config", "data", "metrics", "utils", "train"]
