# Étude des hyperparamètres d'un petit Transformer (DistilBERT)

Projet universitaire d'équipe (4 étudiants) : mesurer **l'influence des hyperparamètres**
sur l'entraînement d'un petit Transformer de classification de texte.

- **Modèle** : `distilbert-base-uncased` (66 M paramètres)
- **Tâche** : classification binaire de sentiments sur **IMDB**
- **Budget** : **15 entraînements maximum**, **< 10 minutes chacun**, sur **Google Colab (GPU T4)**
- **Principe directeur** : une expérience = **un seul hyperparamètre modifié** par rapport à la baseline

---

## 1. Présentation

L'objectif est de comprendre, de façon **empirique et reproductible**, l'effet de cinq
hyperparamètres sur la qualité du modèle (F1, accuracy) et sur le coût de l'entraînement
(temps, mémoire GPU) :

| Hyperparamètre | Rôle étudié |
|---|---|
| Learning Rate | vitesse d'apprentissage et stabilité |
| Warmup | montée progressive du LR en début d'entraînement |
| Scheduler | forme de la décroissance du LR (linear / cosine / constant) |
| Batch Size | compromis vitesse / mémoire / généralisation |
| Optimizer | AdamW vs SGD |

Contrainte forte : **rester dans le budget Colab**. Un run doit tenir en moins de
10 minutes, ce qui impose des sous-ensembles de données réduits (4 000 / 1 000 / 1 000)
et 2 époques.

---

## 2. Hyperparamètres de référence

La configuration de référence est définie **une seule fois** dans [`src/config.py`](src/config.py)
(dataclass `Config`). Toutes les expériences en héritent.

| Paramètre | Valeur baseline | Valeurs testées | Expériences |
|---|---|---|---|
| `model_name` | `distilbert-base-uncased` | — | toutes |
| `dataset_name` | `imdb` | — | toutes |
| `max_length` | 128 | — | toutes |
| `train_size` | 4000 | — | toutes |
| `val_size` | 1000 | — | toutes |
| `test_size` | 1000 | — | toutes |
| `learning_rate` | `2e-5` | `1e-5`, `5e-5`, `1e-4` | 2, 3, 4 |
| `warmup_ratio` | `0.0` | `0.1`, `0.2` | 5, 7, 8 |
| `scheduler_type` | `linear` | `cosine`, `constant` | 6, 7, 14 |
| `batch_size` | `32` | `16`, `64` | 9, 10 |
| `optimizer_type` | `adamw` | `sgd` (lr `2e-5` et `1e-2`) | 11, 12 |
| `num_epochs` | `2` | — | toutes |
| `weight_decay` | `0.01` | — | toutes |
| `seed` | `42` | — | toutes |

> Toute modification d'un de ces champs doit être justifiée par une ligne du
> [`experiments/experiments_plan.csv`](experiments/experiments_plan.csv).

### Métriques suivies

| Métrique | Source | Fichier |
|---|---|---|
| Accuracy, Precision, Recall, F1 | scikit-learn | `results/<étudiant>/exp_XX/` |
| `train_loss`, `val_loss` | boucle d'entraînement | `results/all_experiments.csv` |
| `train_time_s` | `time.perf_counter` | `results/all_experiments.csv` |
| Mémoire GPU (pic allouée / réservée) | `torch.cuda.max_memory_allocated` | `results/all_experiments.csv` |

---

## 3. Équipe

| Membre | Rôle | Expériences | Notebook |
|---|---|---|---|
| **Nada** | Dataset, nettoyage, baseline, learning rate | 4 (1 → 4) | [`01_baseline_lr.ipynb`](notebooks/01_baseline_lr.ipynb) |
| **Fatima** | Warmup et scheduler | 4 (5 → 8) | [`02_warmup_scheduler.ipynb`](notebooks/02_warmup_scheduler.ipynb) |
| **Aya** | Batch size, optimizer, temps et mémoire GPU | 4 (9 → 12) | [`03_batchsize_optimizer.ipynb`](notebooks/03_batchsize_optimizer.ipynb) |
| **Oussama** | Protocole d'évaluation, interactions, ablation, analyse d'erreurs, centralisation | 3 (13 → 15) | [`04_evaluation_analysis.ipynb`](notebooks/04_evaluation_analysis.ipynb) |

---

## 4. Arborescence

```
transformer-hyperparam-study/
├── README.md
├── CONTRIBUTING.md
├── requirements.txt
├── .gitignore
├── .github/
│   └── pull_request_template.md
├── src/
│   ├── __init__.py
│   ├── config.py       # dataclass Config : LA référence, point de départ de tout run
│   ├── data.py         # chargement IMDB, nettoyage Pandas, tokenisation, splits
│   ├── train.py        # boucle d'entraînement générique + CLI
│   ├── metrics.py      # accuracy / precision / recall / F1
│   └── utils.py        # set_seed, gpu_memory_mb, log_experiment
├── notebooks/
│   ├── 01_baseline_lr.ipynb
│   ├── 02_warmup_scheduler.ipynb
│   ├── 03_batchsize_optimizer.ipynb
│   └── 04_evaluation_analysis.ipynb
├── experiments/
│   └── experiments_plan.csv    # plan des 15 expériences (source de vérité)
├── results/
│   ├── nada/
│   ├── fatima/
│   ├── aya/
│   └── oussama/
└── docs/
    └── workflow.md
```

---

## 5. Démarrage rapide (Google Colab)

1. Ouvrir [Google Colab](https://colab.research.google.com/) → **Nouveau notebook**.
2. **Exécution → Modifier le type d'exécution → T4 GPU**.
3. Coller et exécuter :

```python
# 1) Récupérer le dépôt (idempotent : ne reclone pas si déjà présent)
!git clone https://github.com/jaouheraya12-create/transformer-hyperparam-study.git
%cd transformer-hyperparam-study

# 2) Installer les dépendances (2 à 3 minutes sur Colab)
!pip install -q -r requirements.txt

# 3) Vérifier le GPU
!nvidia-smi

# 4) Lancer une expérience (exemple : baseline, expérience 1)
!python -m src.train --id 1 --owner Nada --name baseline
```

> ⚠️ `pip install` sur Colab peut déjà fournir `torch` et `transformers` : le
> `requirements.txt` utilise des bornes minimales (`>=`) pour éviter de tout réinstaller.

### Exemples de commandes par expérience

```bash
# Nada — learning rate (expériences 2 à 4)
python -m src.train --id 2 --owner Nada --name lr_1e-5  --lr 1e-5
python -m src.train --id 3 --owner Nada --name lr_5e-5  --lr 5e-5
python -m src.train --id 4 --owner Nada --name lr_1e-4  --lr 1e-4

# Fatima — warmup et scheduler (expériences 5 à 8)
python -m src.train --id 5 --owner Fatima --name warmup_0.1_linear --warmup 0.1
python -m src.train --id 6 --owner Fatima --name cosine_sans_warmup --scheduler cosine
python -m src.train --id 7 --owner Fatima --name warmup_0.1_cosine --warmup 0.1 --scheduler cosine
python -m src.train --id 8 --owner Fatima --name warmup_0.2_linear --warmup 0.2

# Aya — batch size et optimizer (expériences 9 à 12)
python -m src.train --id 9  --owner Aya --name batch_16     --batch-size 16
python -m src.train --id 10 --owner Aya --name batch_64     --batch-size 64
python -m src.train --id 11 --owner Aya --name sgd_lr_2e-5  --optimizer sgd
python -m src.train --id 12 --owner Aya --name sgd_lr_1e-2  --optimizer sgd --lr 1e-2

# Oussama — interactions, ablation, meilleure combinaison (expériences 13 à 15)
python -m src.train --id 13 --owner Oussama --name interaction_lr5e-5_warmup0.1_cosine_batch64 \
  --lr 5e-5 --warmup 0.1 --scheduler cosine --batch-size 64
python -m src.train --id 14 --owner Oussama --name ablation_sans_scheduler --scheduler constant
# expérience 15 : à définir après analyse des expériences 1 à 14
```

### Sauvegarder le modèle sur Google Drive (jamais sur GitHub)

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
python -m src.train --id 1 --owner Nada --name baseline \
  --save-model-dir /content/drive/MyDrive/hyperparam-study/models
```

---

## 6. Ordre de réalisation

| # | Étape | Responsable | Dépend de |
|---|---|---|---|
| 1 | Mise en place du dépôt, structure, environnement | Oussama + Nada | — |
| 2 | Dataset, nettoyage, baseline (exp. 1) | Nada | 1 |
| 3 | Learning rate (exp. 2 → 4) | Nada | 2 |
| 4 | Warmup et scheduler (exp. 5 → 8) | Fatima | 2 |
| 5 | Batch size et optimizer (exp. 9 → 12) | Aya | 2 |
| 6 | Interactions, ablation, meilleure combinaison (exp. 13 → 15) | Oussama | 3, 4, 5 |
| 7 | Consolidation, figures, rapport | Tous (pilote : Oussama) | 6 |

Détail complet et graphe de dépendances : [`docs/workflow.md`](docs/workflow.md).

> La baseline (exp. 1) est **bloquante** : aucune autre expérience ne doit être
> interprétée avant que ses métriques soient figées dans `experiments_plan.csv`.

---

## 7. Règles de collaboration

1. **Une seule source de configuration** : `src/config.py`. On ne duplique jamais
   les valeurs par défaut dans un notebook ou un script.
2. **Un seul hyperparamètre modifié** par expérience par rapport à la baseline.
   Les interactions (exp. 13 et 15) sont l'exception explicite et documentée.
3. **Même seed (42), mêmes splits, mêmes tailles de sous-ensembles** pour tous.
4. **< 10 minutes par entraînement** : sinon réduire la taille des données, pas le seed.
5. **15 expériences maximum** : toute expérience supplémentaire doit être arbitrée par l'équipe.
6. **Résultats versionnés** dans `results/<étudiant>/` (JSON + rapport de classification).
   Les métriques agrégées vont dans `results/all_experiments.csv`.
7. **Aucun modèle sur GitHub** (`.pt`, `.bin`, `.safetensors`, checkpoints) → Google Drive.
8. **Aucun secret** : pas de token, pas de clé W&B, pas de `.env`.
9. **Une branche par étudiant**, fusion uniquement par **Pull Request** vers `main`
   (voir [`CONTRIBUTING.md`](CONTRIBUTING.md)).

---

## 8. Résultats

| Fichier | Contenu |
|---|---|
| `experiments/experiments_plan.csv` | **plan** des 15 expériences + métriques à remplir |
| `results/all_experiments.csv` | **journal** automatique, une ligne par run terminé |
| `results/<étudiant>/exp_XX_<nom>/metrics.json` | métriques détaillées du run |
| `results/<étudiant>/exp_XX_<nom>/run_config.json` | configuration exacte utilisée |
| `results/<étudiant>/exp_XX_<nom>/classification_report.txt` | rapport par classe |

Les conclusions (meilleurs réglages, compromis temps/mémoire) sont rédigées dans le
rapport final et résumées dans la colonne `observations` du plan d'expériences.

---

## 9. Reproductibilité

- Seed fixée à **42** (`set_seed` : `random`, `numpy`, `torch`, `cuda`).
- Splits déterministes : mélange avec la seed puis découpe 4 000 / 1 000 / 1 000.
- Chaque run écrit sa configuration complète (`run_config.json`).
- Version matérielle : Google Colab, GPU T4 (16 Go). Le temps et la mémoire GPU
  ne sont donc comparables qu'entre runs exécutés sur le même type de machine.

---

## Licence et usage

Projet pédagogique universitaire. Le code est fourni à des fins d'apprentissage ;
les résultats ne doivent pas être présentés comme un benchmark de l'état de l'art.
