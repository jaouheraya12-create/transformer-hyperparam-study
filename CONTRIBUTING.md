# Guide de contribution

Ce document décrit **comment travailler ensemble** sur le projet sans casser les
résultats des autres. Il est court : lis-le avant ton premier commit.

---

## 1. Branches

`main` est **protégée** : personne ne pousse directement dessus.
Chaque étudiant travaille sur **sa** branche, créée depuis `main`.

| Membre | Branche | Périmètre |
|---|---|---|
| Nada | `nada/baseline` | dataset, nettoyage, baseline, learning rate |
| Fatima | `fatima/warmup-scheduler` | warmup et scheduler |
| Aya | `aya/batch-optimizer` | batch size, optimizer, temps et mémoire GPU |
| Oussama | `oussama/evaluation` | protocole d'évaluation, interactions, ablation, analyse d'erreurs, centralisation |

Créer sa branche (une seule fois) :

```bash
git switch main
git pull origin main
git switch -c nada/baseline        # remplacer par ta branche
git push -u origin nada/baseline
```

---

## 2. Cycle de travail

```text
main ──► ma branche ──► commits ──► push ──► Pull Request ──► revue ──► merge dans main
```

1. **Synchroniser** avant de commencer : `git pull origin main` (puis `git merge main`
   dans sa branche si `main` a avancé).
2. **Travailler** uniquement dans son périmètre d'expériences.
3. **Commiter** avec un message clair (voir §6).
4. **Pousser** sa branche.
5. **Ouvrir une Pull Request** vers `main` en utilisant le template
   [`.github/pull_request_template.md`](.github/pull_request_template.md).
6. **Faire relire** par un autre membre (au moins 1 approbation).
7. **Fusionner** (squash merge de préférence) une fois la CI/les vérifications passées.

> Une PR = **une expérience** (ou un petit lot cohérent). Éviter les PR de 2 000 lignes.

---

## 3. Règles d'expérimentation (non négociables)

| Règle | Détail |
|---|---|
| **Partir de `src/config.py`** | La dataclass `Config` est l'unique référence. Ne recopie pas les valeurs dans un notebook. |
| **Un seul hyperparamètre à la fois** | Par rapport à la baseline, une expérience ne change **qu'un** champ. Exceptions documentées : exp. 13 (interaction) et exp. 15 (meilleure combinaison). |
| **Même seed** | `seed = 42` partout. Ne jamais changer la seed pour « améliorer » un score. |
| **Mêmes splits** | 4 000 / 1 000 / 1 000, découpe déterministe de `src/data.py`. |
| **< 10 minutes** | Par entraînement, sur Colab GPU T4. Si ça dépasse : réduire la taille des données, pas la qualité du protocole. |
| **15 expériences maximum** | Au-delà, l'expérience doit être arbitrée par l'équipe et remplacée, pas ajoutée. |
| **Résultats dans `results/<étudiant>/`** | Un sous-dossier par expérience : `exp_XX_<nom>/`. |
| **Pas de modèles sur GitHub** | Poids (`.pt`, `.bin`, `.safetensors`), checkpoints et logs lourds → **Google Drive**. |

### Ce qu'il ne faut PAS modifier sans accord de l'équipe

- `src/config.py` : valeurs par défaut de la baseline (seed, tailles, epochs, `max_length`).
- `src/data.py` : logique de nettoyage et de splits (casse la comparabilité).
- `src/metrics.py` : définition des métriques.
- `experiments/experiments_plan.csv` : les lignes des autres membres.

---

## 4. Résultats : où et comment

```
results/
└── nada/
    └── exp_01_baseline/
        ├── run_config.json              # configuration exacte du run (généré)
        ├── metrics.json                 # accuracy, precision, recall, f1, temps, mémoire
        └── classification_report.txt    # rapport détaillé par classe
results/all_experiments.csv              # journal cumulé (généré par log_experiment)
```

- Le journal `results/all_experiments.csv` est **partagé** : ne pas le réécrire à la main,
  il est complété par `src/utils.py::log_experiment`.
- Remplir la ligne correspondante de `experiments/experiments_plan.csv`
  (colonnes `train_time_s`, `train_loss`, `val_loss`, `accuracy`, `f1`, `observations`).
- Les figures vont dans `results/<étudiant>/figures/` (PNG, < 500 Ko).
- Les modèles entraînés vont sur **Google Drive**, jamais dans le dépôt.

---

## 5. Style de code

- **Commentaires et docstrings en français**, code et noms de variables en anglais.
- Respecter la structure existante (`src/` = code réutilisable, `notebooks/` = démonstration).
- Type hints quand c'est utile ; pas de dépendance nouvelle sans discussion préalable.
- Pas de code mort, pas de cellule de notebook qui ne s'exécute pas de haut en bas.
- Avant de pousser : `python -c "import src.config"` doit fonctionner.

---

## 6. Messages de commit

Format court et explicite, en français à l'impératif :

```text
exp 3 : learning rate 5e-5 (F1 0.842)
data : supprime les doublons après nettoyage HTML
docs : ajoute le protocole d'évaluation
fix : corrige le calcul du warmup en steps
```

Éviter : `update`, `fix`, `modifs`, `test`, `final`, `final2`.

---

## 7. Sécurité

- **Aucun secret** dans le dépôt : token GitHub, clé W&B, identifiants Colab, `.env`.
- Si tu utilises W&B, exporte la clé dans l'environnement Colab, jamais dans un fichier versionné.
- Ne pas coller de token dans une cellule de notebook qui sera poussée.

---

## 8. Checklist avant d'ouvrir une Pull Request

- [ ] Une seule expérience (ou un lot cohérent) dans la PR
- [ ] Ligne ajoutée/mise à jour dans `experiments/experiments_plan.csv`
- [ ] Résultats présents dans `results/<mon_pseudo>/exp_XX_<nom>/`
- [ ] Seed = 42, mêmes splits, même `src/config.py`
- [ ] Un seul paramètre modifié par rapport à la baseline (hors interactions documentées)
- [ ] Entraînement < 10 minutes (temps reporté dans le CSV)
- [ ] Aucun modèle ni secret dans le diff
- [ ] Le notebook s'exécute de bout en bout
- [ ] Un autre membre a relu

---

## 9. En cas de blocage

1. Vérifier que `main` est bien récupéré (`git pull origin main`).
2. Vérifier l'environnement : `python -c "import torch, transformers, datasets, sklearn"`.
3. Vérifier le GPU : `nvidia-smi`.
4. Poser la question dans le canal d'équipe **avant** de modifier le protocole.
