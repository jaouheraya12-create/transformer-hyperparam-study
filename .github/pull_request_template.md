<!--
Merci de remplir cette Pull Request. Elle doit concerner UNE expérience
(ou un petit lot cohérent). Les cases non cochées doivent être justifiées.
-->

## Expérience

| Champ | Valeur |
|---|---|
| Expérience(s) | `exp_XX` |
| Membre | Nada / Fatima / Aya / Oussama |
| Notebook | `notebooks/0X_....ipynb` |
| Hyperparamètre modifié | ... |
| Baseline de comparaison | exp. 1 |

## Résumé

<!-- Décrire en 2-3 phrases : ce qui change, ce qui est observé, ce qui est conclu. -->

-

## Métriques

| Métrique | Baseline (exp. 1) | Cette expérience | Écart |
|---|---|---|---|
| Accuracy | | | |
| Precision | | | |
| Recall | | | |
| F1 | | | |
| Temps d'entraînement (s) | | | |
| Mémoire GPU (Mo) | | | |

## Checklist

- [ ] L'expérience est ajoutée / mise à jour dans `experiments/experiments_plan.csv`
- [ ] Les résultats sont présents dans `results/<étudiant>/exp_XX_<nom>/`
      (`metrics.json`, `run_config.json`, `classification_report.txt`)
- [ ] La seed est **42** (aucune modification pour « améliorer » le score)
- [ ] **Un seul paramètre** est modifié par rapport à la baseline
      (exceptions documentées : exp. 13 et 15)
- [ ] Les splits sont identiques (4 000 / 1 000 / 1 000, `src/data.py` inchangé)
- [ ] L'entraînement dure **moins de 10 minutes** (temps reporté ci-dessus)
- [ ] Le notebook s'exécute de bout en bout sur Colab GPU
- [ ] Aucun modèle (`.pt`, `.bin`, `.safetensors`, checkpoint) n'est versionné
- [ ] Aucun secret (token, clé W&B, `.env`) n'est présent dans le diff
- [ ] La PR a été relue par un autre membre

## Notes pour la revue

<!-- Points d'attention, doutes, résultats inattendus. -->

-

## Références

<!-- Issue liée, message d'équipe, notebook Colab partagé. -->

-
