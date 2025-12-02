# Projet d'Attaques et de Défenses en Apprentissage Fédéré

Une implémentation complète d'attaques et de défenses en apprentissage fédéré utilisant le framework Flower.

## 📁 Structure du Projet

### Implémentation de Base
- `client.py` - Client d'apprentissage fédéré normal
- `client_mal.py` - Client malveillant avec attaques implémentées
- `server.py` - Serveur FedAvg de base
- `prepare_dataset.py` - Chargement et distribution des données CIFAR-10

### Attaque 1 : Inversion d'Étiquettes
- `serveur_attack1.py` - Serveur Attack1 avec journalisation CSV
- `serveur_attack1_defense.py` - Serveur Attack1 avec mécanismes de défense
- `plot_attack1.py` - Script de visualisation pour les résultats attack1
- `plot_defense_attack1.py` - Visualisation pour l'efficacité des défenses
- `run_attack1_with_defenses.sh` - Script d'expérimentation automatisée

### Attaque 2 : Empoisonnement de Modèle 
- `serveur_attack2.py` - Implémentation du serveur Attack2
- `serveur_attack2_defense.py` - Serveur Attack2 avec défenses
- `plot_attack2.py` - Visualisation pour les résultats attack2
- `run_attack2_with_defenses.sh` - Script d'expérimentation automatisée

### Résultats Expérimentaux
- `results_attack1/` - Résultats de l'attaque d'inversion d'étiquettes (40 expériences)
- `results_attack1_median/` - Résultats de défense utilisant FedMedian
- `results_attack1_trimmed/` - Résultats de défense utilisant FedTrimmedAvg
- `results_attack2/` - Résultats de l'attaque d'empoisonnement de modèle

### Sorties de Visualisation
- `attack1_results.png` - Comparaison de l'efficacité d'Attack1
- `attack1_median_iid_vs_non_iid.png` - Résultats de défense FedMedian
- `attack1_trimmed_iid_vs_non_iid.png` - Résultats de défense FedTrimmedAvg
- `attack2_model_poisoning_results.png` - Efficacité d'Attack2


## 🎯 Fonctionnalités Implémentées

### Attaques
- **Inversion d'Étiquettes** : Décalage des étiquettes de +1 (50% de probabilité par round)
- **Empoisonnement de Modèle** : Montée de gradient avec injection de bruit

### Défenses
- **FedMedian** : Agrégation par médiane coordonnée par coordonnée
- **FedTrimmedAvg** : Agrégation par moyenne tronquée (supprime les valeurs extrêmes)

### Configuration Expérimentale
- **Clients** : 5 au total (0-3 malveillants)
- **Distributions de Données** : IID et Non-IID
- **Répétitions** : Plusieurs exécutions par configuration
- **Rounds** : 20 rounds d'entraînement par expérience

---
*Projet terminé : Décembre 2025*

