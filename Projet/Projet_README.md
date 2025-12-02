# Projet d'Attaques et de Défenses en Apprentissage Fédéré

Une implémentation complète d'attaques et de défenses en apprentissage fédéré utilisant le framework Flower.

## 🚀 Installation

### Prérequis
- Python 3.10+
- Conda (recommandé)

### Étapes d'installation

1. **Créer l'environnement Conda** :
```bash
conda env create -f environment.yml
conda activate fl-miage
```

2. **Installer Flower** (si nécessaire) :
```bash
python -m pip install flwr
```

3. **Télécharger le dataset CIFAR-10** :
Le dataset sera automatiquement téléchargé lors de la première exécution dans le dossier `dataset/`.

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

## 📖 Guide d'Utilisation

### Exécution Manuelle

#### 1. Test Rapide (Basique)
```bash
# Terminal 1: Démarrer le serveur
python server.py --round 10

# Terminal 2: Client normal
python client.py --node_id 0

# Terminal 3: Client malveillant
python client_mal.py --node_id 1 --attack_type label_flipping
```

#### 2. Attaque 1 : Inversion d'Étiquettes

**Serveur** :
```bash
python serveur_attack1.py \
    --round 20 \
    --data_split iid \
    --attack_type label_flipping \
    --n_mal 1 \
    --run_id 0
```

**Clients** (dans des terminaux séparés) :
```bash
# Client malveillant (1 instance)
python client_mal.py --node_id 0 --data_split iid --attack_type label_flipping

# Clients normaux (4 instances)
python client.py --node_id 1 --data_split iid
python client.py --node_id 2 --data_split iid
python client.py --node_id 3 --data_split iid
python client.py --node_id 4 --data_split iid
```

#### 3. Attaque 2 : Empoisonnement de Modèle

**Serveur** :
```bash
python serveur_attack2.py \
    --round 20 \
    --data_split iid \
    --attack_type model_poisoning \
    --n_mal 2 \
    --run_id 0
```

**Clients** :
```bash
# Clients malveillants
python client_mal.py --node_id 0 --data_split iid --attack_type model_poisoning
python client_mal.py --node_id 1 --data_split iid --attack_type model_poisoning

# Clients normaux
python client.py --node_id 2 --data_split iid
python client.py --node_id 3 --data_split iid
python client.py --node_id 4 --data_split iid
```

#### 4. Expériences avec Défenses

**Attaque 1 + Défense** :
```bash
python serveur_attack1_defense.py \
    --round 20 \
    --data_split iid \
    --attack_type label_flipping \
    --defense median \
    --n_mal 1 \
    --run_id 0
```

**Attaque 2 + Défense** :
```bash
python serveur_attack2_defense.py \
    --round 20 \
    --data_split iid \
    --attack_type model_poisoning \
    --defense trimmed \
    --n_mal 2 \
    --run_id 0
```

### Exécution Automatisée

#### Scripts Disponibles

1. **Expériences Complètes Attack1** :
```bash
bash run_remaining_experiments.sh
# Exécute toutes les configurations Attack1 (40 expériences)
```

2. **Expériences Attack1 avec Défenses** :
```bash
bash run_attack1_with_defenses.sh
# Exécute Attack1 avec FedMedian et FedTrimmedAvg (16 expériences)
```

3. **Expériences Complètes Attack2** :
```bash
bash run_remaining_experiments_attack2.sh
# Exécute toutes les configurations Attack2 (40 expériences)
```

4. **Expériences Attack2 avec Défenses** :
```bash
bash run_attack2_with_defenses.sh
# Exécute Attack2 avec FedMedian et FedTrimmedAvg (16 expériences)
```

5. **Test Rapide** :
```bash
bash test_single_experiment.sh        # Test Attack1
bash test_single_experiment_attack2.sh # Test Attack2
```

### Paramètres Principaux

| Paramètre | Description | Valeurs Possibles |
|-----------|-------------|-------------------|
| `--round` | Nombre de rounds d'entraînement | Entier (ex: 10, 20) |
| `--data_split` | Type de distribution des données | `iid`, `non_iid_class` |
| `--attack_type` | Type d'attaque | `label_flipping`, `model_poisoning` |
| `--n_mal` | Nombre de clients malveillants | 0, 1, 2, 3 |
| `--run_id` | ID de répétition | 0, 1, 2, 3, 4 |
| `--defense` | Stratégie de défense | `none`, `median`, `trimmed` |
| `--node_id` | ID du client | 0, 1, 2, 3, 4 |

## 📊 Génération des Visualisations

### Visualiser les Résultats Attack1
```bash
python plot_attack1.py
# Génère: attack1_results.png
```

### Visualiser les Résultats Attack2
```bash
python plot_attack2.py
# Génère: attack2_model_poisoning_results.png
```

### Visualiser les Défenses Attack1
```bash
python plot_defense_attack1.py
# Génère: attack1_median_iid_vs_non_iid.png et attack1_trimmed_iid_vs_non_iid.png
```

### Visualiser les Défenses Attack2
```bash
python plot_attack2_defense.py
# Génère: attack2_median_results.png et attack2_trimmed_results.png
```

## 📈 Format des Résultats

Les résultats sont sauvegardés en CSV avec les colonnes suivantes :
- `round` : Numéro du round (1-N)
- `accuracy` : Précision du modèle à ce round
- `loss` : Perte du modèle à ce round

**Format des noms de fichiers** :
- Attack1 : `label_flipping_{data_split}_mal{n_mal}_run{run_id}.csv`
- Attack1 + Défense : `label_flipping_{defense}_{data_split}_mal{n_mal}_run{run_id}.csv`
- Attack2 : `model_poisoning_{data_split}_mal{n_mal}_run{run_id}.csv`

## 🔍 Tests et Validation

### Test de l'Attaque Label Flipping
```bash
python test_attack.py
```

### Notebooks de Test des Défenses
- `test.fedmedian.ipynb` : Test de FedMedian
- `test.xxfed.ipynb` : Test de FedTrimmedAvg

## ⚠️ Notes Importantes

1. **Ordre d'exécution** : Toujours démarrer le serveur avant les clients
2. **Temps d'exécution** : Chaque expérience prend environ 10-15 minutes
3. **Ports** : Le serveur écoute sur `0.0.0.0:8080` par défaut
4. **CPU/GPU** : Le code utilise automatiquement CUDA si disponible
5. **Résultats** : Les CSV sont créés automatiquement dans les dossiers `results*/`

## 🐛 Dépannage

### Problème : Le serveur ne démarre pas
- Vérifier que le port 8080 n'est pas déjà utilisé
- S'assurer que l'environnement Conda est activé

### Problème : Les clients ne se connectent pas
- Vérifier que le serveur est démarré et écoute
- Vérifier que tous utilisent le même `data_split`
- Attendre 10 secondes après le démarrage du serveur

### Problème : Résultats manquants
- Vérifier que le serveur s'est terminé correctement (code 0)
- Vérifier les permissions d'écriture dans le dossier `results*/`

---
*Projet terminé : Décembre 2025*

