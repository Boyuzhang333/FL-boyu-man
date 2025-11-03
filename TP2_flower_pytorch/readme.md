**Prénom** :  

**Nom** :  

**Date** :  

# TP2 : Introduction à l'Apprentissage Fédéré  

## Installation :  
* Initialiser l'environnement conda pour le cours  
* [Installation de PyTorch](https://pytorch.org/)  
* Installer le package Flower  
  ```bash
  pip install flwr[simulation]
  pip install tqdm


## Testez:
* Ouvrir trois terminaux dans PyCharm et suivre les étapes suivantes :
  ```bash
  > cd TP2_flower_pytorch
  ```
* Terminal 1:
  ```bash
  > python server.py
  ```
* Terminal 2:
  ```bash
  > python client.py --node_id 0
  ```
* Terminal 3:
  ```bash
  > python client.py --node_id 1
  ```

# Questions :
## Jeux de données

* Quel jeu de données a été testé ?

    Réponse : cifar10

* Comment les données sont-elles distribuées entre les clients ? Où peut-on trouver cette information dans le code ?
    Combien d'images chaque client utilise-t-il pour l'entraînement et la validation ?

    Réponse : iid, 90% pour l'entraînement et 10% pour la validation

## Modèle d'entraînement

*    Quel type de modèle a été utilisé ?
    Réponse : CNN 

* Combien de couches ce modèle possède-t-il et pouvez-vous détailler sa structure ?
    Réponse : 4 couches cachés

## Processus d'entraînement
* Pour chaque client, quel algorithme a été appliqué pour entraîner le modèle local ? Où peut-on trouver cette information dans le code ?

    Réponse : descente de gradient par mini-lot

* Pour chaque client, combien d'époques de calcul local ont été effectuées ? Où peut-on trouver cette information dans le code ?

    Remarque : Lorsque l’ensemble de données a été entièrement parcouru une fois, cela correspond à une époque.

    Réponse : 1 époque

* Combien de tours de communication au total ? Où peut-on trouver cette information dans le code ?

    Réponse : 10 tours

## Évaluation

* Comment l’évaluation est-elle réalisée (par les clients ou le serveur) ?
    Quelles sont les tailles des ensembles de données utilisés pour l’évaluation ?

    Réponse : les clients (10% de données locals)

* Les résultats de l’évaluation sont enregistrés dans le fichier log.txt.
    Où peut-on trouver cette information dans le code ?

    Réponse : weighted_average

* Quelle est l’exactitude finale du modèle ?

    Réponse : ~62% 

## Exercises:
1. Ajouter une option pour la distribution des données (data_split=="non_iid_number") où le pourcentage d’images est tiré selon la
[le loi de Dirichet](https://fr.wikipedia.org/wiki/Loi_de_Dirichlet). 

    Remarque : Vous pouvez utiliser la fonction fournie par numpy :
    
    **Réponse** :
    ```python
    #(Veuillez copie-coller votre code ici 
      
   
    ```
    (alpha est un argument flottant pour ajuster le niveau de non-iid : plus alpha est petit, plus les différences dans le nombre d'images entre les clients sont grandes ; n est le nombre total de clients).
2. Retester votre code avec alpha=0.8 :
   1. Quelles sont les tailles des ensembles de données pour chaque client ?
       Réponse :
   
   2. Quelle est l’exactitude finale du modèle ?
   
      Réponse :
   
3. Exécuter votre code comme suit :
   * In Terminal 1:
     ```bash
     > python server.py --rounds 10
     ```
   * In Terminal 2:
     ```bash
     > python client.py --node-id 0 --data_split non_iid_class -local_epochs 1
     ```
   * In Terminal 3:
     ```bash
     > python client.py --node-id 1 --data_split non_iid_class -local_epochs 1
     ```
     1. Comment les données sont-elles distribuées entre les clients (indice : voir prepare_dataset.py) ?
         Réponse :

     2. Qu’observez-vous sur le modèle appris par rapport aux résultats précédents ?
         Réponse :
  
4. Retester le code avec les mêmes commandes sauf rounds=5, local_epochs=2.

    * L'exécution est-elle plus rapide ? Pourquoi ?
        Réponse :

    * Quelle est votre observation sur le modèle appris par rapport aux résultats précédents ?
        Réponse :

5. Retester le code avec les mêmes commandes sauf rounds=2, local_epochs=5.

    * L'exécution est-elle plus rapide ? Pourquoi ?

    Réponse :

    * Quelle est votre observation sur le modèle appris par rapport aux résultats précédents ?

    Réponse :

6. Compléter le code avec l’autre jeu de données CIFAR100 (fourni par torchvision).

    Réponse :

   
           ```