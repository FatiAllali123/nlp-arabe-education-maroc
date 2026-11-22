# Deep Learning : classification et génération de texte arabe (RNN, LSTM, GRU, GPT-2)


## 1. De quoi parle ce projet ?

Dans ce TP, j'ai travaillé sur des textes **en arabe** parlant d'un seul sujet : **l'éducation au Maroc**. Le travail est en deux parties :

- **Partie 1 :** j'ai construit un dataset, je l'ai nettoyé, puis j'ai entraîné quatre modèles (RNN, Bidirectional RNN, GRU, LSTM) pour classer les textes selon leur pertinence. J'ai aussi testé ces modèles sur de la génération de texte avec le score BLEU.
- **Partie 2 :** j'ai pris un GPT-2 déjà entraîné sur l'arabe, je l'ai adapté à mon dataset (fine-tuning) et je lui ai fait écrire des paragraphes à partir d'une phrase.

Tout a été fait avec **PyTorch** sur **Google Colab**.

## 2. Contenu du dépôt

```
├── notebooks/
│   ├── 01_preparation_dataset.ipynb      # collecte des textes et création du dataset
│   ├── Partie1_Classification.ipynb      # prétraitement, RNN/BiRNN/GRU/LSTM, évaluation, BLEU
│   └── Partie2_GPT2_final.ipynb          # fine-tuning de GPT-2 et génération
├── data/
│   └── dataset_education_maroc_full.csv  # 342 textes avec leur score
├── .gitignore
└── README.md
```

Les notebooks sont déposés **déjà exécutés** : tous les résultats (tableaux, courbes, matrices de confusion, textes générés) sont visibles directement dans GitHub, dans les sorties des cellules. Je n'ai donc pas mis les fichiers de résultats (CSV, JSON) dans le dépôt, ils sont recréés quand on relance les notebooks.

Les poids des modèles (le GPT-2 fine-tuné fait environ 500 Mo) ne sont pas mis sur GitHub car c'est trop gros. On peut les regénérer en relançant les notebooks.

## 3. Le dataset

- **342 textes** (passages d'environ 100 à 120 mots) issus de **93 articles** (Wikipédia arabe et articles de revue).
- Chaque texte a un **score de pertinence entre 0 et 10**.
- Pour faire de la classification, j'ai découpé le score en 3 classes : **faible** (0 à 5,5), **moyenne** (6 à 7,5) et **forte** (8 à 10). Il y a 84 textes faibles, 103 moyens et 155 forts.
- Comme plusieurs textes viennent du même article, j'ai séparé train / validation / test **par article** (246 / 37 / 59 textes). Sinon, des morceaux du même article se retrouveraient dans le train et dans le test, et les résultats seraient trop beaux pour être vrais.

## 4. Partie 1 : classification avec RNN, BiRNN, GRU et LSTM

**Prétraitement :** nettoyage (j'enlève la ponctuation, les chiffres, les voyelles), tokenisation, suppression des mots vides, stemming (Snowball et ISRI), lemmatisation (qalsadi) et normalisation des lettres. J'ai gardé la lemmatisation pour entraîner les modèles. Ça fait passer le vocabulaire de 10 087 mots à 4 130.

**Modèles :** les quatre ont la même structure (Embedding, couche récurrente, couche linéaire). Seule la cellule change. J'ai cherché les meilleurs hyperparamètres avec une recherche aléatoire (12 essais par modèle, choisis sur la validation).

**Résultats sur le test (59 textes) :**

| Modèle | Accuracy | F1 macro |
|---|---|---|
| RNN | 0,373 | 0,366 |
| BiRNN | 0,424 | 0,396 |
| GRU | 0,492 | 0,487 |
| **LSTM** | **0,593** | **0,590** |

Le **LSTM est le meilleur**, puis le GRU. Le RNN simple est à peine meilleur que le hasard (33 % avec 3 classes). C'est logique : le LSTM et le GRU ont des portes qui gardent mieux l'information, alors que le RNN simple oublie vite. Mais le test n'a que 59 textes, donc les petits écarts (par exemple GRU contre BiRNN) ne sont pas très fiables.

**Le BLEU :** le BLEU compare un texte généré avec un texte de référence, donc il ne marche pas pour un classifieur. J'ai donc fait écrire aux quatre architectures la suite d'un texte (20 mots à partir de 30 mots). Résultat : les quatre modèles sont mauvais (BLEU-4 de 0,0006) et font même moins bien qu'une liste fixe des mots les plus fréquents. Ils répètent toujours les mêmes expressions (« في المغرب »). Avec si peu de données, des RNN entraînés de zéro ne savent pas écrire de l'arabe.

## 5. Partie 2 : fine-tuning de GPT-2

**Modèle :** `aubmindlab/aragpt2-base` (un GPT-2 de 135 millions de paramètres, déjà entraîné sur de l'arabe), avec la bibliothèque `transformers`. Le TP parle de `pytorch-transformers`, mais cette bibliothèque a été renommée `transformers`. L'utilisation est la même.

**Ce que j'ai fait :** fine-tuning sur mes textes (mêmes articles de train que la partie 1), puis génération de paragraphes à partir d'une phrase, avec plusieurs méthodes de décodage (greedy, beam search, sampling).

**Résultats sur le test :**

| Modèle | Perplexité test | BLEU-2 | BLEU-4 |
|---|---|---|---|
| RNN (partie 1) | - | 0,009 à 0,011 | 0,0006 |
| GPT-2 pré-entraîné | 117,5 | 0,039 | 0,0130 |
| **GPT-2 fine-tuné** | **89,3** | **0,049** | **0,0164** |

GPT-2 écrit un texte beaucoup plus cohérent que les RNN. Le fine-tuning baisse la perplexité d'environ 24 % et le BLEU augmente aussi.

**Le problème que j'ai eu :** mon premier fine-tuning a rendu le modèle *moins bon* qu'avant (perte de validation 4,75 contre 4,52). Le deuxième essai était stable mais ne changeait presque rien, et la perte d'entraînement restait bizarrement haute (8 à 10 alors que le modèle de départ est à 4,5). J'ai fini par faire un test de diagnostic : en mode entraînement, le **dropout** (0,1) fait doubler la perte du modèle de départ (10,54 contre 5,20). Après avoir désactivé le dropout pendant le fine-tuning, tout s'est arrangé.

## 6. Limites

- Le dataset est **petit** (342 textes). C'est la limite principale, surtout pour les RNN.
- Le test est petit (59 textes en classification, 165 fenêtres pour le BLEU) et j'ai fait un seul entraînement par modèle : les petits écarts ne sont pas forcément significatifs.
- En génération, l'accuracy sur les mots est trompeuse : 38,8 % des mots à prédire sont inconnus du vocabulaire, et les RNN ont surtout appris à prédire « inconnu ».
- Le BLEU n'a qu'une seule référence : il est bas même pour une suite correcte.
- GPT-2 **invente des informations**. Il écrit de façon fluide mais peut se tromper ou se contredire (par exemple, il a parlé de l'Arabie saoudite alors que ma phrase parlait du Maroc). Ses textes ne sont pas des sources fiables.
- Le fine-tuné a appris à terminer ses textes très tôt, donc certaines suites sont trop courtes. Le texte généré n'a pas non plus de ponctuation, car j'ai gardé seulement les lettres arabes.

## 7. Comment relancer le projet

1. Ouvrir les notebooks dans Google Colab et mettre `dataset_education_maroc_full.csv` dans un dossier Google Drive nommé `Projet_NLP_Education_Maroc`.
2. Lancer `Partie1_Classification.ipynb` en entier. Il crée un dossier `resultats_partie1` dans Google Drive (résultats, vocabulaire, `config.json`). La partie 2 relit ce dossier pour avoir la même séparation train / test et comparer son BLEU avec celui de la partie 1, donc il faut lancer la partie 1 d'abord.
3. Lancer `Partie2_GPT2_final.ipynb` avec un **GPU** (Colab : GPU T4). Le premier lancement télécharge le modèle (environ 500 Mo). Les résultats sont enregistrés dans un dossier `resultats_partie2` sur Drive.

---

## 8. Ce que j'ai appris

Voilà ce que je retiens de ce TP :

1. **Les données comptent plus que le modèle.** Avec 342 textes, tous mes modèles faisaient du sur-apprentissage : presque 100 % sur le train, bien moins sur la validation. J'ai compris que la taille du dataset limite tout, même quand on fait un bon réglage.

2. **Bien séparer train et test.** Au début, je ne pensais pas que des textes du même article pouvaient fausser les résultats. Séparer par article m'a appris à me méfier de la fuite d'information (*data leakage*).

3. **Le prétraitement de l'arabe n'est pas parfait.** Le stemming et la lemmatisation font parfois des erreurs (par exemple « ملكي » devient « لكي »). Ils réduisent le vocabulaire, mais ils ne sont pas toujours justes.

4. **RNN, GRU et LSTM : j'ai vu la différence.** Le RNN simple apprend mal, le GRU et le LSTM font mieux grâce à leurs portes. En classification, ça se voit dans les chiffres (0,37 de F1 pour le RNN, 0,59 pour le LSTM).

5. **Une métrique peut mentir.** L'accuracy de 0,39 en génération avait l'air correcte, mais elle venait juste du fait que les modèles prédisaient « inconnu ». Il faut toujours regarder les textes et pas seulement les chiffres. J'ai aussi compris que le BLEU a des limites, et qu'une référence minimale très simple est utile pour savoir si un modèle vaut quelque chose.

6. **Le transfert d'apprentissage est puissant.** Là où mes RNN échouaient, GPT-2 écrit tout de suite un bon arabe, parce qu'il a déjà appris la langue sur un énorme corpus. Il suffit ensuite de l'adapter à mon sujet.

7. **Un fine-tuning peut échouer, et il faut comparer avec le modèle de départ.** Mon premier fine-tuning avait empiré le modèle, et sans la comparaison sur la validation, je ne l'aurais pas vu. Le plus important que j'ai retenu est la façon de déboguer : devant une perte d'entraînement bizarre, j'ai fait un petit test (mode évaluation contre mode entraînement) au lieu de continuer à changer les hyperparamètres au hasard. Ça m'a permis de trouver le vrai problème, le dropout.

8. **Une IA qui écrit bien n'écrit pas forcément vrai.** Les textes de GPT-2 sont fluides, mais pleins d'informations inventées. C'est important à savoir si on veut l'utiliser pour de vraies applications.

**Les difficultés** : régler la tâche de classification avec un petit dataset, comprendre pourquoi les RNN ne généraient pas de bon texte, et trouver pourquoi le fine-tuning ne marchait pas au début.
