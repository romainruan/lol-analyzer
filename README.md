# lol-analyzer

Analyse de parties de League of Legends : collecte via l'API Riot, modèle de
probabilité de victoire (Scikit-learn), moments clés, puis débrief généré par un
LLM local (Ollama). Le tout conteneurisé avec Docker.

> Projet personnel, non affilié à Riot Games. Ne publiez ni votre clé API ni de
> données personnelles de joueurs.

## État d'avancement

- [x] Client API avec rate limiting (`src/riot_client.py`)
- [x] Collecte des parties et timelines (`src/collect.py`)
- [x] Features à 5/10/15/20/25 min, sans fuite de données (`src/features.py`)
- [x] Construction du jeu de données (`src/build_dataset.py`)
- [x] Features par lane, dont mid contre mid (`src/features.py`)
- [x] Entraînement et évaluation par minute, validation croisée par partie (`src/train.py`)
- [x] Repères du rôle mid et comparaison avec vos parties (`src/benchmarks.py`)
- [x] Courbe de probabilité de victoire par partie et moments clés (`src/winprob.py`)
- [ ] Interprétation (SHAP)
- [x] Débrief par LLM local : résumé contrôlé, reprise automatique et repli (`src/debrief.py`)
- [ ] Interface Streamlit

## Démarrage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # puis collez votre clé Riot dans .env
python -m pytest -q         # les tests ne nécessitent pas de clé
```

### 1. Collecter des données

```bash
# vos parties classées
python -m src.collect --riot-id "VotrePseudo#EUW" --count 30

# un échantillon de votre tranche de rang (ex. Gold II) pour entraîner le modèle
python -m src.collect --tier GOLD --division II --pages 2 --per-player 5
```

Une partie + sa timeline = 2 requêtes, et la clé de dev autorise ~100 requêtes
par 2 minutes : 1 000 parties prennent environ 40 minutes. Les téléchargements
sont repris là où ils se sont arrêtés.

### 2. Construire le tableau d'entraînement

```bash
python -m src.build_dataset --raw-dir data/raw/sample --out data/dataset.csv
```

Chaque ligne est une partie vue à une minute donnée (écarts d'or, d'XP, de CS,
de kills, de tours, d'objectifs et de wards entre l'équipe bleue et la rouge),
la cible étant `blue_win`.

### 3. Entraîner et évaluer

```bash
python -m src.train --data data/dataset.csv --out results
```

Un modèle par minute (5, 10, 15, 20, 25), évalué hors échantillon avec une
validation croisée **par partie**. Trois modèles sont comparés : régression
logistique sur les écarts d'équipe, gradient boosting, et régression logistique
avec l'écart d'or de chaque lane. Sorties dans `results/` :
`metrics_by_minute.csv`, `auc_by_minute.png` (AUC avec intervalle de confiance
bootstrap), `lane_coefficients.csv` (poids de chaque lane) et `calibration.csv`.

### 4. Repères mid et comparaison avec vos parties

```bash
python -m src.benchmarks --raw-dir data/raw/sample --me-dir data/raw/me
```

Affiche les percentiles des mid laners de l'échantillon (CS/min, vision/min, KDA,
participation aux kills, part des dégâts, CS/or/XP à 10 min) et situe vos propres
parties en mid par rapport à eux. `collect --riot-id` enregistre votre puuid dans
`data/raw/me/puuid.txt` pour retrouver vos parties.

### 5. Courbe de probabilité de victoire et moments clés

```bash
# une ligne par minute (2 à 40) au lieu de 5/10/15/20/25
python -m src.build_dataset --raw-dir data/raw/sample --all-minutes --out data/dataset_all.csv
python -m src.winprob train --data data/dataset_all.csv
python -m src.winprob analyze --raw-dir data/raw/me --limit 5
```

Un seul modèle (régression logistique) sert pour toutes les minutes : le poids de
chaque variable évolue avec le temps. Pour chaque partie analysée, `results/curves/`
contient un graphique (probabilité de victoire minute par minute, vue de votre
équipe) et un JSON avec les **moments clés** : les plus gros sauts de probabilité,
associés aux événements de la minute (objectifs, tours, kills). Ce JSON est
l'entrée du débrief par LLM : le modèle de langage ne lit jamais la timeline brute.

Avertissement : un saut de probabilité n'est pas une cause. Il signale qu'un
écart d'or, d'XP ou d'objectifs a bougé ; l'événement associé est le plus
plausible, pas prouvé.

### 6. Débrief par un LLM local (Ollama)

```bash
ollama pull qwen2.5:7b                          # une seule fois (environ 4,7 Go)
python -m src.winprob analyze --limit 5         # courbes et moments clés
python -m src.debrief --limit 1 --dry-run       # affiche ce que le modèle recevra
python -m src.debrief --limit 3                 # mode hybride (par défaut)
python -m src.debrief --limit 3 --mode template # sans modèle de langage (référence)
python -m src.debrief --limit 3 --mode llm      # le modèle écrit tout le débrief
```

**Le code analyse, le modèle rédige.** Une première version donnait au modèle des
chiffres bruts à interpréter : il mélangeait les chiffres, attribuait au joueur la
probabilité de l'équipe et inventait des explications. Trois modes permettent de
comparer leur fidélité :

| Mode | Qui écrit quoi | Rôle du contrôle |
|---|---|---|
| `template` | le code écrit tout (aucun LLM) | référence |
| `hybrid` (défaut) | le code écrit moments clés, points forts, axes de progrès, tendance ; le modèle n'écrit que le résumé (3 phrases) | accepte, fait réessayer ou remplace le résumé |
| `llm` | le modèle écrit tout | signale les problèmes |

**Mode hybride.** Le modèle ne voit que la partie et la forme de la courbe
(« 9e minute : de 54 % à 44 % »), jamais les événements ni les autres joueurs : il ne
peut donc se tromper ni de cause ni de champion. Son résumé passe le contrôle ; en cas
d'échec il reçoit la liste de ses erreurs et un autre tirage (graine différente), jusqu'à
3 tentatives (`--attempts`). Si aucune ne passe, le résumé écrit par le code le remplace
et la fiche de relecture le signale.

**Contrôle automatique** : nombres absents des faits, pourcentages qui n'existent pas
comme pourcentages, transitions « de A % à B % » qui mélangent deux sauts, moments clés
omis, champions cités à la mauvaise minute, minutes écrites en lettres, mots anglais,
caractères chinois, formulations causales (« grâce à », « suite à »...), sections écrites
sans données, résumé de plus de 3 phrases. Il ne vérifie ni le sens des phrases, ni qui a
tué qui : relisez chaque débrief.

**Mesure de fidélité.** Chaque exécution met à jour `results/debriefs/review.csv` (une
ligne par débrief et par mode, le mode hybride étant étiqueté `hybride v4`). Remplissez à
la main `erreurs_chiffres`, `inventions` et `omissions` : vos annotations sont conservées
quand vous relancez. Le nombre d'erreurs par débrief permet de comparer les modes et les
modèles (`--model llama3.1:8b`). Fermez le fichier dans Excel avant de relancer.

**Observations préliminaires** (Qwen2.5 7B, 3 parties, à confirmer sur une dizaine) :
le mode `llm` produit des problèmes détectés automatiquement sur les 3 débriefs
(causalité, sections écrites sans données, différences de pourcentages calculées) et
des erreurs que seule la relecture trouve (un kill attribué au mauvais joueur) ; le
résumé du mode hybride contenait encore des formulations causales et des fautes de
français, d'où la reprise automatique.

### Avec Docker

```bash
docker compose build
docker compose run --rm app python -m src.collect --riot-id "VotrePseudo#EUW"
```

## Points de rigueur

- Découper train/test **par partie** (jamais deux minutes d'une même partie des
  deux côtés), idéalement par patch ou par date.
- Mesurer AUC, log-loss et calibration, pas seulement la précision.
- Vérifier chaque débrief LLM : un modèle local peut inventer un chiffre.

## Limites connues

- Les modèles de minutes tardives n'utilisent que les parties assez longues
  (biais de survie) : comparer les minutes entre elles demande de la prudence.
- Quelques centaines de parties donnent des intervalles de confiance larges ;
  les différences de quelques points d'AUC entre modèles ne sont pas significatives.
- Les poids du modèle « lanes » sont des corrélations, pas des effets causaux :
  les lanes d'une même équipe sont corrélées entre elles.
- Données issues d'un seul niveau de jeu (Master) : ne pas généraliser aux autres rangs.
- `league-v4/entries` ne couvre que IRON à DIAMOND ; Master et au-delà utilisent
  d'autres endpoints.
- La clé de développement expire toutes les 24 h.
