# Toolbox VAE/PairVAE

AutoFill regroupe les scripts nécessaires pour prétraiter les fichiers
expérimentaux, entraîner les modèles VAE et PairVAE et analyser les résultats.
Le dépôt est pensé pour des personnes à l’aise avec Python mais encore en
apprentissage côté IA : chaque étape fournit des explications pas-à-pas et un
renvoi vers la documentation Sphinx.

## Présentation du projet

AutoFill s’inscrit dans un projet de recherche visant à « auto-compléter » des
signaux de diffusion comme la diffusion des rayons X aux petits angles (SAXS)
et les mesures LES. Les scripts fournis automatisent l’ensemble du pipeline :

- nettoyer et structurer les métadonnées issues des expériences de SAXS/LES ;
- convertir les séries temporelles en HDF5 pour alimenter des VAEs spécialisés
  par modalité ;
- entraîner un modèle PairVAE capable de projeter des signaux LES vers SAXS
  (et inversement) afin de compléter des expériences coûteuses ;
- générer des reconstructions, calculer des métriques et ajuster des paramètres
  physiques via SASFit pour valider les résultats.

Cette logique se retrouve dans le code : les modules `src/dataset` décrivent les
différentes modalités SAXS/LES, tandis que `src/model/pairvae` assemble deux
VAEs pré-entraînés pour réaliser des reconstructions croisées.

## Documentation

- Documentation locale (Sphinx) — comment la construire:
  1. Installer les dépendances de documentation
     - Avec uv (recommandé):
       - `uv pip install -e ".[docs]"`
       - ou si vous utilisez `uv sync`: `uv sync --extra docs`
     - Avec pip:
       - `pip install sphinx sphinx-rtd-theme myst-parser`
  2. Construire le site HTML
     - `sphinx-build -b html docs docs/_build/html`
     - (alternatif) `python -m sphinx -b html docs docs/_build/html`
  3. Ouvrir la documentation générée
     - `docs/_build/html/index.html`

- Guides : [Tutoriels pas-à-pas en anglais](docs/tutorials.rst) pour suivre
  chaque script du pipeline, y compris la partie validation.
- Formats de données : [Référence des HDF5/JSON générés](docs/data_formats.rst)
  avec des exemples de code pour inspecter les artefacts.

### Auteurs

- **Julien Rabault** (julien.rabault@irit.fr)
- **Caroline de Pourtalès** (caroline.de-pourtales@irit.fr)

## Structure du projet

```
AutoFill/
├─ src/
│  ├─ dataset/          # gestion des données
│  ├─ model/            # architectures et entraînement
│  └─ scripts/          # pipeline CLI : prétraitement, conversion, entraînement
├─ configs/             
│  ├─ vae.yml           # config pour VAE
│  └─ pairvae.yml       # config pour PairVAE
├─ requirements.txt     # dépendances
└─ README.md            # guide d’utilisation
```

## Installation

### 1. Prérequis
   Python 3.8+ : uv peut gérer automatiquement l'installation de Python si nécessaire.

### 2. Cloner le projet

```bash
git clone https://github.com/JulienRabault/AutoFill.git
cd AutoFill
```

### 3. Installer uv

**Sur Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

> Remarque : uv sera installé dans ~/.local/bin. Assurez-vous que ce répertoire est inclus dans votre variable
> d'environnement PATH.

**Sur Windows**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

> Remarque : Après l'installation, fermez et rouvrez votre terminal pour que uv soit reconnu. Si nécessaire, ajoutez
> manuellement le chemin d'installation de uv à votre variable d'environnement PATH.

Puis installez les dépendances (optionnel) :

```bash
uv sync --no-dev
```

###  4. (Optionnel) Créer un environnement virtuel avec `venv`

Si vous ne souhaitez pas utiliser `uv`, vous pouvez créer un environnement virtuel avec `venv` :

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt # (Windows : .\env\Scripts\activate)
```

> Pour la suite, si vous utilisez `venv` ou `conda`, remplacez `uv run` par `python`.

## Pipeline

Le workflow complet se déroule en six étapes :

1. **Prétraiter les métadonnées CSV** pour nettoyer et fusionner les sources.
2. **Convertir les fichiers `.txt` en HDF5** (VAE ou PairVAE) tout en générant le dictionnaire JSON des métadonnées.
3. **Entraîner un VAE** à l’aide du HDF5 et du dictionnaire de conversion.
4. **Préparer les jeux de données appariés** puis **entraîner le PairVAE**.
5. **Lancer l’inférence** sur de nouveaux fichiers ou sur un dataset existant.
6. **(Optionnel) Calculer des métriques ou lancer une recherche de grilles**.

Les sections ci-dessous détaillent chaque étape. Pour davantage de contexte (structure HDF5, choix des hyperparamètres, bonnes pratiques), reportez-vous à `docs/tutorials.rst` et `docs/data_formats.rst`.

### 1. Prétraitement CSV (`scripts/01_csv_pre_process.py`)

Fusionnez et nettoyez plusieurs fichiers CSV de métadonnées avant toute conversion. Le script se concentre sur le CPU et peut prendre plusieurs heures sur de gros volumes de données (pensez à `tmux`).

```bash
python scripts/01_csv_pre_process.py \
  data/raw_csv/file1.csv data/raw_csv/file2.csv \
  data/metadata_clean.csv \
  -s ";"
```

Arguments importants :

- `inputs` : un ou plusieurs chemins vers des fichiers CSV (séparateur `;`).
- `output` : chemin du fichier CSV nettoyé produit (séparateur `,`).
- `--sep` : séparateur du CSV d’entrée si différent de `;`.

Résultat : `data/metadata_clean.csv`, un unique fichier propre prêt pour la conversion.

**Vérifier les chemins `path`**

Les valeurs de la colonne `path` doivent être relatives à `--data_dir`. Utilisez `scripts/saminitycheck.py` pour détecter les chemins manquants avant la conversion :

```bash
python scripts/saminitycheck.py \
  --csv data/metadata_clean.csv \
  --basedir data/txt/
```

### 2. Conversion `.txt` → HDF5 (`scripts/02_txtTOhdf5.py`)

Ce script transforme les séries temporelles `.txt` et le CSV nettoyé en un fichier HDF5 compatible VAE, ainsi qu’en un dictionnaire JSON décrivant l’encodage des métadonnées catégorielles.

```bash
python scripts/02_txtTOhdf5.py \
  --data_csv_path data/metadata_clean.csv \
  --data_dir data/txt/ \
  --final_output_file data/all_data.h5 \
  --json_output data/metadata_dict.json \
  --pad_size 900
```

Arguments clés :

- `--data_csv_path` : métadonnées nettoyées contenant au moins une colonne `path`.
- `--data_dir` : dossier racine des fichiers `.txt`.
- `--pad_size` : longueur maximale des séries ; les signaux sont tronqués ou complétés par des zéros si nécessaire.
- `--json_output` : fichier qui contiendra le mapping `{valeur → entier}` pour chaque métadonnée catégorielle.

L’HDF5 résultant contient les jeux `data_q`, `data_y`, `len`, `csv_index` ainsi que chaque colonne de métadonnées. Une description détaillée du format et des types est disponible dans `docs/data_formats.rst`.

### 3. Entraîner un VAE (`scripts/03_train.py --mode vae`)

Une fois `all_data.h5` et `metadata_dict.json` générés, lancez `scripts/03_train.py` :

```bash
python scripts/03_train.py \
  --mode vae \
  --config config/vae_saxs.yaml \
  --name AUTOFILL_SAXS_VAE \
  --hdf5_file data/all_data.h5 \
  --conversion_dict_path data/metadata_dict.json \
  --technique saxs \
  --material ag
```

Points d’attention pour les débutants :

- `--technique` et `--material` surchargent les filtres du YAML et ajustent aussi le `pad_size`. Ne les cumulez pas avec des valeurs contradictoires dans la configuration.
- Assurez-vous que `model.args.input_dim` dans le YAML correspond exactement au `pad_size` utilisé lors de la conversion.
- Gardez `num_workers` raisonnable (ex. nombre de cœurs CPU disponibles) pour éviter une surcharge.

Les paramètres essentiels à personnaliser dans la configuration :

- `run_name` / `logdir` pour contrôler où sont stockés logs et checkpoints.
- `dataset.hdf5_file`, `dataset.conversion_dict_path` et `dataset.metadata_filters` pour sélectionner vos données.
- `training` (epochs, batch_size, scheduler) pour ajuster l’optimisation.

### 4. Conversion appariée et PairVAE

Pour des signaux appariés (ex. SAXS/LES), convertissez d’abord les fichiers avec `scripts/04_pair_txtTOhdf5.py` :

```bash
python scripts/04_pair_txtTOhdf5.py \
  --data_csv_path data/metadata_clean.csv \
  --data_dir data/txt/ \
  --final_output_file data/all_pair_data.h5 \
  --json_output data/pair_metadata_dict.json \
  --pad_size 900
```

Le fichier HDF5 généré contient `data_q_saxs`, `data_y_saxs`, `data_q_les`, `data_y_les`, `len`, `valid` et `csv_index`, plus les métadonnées alignées. Les chemins `saxs_path` et `les_path` doivent tous deux être valides.

Ensuite, lancez l’entraînement PairVAE :

```bash
python scripts/03_train.py \
  --mode pair_vae \
  --config config/pair_vae.yaml \
  --name AUTOFILL_SAXS_PAIRVAE \
  --hdf5_file data/all_pair_data.h5 \
  --conversion_dict_path data/pair_metadata_dict.json \
  --material ag
```

La configuration `pair_vae` contrôle notamment les poids de reconstruction et la taille du latent partagé. Consultez `docs/tutorials.rst` pour un exemple complet d’ajustement.

### 5. Inférence (`scripts/05_infer.py`)

Utilisez ce script pour charger un checkpoint et générer des reconstructions ou des prédictions.

```bash
python scripts/05_infer.py \
  --outputdir outputs/vae_demo \
  --checkpoint logs/vae_model.ckpt \
  --data_path data/new_data.h5 \
  --batch_size 64 \
  --plot
```

Options principales :

- `--data_path` : HDF5 ou CSV à traiter. Pour un CSV, ajoutez `--data_dir` afin que le script retrouve les fichiers `.txt`.
- `--mode` : uniquement pour PairVAE (`les_to_saxs`, `saxs_to_les`, etc.).
- `--sample_frac` : pour tester rapidement un sous-ensemble.
- `--plot` : enregistre les signaux au format PNG en plus des `.npy`.

Les sorties sont placées dans le dossier indiqué, au format `.npy` (shape `[pad_size, 2]` : `y` et `q`).

### 6. Validation et métriques (`scripts/06_val_metrics.py`)

Après l’inférence, utilisez `scripts/06_val_metrics.py` pour consolider les performances du modèle et générer un rapport reproductible. Le script charge un checkpoint Lightning (`.ckpt`), parcourt un HDF5 et calcule :

- des métriques de reconstruction (MAE, MSE, RMSE, R²) sur un sous-ensemble aléatoire du dataset ;
- des ajustements physiques via SASFit (diamètre, longueur et concentration) pour un petit échantillon, avec comparaison optionnelle à la vérité terrain.

```bash
python scripts/06_val_metrics.py \
  --checkpoint logs/vae_model.ckpt \
  --data_path data/all_data.h5 \
  --conversion_dict data/metadata_dict.json \
  --outputdir outputs/vae_metrics \
  --eval_percentage 0.1 \
  --sasfit_percentage 0.005
```

Points importants :

- **VAE** : fournissez uniquement `--checkpoint`, `--data_path`, `--conversion_dict` et `--outputdir`.
- **PairVAE** : ajoutez `--mode` (`les_to_saxs` ou `saxs_to_saxs`) pour préciser le domaine de sortie évalué ; les autres modes ne possèdent pas de vérité terrain adaptée pour les reconstructions.
- Ajustez `--eval_percentage` (reconstruction) et `--sasfit_percentage` (SASFit) selon la taille du dataset : 0.05 représente 5 % des échantillons.
- Pour accélérer SASFit, fixez `--n_processes` à un nombre de cœurs adaptés ; par défaut, le script utilise *n_cpu - 1*.

Les résultats sont écrits dans `--outputdir` :

- `validation_metrics.yaml` récapitule toutes les métriques et la configuration utilisée ;
- `metrics_summary.txt` propose un résumé prêt à partager ;
- `reconstruction_metrics_detailed.csv` contient les scores par échantillon lorsque des reconstructions ont été calculées.

Pour des explorations plus avancées (recherche de grilles, comparaison de modèles), consultez la section [Expert: Grid Search](#expert-grid-search).


### Expert: Grid Search

Vous pouvez automatiser l’optimisation des hyperparamètres grâce à la recherche par grille intégrée.

**1. Définissez un bloc `param_grid` dans votre fichier YAML de configuration (voir `config/vae.yaml`):**

Le bloc `param_grid` vous permet de spécifier les hyperparamètres à explorer et les différentes valeurs à tester pour
chacun. Chaque clé doit correspondre à un paramètre de votre configuration, en utilisant la notation pointée (`.`) pour
accéder aux champs imbriqués.

```yaml
param_grid:
  training.beta: [ 0.001, 0.0001 ]                # Teste deux valeurs pour beta
  model.args.latent_dim: [ 64, 128, 256 ]         # Teste trois dimensions latentes différentes
  training.batch_size: [ 16, 32, 64 ]             # Teste trois tailles de batch
```

- La recherche par grille générera automatiquement toutes les combinaisons possibles de ces valeurs.
- Les clés (par exemple `training.beta`, `model.args.latent_dim`) doivent correspondre à la structure de votre fichier
  de configuration.
- Vous pouvez ajouter ou retirer des paramètres selon vos besoins, et chaque paramètre peut avoir autant de valeurs que
  vous le souhaitez.

#### Comment ça marche:

Avec l’exemple ci-dessus, la recherche par grille lancera (2 \times 3 \times 3 = 18) entraînements différents, chacun
avec une combinaison unique de beta, latent_dim et batch_size.

> Vous pouvez inclure n’importe quel paramètre de votre fichier de configuration dans le `param_grid`, à condition
> d’utiliser le bon chemin (notation pointée) vers ce paramètre.

**2. Lancez la recherche par grille:**

```bash
python scripts/03_train.py \
  --mode vae \
  --gridsearch \
  --config config/vae.yml \
  --name AUTOFILL_SAXS_VAE \
  --hdf5_file data/all_data.h5 \
  --conversion_dict_path data/all_data.json \
  --technique saxs \
  --material ag
```

- Chaque combinaison de paramètres sera testée séquentiellement.
- Les résultats et configurations sont sauvegardés dans le dossier `gridsearch_results/`.
- Utilisez l’option `verbose` dans votre configuration ou dans le code pour contrôler l’affichage des logs.

> La recherche par grille écrasera les valeurs correspondantes de votre configuration de base pour chaque essai.  
> Assurez-vous que les clefs de `param_grid` correspondent à la structure imbriquée de votre fichier de configuration.

---

### Contact

Pour toute question ou problème, n’hésitez pas à contacter :

- **Julien Rabault** (julien.rabault@irit.fr)
- **Caroline de Pourtalès** (caroline.de-pourtales@irit.fr)
