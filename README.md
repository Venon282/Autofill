# Toolbox VAE/PairVAE

# TODO :

- [x] Faire l'inference => utiliser les modeles
- [ ] Expliquer la sorti des train, poids + analyse de courbe
- [x] Bouger les fichiers/dossier
- [ ] Renforcer les explications sur le fonctionnement du pairVAE ?

### Auteurs :

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

1. **[Prétraitement CSV](#1-prétraitement-csv)** : fusion et nettoyage → `metadata_clean.csv`
2. **[Conversion .txt → HDF5 (VAE)](#2-conversion-txt--hdf5-avec-txttohdf5py)** : séries temporelles + métadonnées →
   `all_data.h5` + `metadata_dict.json`
3. **[Entraînement du modèle VAE](#3-entrainement-du-modèle-à-partir-du-fichier-hdf5)** : filtre, configuration YAML →
   lancement du training VAE
4. **[Conversion .txt → HDF5 (PairVAE)](#5-conversion-txt--hdf5-pour-pairvae)** : séries temporelles + métadonnées →
   `all_data.h5` + `metadata_dict.json`
5. **[Entraînement du modèle PAIRVAE](#6-entrainement-du-modèle-pairvae)** : filtre, configuration YAML → lancement du
   training PairVAE
6. **[Inference (optionnelle)](#6-inference-optionnelle)** : analyse des résultats à partir des poids entraînés
7. **[Expert: Grid Search](#expert-grid-search)** : optimisation des hyperparamètres
   avec la recherche par grille intégrée.

### 1. Prétraitement CSV

`01_csv_pre_process.py`

Ce script fusionne et nettoie plusieurs fichiers CSV de métadonnées. Cette convertion tourne sur CPU et demande beaucoup
de ressource pour aller vite. 

>Utiliser `tmux` pour lancer le script en arrière plan. Par exemple pour 1.5M de `.txt` cela prend environ 8h.

**Arguments:**

- `<inputs>` : un ou plusieurs chemins vers des fichiers CSV (séparateur ;).
- `<output>` : chemin du fichier CSV nettoyé de sortie (séparateur ,).
- `--sep` : séparateur des colonnes dans le csv

```bash
python scripts/01_csv_pre_process.py \
  data/raw_csv/file1.csv data/raw_csv/file2.csv \
  data/metadata_clean.csv \
  -s ";"
```

> **Exemple**: après exécution, le fichier `data/metadata_clean.csv` contient toutes les métadonnées normalisées. Vous
> pourrez l’utiliser à l’étape suivante pour la conversion au format HDF5.

### 2. Conversion `.txt` → HDF5 avec `02_txtTOhdf5.py`

Objectif: convertir les séries temporelles (`.txt`) et le CSV de métadonnées en un unique fichier HDF5.

Arguments:

* `--data_csv_path` : chemin vers le fichier CSV de métadonnées (doit contenir au moins une colonne path vers les
  fichiers .txt).
* `--data_dir` : dossier racine contenant les fichiers .txt.
* `--final_output_file` : chemin de sortie pour le fichier .h5 généré.
* `--json_output` : chemin de sortie pour le dictionnaire de conversion des métadonnées catégorielles (au format JSON).
* `--pad_size` : longueur maximale des séries temporelles (padding ou troncature appliqué si nécessaire). Default : 500.

```bash
python scripts/02_txtTOhdf5.py \
  --data_csv_path data/metadata_clean.csv \
  --data_dir data/txt/ \
  --final_output_file data/all_data.h5 \
  --json_output data/metadata_dict.json \
  --pad_size 900
```

> **Exemple**: en sortie, `data/all_data.h5` contient `data_q`, `data_y`, `len`, `csv_index` et toutes les métadonnées,
> et `data/metadata_dict.json` recense les encodages catégoriels. Vous utiliserez ces deux fichiers pour l’entraînement.

**Structure HDF5 générée :**

```text
final_output.h5
├── data_q          [N, pad_size]
├── data_y          [N, pad_size]
├── len             [N]
├── csv_index       [N]
├── <metadata_1>    [N]
├── <metadata_2>    [N]
└── ...             [N]
```

> **Note:** `data_q` et `data_y` sont les séries temporelles et `csv_index` est l’index du CSV d’origine. Les colonnes
> de métadonnées sont ajoutées à la fin.

**Attention aux chemins (path) dans le CSV :**

Les chemins indiqués dans la colonne path du CSV doivent être relatifs au répertoire --data_dir. Le script les concatène
pour localiser les fichiers `.txt`. Toute incohérence entraînera des erreurs ou des fichiers ignorés.
Avant de lancer la conversion, vous pouvez utiliser `saminitycheck.py` pour valider que tous les fichiers .txt
référencés dans le CSV existent réellement dans le répertoire `--data_dir`.

**Exécutez le script `saminitycheck.py` :**

```bash
python scripts/saminitycheck.py \
  --csv data/metadata_clean.csv \
  --basedir data/txt/
```

Ce script vérifiera que chaque chemin dans la colonne path (colonne contenant `"path"` dans son nom) du CSV correspond à
un fichier existant dans le répertoire `--basedir`. Si des fichiers manquent, ils seront listés.

### 3. Entraînement du modèle à partir du fichier HDF5 `03_train.py`

Une fois le HDF5 et le JSON générés, lancez l’entraînement:

```bash
python scripts/03_train.py \
  --mode vae \
  --config config/vae.yaml \
  --name AUTOFILL_SAXS_VAE \
  --hdf5_file data/all_data.h5 \
  --conversion_dict_path data/metadata_dict.json \
  --technique saxs \
  --material ag
```

> **IMPORTANT** : l'utilisation de `--technique` et `--material`dans les paramettres du script, SURCHARGE les filtres
> définis dans le fichier de configuration YAML. Cela surcharge aussi les `pad_size` dans les transformations .
>
> `LES` pad_size **500** par defaut
>
> `SAXS` pad_size **54** par defaut
>
> Il est donc préférable de ne pas les définir dans le YAML si vous utilisez ces arguments.

> **Exemple**: ici `data/all_data.h5` et `data/metadata_dict.json` sont issus de l’étape précédente, et seront filtrés
> sur `technique=saxs` et `material=ag`.

### Paramètres minimum modifiables dans le YAML (config/vae.yml)

* **`experiment_name`** : nom de l’expérience (création de sous-dossier dans logdir).
* **`logdir`** : dossier où seront stockés logs et checkpoints.
* **`dataset`**

    * `hdf5_file` : chemin vers votre fichier `.h5`.
    * `conversion_dict_path` : chemin vers le JSON de mapping.
    * `metadata_filters` : filtres à appliquer sur les métadonnées (ex. material, technique, type, shape).
    * `sample_frac` : fraction d’échantillonnage (entre 0.0 et 1.0).
* **`transforms_data`**

    * `q` et `y` – `PaddingTransformer.pad_size` doit correspondre à `pad_size` utilisé lors de la conversion `.txt`.
* **`training`**

    * `num_epochs` : nombre d’époques maximales.
    * `batch_size` : taille de batch.
    * `num_workers` : nombre de workers DataLoader. (Nombre de cpu disponible)
    * `max_lr`, `T_max`, `eta_min` : planning de taux d’apprentissage.
    * `beta` : coefficient β du VAE.
* **`model.args`**
    * `input_dim` : doit être égal à pad_size.

> **Note :** en dehors de ces clefs, tout autre paramètre dans le YAML n’est pas nécessairement safe à modifier si vous
> débutez en IA. Respectez surtout la cohérence pad_size / input_dim et les chemins d’accès pour éviter les erreurs.


### 4. Entraînement du modèle PAIRVAE à partir du fichier HDF5 `03_train.py`

De la même manière que [Conversion .txt → HDF5 (VAE)](#2-conversion-txt--hdf5-avec-txttohdf5py), vous pouvez convertir
vos séries temporelles en un fichier HDF5 pour l’entraînement du PairVAE. Le script `04_pair_txtTOhdf5.py` est conçu
pour cela.

Arguments :

```bash
python scripts/03_train.py \
  --mode pairvae \
  --config config/pairvae.yml \
  --name AUTOFILL_SAXS_PAIRVAE \
  --hdf5_file data/all_data_pair.h5 \
  --conversion_dict_path data/pair_metadata_dict.json \
  --material ag
```

> Les chemins dans `saxs_path` et `les_path` doivent être relatifs à `--data_dir`. Vous pouvez contrôler l’existence de
> chaque paire de fichiers avec `scripts/saminitycheck.py` au besoin.

Structure du HDF5 généré (final_output.h5) :

```text
final_output.h5
├── data_q_saxs    [N, pad_size]
├── data_y_saxs    [N, pad_size]
├── data_q_les     [N, pad_size]
├── data_y_les     [N, pad_size]
├── len            [N]
├── valid          [N]
├── csv_index      [N]
├── <metadata_1>   [N]
├── <metadata_2>   [N]
└── ...            ...
```

Une fois la conversion terminée, vous obtenez :

- `data/all_pair_data.h5` prêt pour l’entraînement ;
- `data/pair_metadata_dict.json` contenant vos mappings catégoriels.

### 5. Entraînement du modèle PAIRVAE

L’entraînement du PairVAE se fait de la même manière
que [Entraînement du modèle VAE](#3-entrainement-du-modèle-à-partir-du-fichier-hdf5), mais avec un fichier HDF5
différent et une configuration YAML différente.

TODO

Dans cette exmple `data/all_data.h5` et `data/metadata_dict.json` sont issus de l’étape précédente, et seront filtrés
sur `technique=saxs` et `material=ag`.

### 6. Inference (optionnelle)

Le script `05_infer.py` permet de lancer l'inférence avec les modèles VAE ou PairVAE directement en ligne de commande. 

> **Note** : Lors de l'entrainnement, le modèle est sauvegarde le conversion_dict avec le quel il a été entrainé.
> Lors de l'inference, si vous utilisez un dataset H5, il faut que le conversion_dict soit le même que celui utilisé. Vous pouvez les trouver dans la configuration sauvegarder lors de l'entrainnement.

**Utilisation générale** :
```bash
python scripts/05_infer.py \
  --outputdir <CHEMIN_ENREGISTREMENT> \
  --checkpoint <CHEMIN_CHECKPOINT> \
  --data_path <FICHIER_DONNÉES> \
  [--data_dir <DOSSIER_DONNÉES>] \
  [--mode <MODE_CONVERSION>] \
  [--batch_size <TAILLE_BATCH>]
```

**Arguments principaux** :

| Argument           | Obligatoire | Description                                    |
|--------------------|-------------|------------------------------------------------|
| `-c/--checkpoint`  | ✓ | Chemin vers le fichier de checkpoint (.ckpt)   |
| `-o/--outputdir`  | ✓ | Chemin vers le dossier d'enregistrement  |
| `-d/--data_path`   | ✓ | Chemin vers les données d'entrée (.h5 ou .csv) |
| `-s/--sample_frac`   | ❌  | Fraction du dataset à utiliser (0<s<1) (défaut: 1.0) |
| `--mode`           | PairVAE only | `les_to_saxs` ou `les_to_les` ou `saxs_to_saxs` ou`saxs_to_les` pour le PairVAE |
| `-bs/--batch_size` | ❌ | Taille de batch (défaut: 32)                   |
| `-dd/--data_dir` | ❌ | Chemin vers le dossiers des données txt        |
| `--plot` | ❌ | Booléen indiquant l'enregistrement des signaux au format png       |

**Exemple pour VAE** :
```bash
python scripts/05_infer.py \
  --outputdir dossier_test_vae \
  --checkpoint logs/vae_model.ckpt \
  --data_path data/new_data.h5 \
  --batch_size 64
  --plot
```

**Exemple pour PairVAE** :
```bash
python scripts/05_infer.py \
  --outputdir dossier_test_pairvae \
  --checkpoint logs/pairvae_model.ckpt \
  --data_path data/pair_data.h5 \
  --mode les_to_saxs \
  --batch_size 32
```

#### Sorties

Les prédictions sont sauvegardées dans le dossier `inference_outputs` sous forme de fichiers `.npy` :
- Format : tableau NumPy de shape `(N, 2)` où `N` = longueur de la série temporelle
- Colonne 0 : valeurs prédites (`y`)
- Colonne 1 : vecteur q correspondant (`q`)

```
prediction_12345.npy  # Nom généré à partir de l'index CSV ou du nom du fichier
├── [ [y1, q1],
│     [y2, q2],
│     ...            ]
└── shape (pad_size, 2)
```

#### Formats supportés

| Modèle    | Formats d'entrée | Modes (PairVAE)       |
|-----------|------------------|-----------------------|
| VAE       | `.h5`, `.csv`    | -                     |
| PairVAE   | `.h5`, `.csv`    | `les_to_saxs`, `saxs_to_les` |

> **Note** : Pour le VAE avec des données CSV, assurez-vous que le fichier contient une colonne `path` pointant vers les fichiers `.txt` à prédire et de preciser `--data_dir`.

---

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
