# AutoFill — Toolbox VAE / PairVAE

AutoFill regroupe les scripts nécessaires pour prétraiter les fichiers expérimentaux, entraîner les modèles **VAE** et **PairVAE**, et analyser les résultats.

---

## Arborescence du projet

```

AutoFill/
├─ src/
│  ├─ dataset/          # gestion et préparation des données
│  ├─ model/            # architectures et entraînement des modèles
│  └─ scripts/          # pipeline CLI : prétraitement, conversion, entraînement
├─ configs/
│  ├─ vae.yml           # configuration du modèle VAE
│  └─ pairvae.yml       # configuration du modèle PairVAE
├─ docs/                # documentation Sphinx
├─ requirements.txt     # dépendances de base
└─ README.md            # ce guide

````

---

## Installation

### 1. Prérequis

- Python **3.8+**
- Git
> `uv` peut gérer automatiquement l’installation de la bonne version de Python.

---

### 2. Cloner le dépôt

```bash
git clone https://github.com/JulienRabault/AutoFill.git
cd AutoFill
````

---

### 3. Installer `uv`

**Sur Linux :**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Sur Windows (PowerShell) :**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

### 4. Installer les dépendances

```bash
uv sync --no-dev
```

Cela crée automatiquement un environnement virtuel `.venv` et installe les dépendances du projet.

---

### 5. (Optionnel) Utiliser `venv` manuellement

Si vous préférez une installation classique :

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

> Sur Windows :
> `.\env\Scripts\activate`

Pour les commandes suivantes, remplacez simplement `uv run` par `python`.

---

## Construire la documentation avec uv

`uv` simplifie la gestion des dépendances de documentation (Sphinx).

### 1. Installer les dépendances de documentation

```bash
uv sync --extra docs
```

Équivalent à :

```bash
pip install -e ".[docs]"
```

### 2. Activer l’environnement

```bash
source .venv/bin/activate
```

### 3. Générer la documentation

```bash
sphinx-build -b html docs docs/_build/html
```

### 4. Consulter la documentation

Ouvrir :

```
docs/_build/html/index.html
```

---

## Licence

Ce projet est distribué sous licence **Apache 2.0**.
Voir le fichier `LICENSE` pour plus d’informations.


### Contact

Pour toute question ou problème, n’hésitez pas à contacter :

- **Julien Rabault** (julien.rabault@irit.fr)
- **Caroline de Pourtalès** (caroline.de-pourtales@irit.fr)
