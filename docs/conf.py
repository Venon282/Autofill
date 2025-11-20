"""Configuration file for the Sphinx documentation builder."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Ensure the project root is on sys.path so 'src' is importable as a package
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

project = "AutoFill"
author = "AutoFill Contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

autodoc_typehints = "description"
autodoc_mock_imports = [
    "lightning",
    "lightning.pytorch",
    "mlflow",
    "torch",
    "h5py",
    "numpy",
    "pandas",
    "sklearn",
    "matplotlib",
    "seaborn",
    "tqdm",
    "torch",
    "yaml",
    "hydra",
    "uniqpath",
    "omegaconf",
    "lmfit",
    "sasmodels",
    "sasmodels.core",
    "sasmodels.data",
    "sasmodels.direct_model",
]
