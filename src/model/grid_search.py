"""Grid search utilities for sweeping training hyperparameters."""

import copy
import itertools
from typing import Dict, Iterable
import torch

from src.model.trainer import make_trainer
from src.logging_utils import get_logger

logger = get_logger(__name__)

#region Grid Search
class GridSearch:
    """Enumerate combinations from param_grid and launch training runs."""

    def __init__(self, config: Dict, show_progressbar= True) -> None:
        self.base_config = config
        self.param_grid = self._get_param_grid()
        self.show_progressbar = show_progressbar

    def _get_param_grid(self) -> Dict[str, Iterable]:
        """Return the parameter grid declared in the configuration."""
        param_grid = self.base_config.get("param_grid")
        if not param_grid:
            raise ValueError("No 'param_grid' found in the config file.")
        return param_grid

    def _update_config(self, config: Dict, param_set: Dict[str, object]) -> None:
        """Recursively update nested configuration keys from param_set."""
        for key, value in param_set.items():
            current = config
            parts = key.split(".")
            for part in parts[:-1]:
                if part not in current:
                    raise KeyError(f"Invalid config path '{key}': missing '{part}' in {list(current.keys())}")
                current = current[part]
            current[parts[-1]] = value

    def _safe_train(self, config: Dict, index: int, total_runs: int) -> None:
        """Safely create and train a trainer, with clear error handling."""
        try:
            logger.info("Initializing training trainer (run %d/%d)...", index + 1, total_runs)
            config.pop("param_grid", None)
            trainer = make_trainer(config, verbose=False, show_progressbar=self.show_progressbar)
            logger.info("GPU available: %s", torch.cuda.is_available())
            logger.info("Starting training for run %d/%d", index + 1, total_runs)
            trainer.train()
            logger.info("Finished run %d/%d successfully", index + 1, total_runs)
        except (KeyError, ValueError, TypeError) as e:
            logger.error("Configuration error in run %d/%d: %s", index + 1, total_runs, e)
        except Exception as e:
            logger.exception("Unexpected error during run %d/%d: %s", index + 1, total_runs, e)

    def run(self) -> None:
        """Iterate over the parameter grid and invoke the appropriate training trainer."""
        keys, values = zip(*self.param_grid.items())
        total_runs = 1
        for value_list in values:
            total_runs *= len(value_list)
        logger.info("=" * 80)
        logger.info("Starting grid search with %d total runs", total_runs)
        logger.info("Parameter grid: %s", self.param_grid)
        logger.info("=" * 80)
        for index, combination in enumerate(itertools.product(*values)):
            logger.info("-" * 80)
            logger.info("Run %d/%d", index + 1, total_runs)
            param_set = dict(zip(keys, combination))
            logger.info("Parameters: %s", param_set)
            config = copy.deepcopy(self.base_config)
            try:
                self._update_config(config, param_set)
            except KeyError as e:
                logger.error("Invalid parameter path: %s. Skipping run.", e)
                continue
            base_run_name = config.get("run_name", "run")
            param_str = "_".join(f"{k.split('.')[-1]}={val}" for k, val in param_set.items())
            config["run_name"] = f"{base_run_name}_grid_{index}_{param_str}"
            self._safe_train(config, index, total_runs)
        logger.info("=" * 80)
        logger.info("All grid search runs completed.")
#endregion
