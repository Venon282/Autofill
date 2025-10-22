"""Grid search utilities for sweeping training hyperparameters."""

import copy
import itertools
from typing import Dict, Iterable

from src.logging_utils import get_logger
from src.model.trainer import TrainPipeline


logger = get_logger(__name__)


class GridSearch:
    """Enumerate combinations from ``param_grid`` and launch training runs."""

    def __init__(self, config: Dict) -> None:
        self.base_config = config
        self.param_grid = self._get_param_grid()

    def _get_param_grid(self) -> Dict[str, Iterable]:
        """Return the parameter grid declared in the configuration."""

        param_grid = self.base_config.get("param_grid")
        if not param_grid:
            raise ValueError("No 'param_grid' found in the config file.")
        return param_grid

    def _update_config(self, config: Dict, param_set: Dict[str, object]) -> None:
        """Recursively update nested configuration keys from ``param_set``."""

        for key, value in param_set.items():
            current = config
            parts = key.split(".")
            for part in parts[:-1]:
                current = current[part]
            current[parts[-1]] = value

    def run(self) -> None:
        """Iterate over the parameter grid and invoke :class:`TrainPipeline`."""

        keys, values = zip(*self.param_grid.items())
        total_runs = 1
        for value_list in values:
            total_runs *= len(value_list)
        logger.info("=" * 60)
        logger.info("Starting grid search: %d runs", total_runs)
        logger.info("Parameter grid: %s", self.param_grid)
        logger.info("=" * 60)
        for index, combination in enumerate(itertools.product(*values)):
            logger.info("-" * 60)
            logger.info("Run %d/%d", index + 1, total_runs)
            import torch

            logger.info("GPU available: %s", torch.cuda.is_available())
            param_set = dict(zip(keys, combination))
            logger.info("Parameters: %s", param_set)
            config = copy.deepcopy(self.base_config)
            self._update_config(config, param_set)
            base_run_name = config.get("run_name", "")
            param_str = "_".join(f"{k.split('.')[-1]}={val}" for k, val in param_set.items())
            config["run_name"] = f"{base_run_name}_grid_{index}_{param_str}"
            logger.info("Run name: %s", config["run_name"])
            logger.info("Initializing training pipeline...")
            config.pop("param_grid", None)
            trainer = TrainPipeline(config, verbose=False)
            logger.info(
                "Starting training for run %d/%d", index + 1, total_runs
            )
            trainer.train()
            logger.info("Finished run %d/%d", index + 1, total_runs)
        logger.info("=" * 60)
        logger.info("All grid search runs completed.")
