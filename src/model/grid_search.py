"""Grid search utilities for sweeping training hyperparameters."""

import copy
import itertools
from typing import Dict, Iterable

from src.model.trainer import TrainPipeline


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
        print("=" * 60)
        print(f"[GridSearch] Starting grid search: {total_runs} runs")
        print(f"[GridSearch] Parameter grid: {self.param_grid}")
        print("=" * 60)
        for index, combination in enumerate(itertools.product(*values)):
            print("-" * 60)
            print(f"[GridSearch] Run {index + 1}/{total_runs}")
            import torch

            print(f" GPU AVAILABLE: {torch.cuda.is_available()}")
            param_set = dict(zip(keys, combination))
            print(f"[GridSearch] Parameters: {param_set}")
            config = copy.deepcopy(self.base_config)
            self._update_config(config, param_set)
            base_run_name = config.get("run_name", "")
            param_str = "_".join(f"{k.split('.')[-1]}={val}" for k, val in param_set.items())
            config["run_name"] = f"{base_run_name}_grid_{index}_{param_str}"
            print(f"[GridSearch] Run name: {config['run_name']}")
            print("[GridSearch] Initializing training pipeline...")
            config.pop("param_grid", None)
            trainer = TrainPipeline(config, verbose=False)
            print(f"[GridSearch] Starting training for run {index + 1}/{total_runs}")
            trainer.train()
            print(f"[GridSearch] Finished run {index + 1}/{total_runs}")
        print("=" * 60)
        print("[GridSearch] All grid search runs completed.")
