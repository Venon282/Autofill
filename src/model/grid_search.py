import copy
import itertools

from src.model.trainer import TrainPipeline


class GridSearch:
    def __init__(self, config):
        self.base_config = config
        self.param_grid = self._get_param_grid()

    def _get_param_grid(self):
        param_grid = self.base_config.get("param_grid")
        if not param_grid:
            raise ValueError("No 'param_grid' found in the config file.")
        return param_grid

    def _update_config(self, config, param_set):
        for k, val in param_set.items():
            d = config
            parts = k.split(".")
            for p in parts[:-1]:
                d = d[p]
            d[parts[-1]] = val

    def run(self):
        keys, values = zip(*self.param_grid.items())
        total_runs = 1
        for v in values:
            total_runs *= len(v)
        print("=" * 60)
        print(f"[GridSearch] Starting grid search: {total_runs} runs")
        print(f"[GridSearch] Parameter grid: {self.param_grid}")
        print("=" * 60)
        for i, v in enumerate(itertools.product(*values)):
            print("-" * 60)
            print(f"[GridSearch] Run {i + 1}/{total_runs}")
            import torch
            print(f" GPU AVAILABLE: {torch.cuda.is_available()}")
            param_set = dict(zip(keys, v))
            print(f"[GridSearch] Parameters: {param_set}")
            config = copy.deepcopy(self.base_config)
            self._update_config(config, param_set)
            base_run_name = config.get("run_name", "")
            param_str = "_".join(f"{k.split('.')[-1]}={val}" for k, val in param_set.items())
            config["run_name"] = f"{base_run_name}_grid_{i}_{param_str}"
            print(f"[GridSearch] Run name: {config['run_name']}")
            print(f"[GridSearch] Initializing training pipeline...")
            config.pop("param_grid", None)
            trainer = TrainPipeline(config, verbose=False)
            print(f"[GridSearch] Starting training for run {i + 1}/{total_runs}")
            trainer.train()
            print(f"[GridSearch] Finished run {i + 1}/{total_runs}")
        print("=" * 60)
        print("[GridSearch] All grid search runs completed.")
