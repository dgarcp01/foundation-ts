from pathlib import Path
from random import seed as set_python_seed

import hydra
from omegaconf import DictConfig, OmegaConf


def setup_seed(seed: int) -> None:
    """Set the base seed for reproducible examples."""
    set_python_seed(seed)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_seed(cfg.project.seed)

    output_dir = Path.cwd()

    print(f"Project: {cfg.project.name}")
    print(f"Seed: {cfg.project.seed}")
    print(f"Hydra output directory: {output_dir}")
    print()
    print("Loaded configuration:")
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()

