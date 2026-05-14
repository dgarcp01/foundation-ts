from pathlib import Path
from random import seed as set_python_seed

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from foundation_ts.data import load_dataicann, make_dataicann_windows
from foundation_ts.models import build_model
from foundation_ts.data import load_dataicann, make_dataicann_windows
import torch
import pdb




def setup_seed(seed: int) -> None:
    """Set the base seed for reproducible examples."""
    set_python_seed(seed)
    torch.manual_seed(seed)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_seed(cfg.project.seed)

    output_dir = Path.cwd()
    data_path = Path(to_absolute_path(str(cfg.data.path)))
    mat_file_path = data_path if data_path.suffix == ".mat" else data_path / "dataicann.mat"

    window_size = int(cfg.data.get("window_size", 512))
    stride = int(cfg.data.get("stride", window_size))

    df = load_dataicann(mat_file_path)
    windows = make_dataicann_windows(
        df,
        window_size=window_size,
        stride=stride,
    )
    x = torch.as_tensor(windows, dtype=torch.float32)

    model_params = OmegaConf.to_container(cfg.model.params, resolve=True)
    model = build_model(cfg.model.name, model_params)
    #pdb.set_trace()
    y = model.predict(x)
    print(y.shape)

    print(f"Project: {cfg.project.name}")
    print(f"Seed: {cfg.project.seed}")
    print(f"Hydra output directory: {output_dir}")
    print(f"Input windows: {x.shape}")
    print(f"Model output: {getattr(y, 'shape', type(y).__name__)}")
    print()
    print("Loaded configuration:")
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
