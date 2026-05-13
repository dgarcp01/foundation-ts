from pathlib import Path
from typing import Any

from foundation_ts.models.base import BaseTimeSeriesModel


class MomentModel(BaseTimeSeriesModel):
    """Wrapper for the MOMENT time-series foundation model."""

    def __init__(
        self,
        model_id: str = "AutonLab/MOMENT-1-large",
        task_name: str = "embedding",
        model_kwargs: dict[str, Any] | None = None,
        device: str | None = None,
        channel_first: bool = False,
        output_key: str | None = None,
    ):
        self.model_id = model_id
        self.task_name = task_name      #Indicates the task to solve with moment
        self.model_kwargs = dict(model_kwargs or {})
        self.device = device
        self.channel_first = channel_first 
        self.output_key = output_key        # Moment model returns an object. Output_key indicates the key of the object to be returned 

        self.model = self._load_moment_pipeline()

    def predict(self, x: Any) -> Any:
        x_enc = self._prepare_input(x)

        self.model.eval()
        outputs = self.model(x_enc=x_enc)

        if self.output_key is None:
            return outputs

        if not hasattr(outputs, self.output_key):
            raise AttributeError(f"MOMENT output does not expose '{self.output_key}'.")

        return getattr(outputs, self.output_key)

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> "MomentModel":
        return cls(model_id=str(path), **kwargs)

    def _load_moment_pipeline(self) -> Any:
        try:
            from momentfm import MOMENTPipeline
        except ImportError as exc:
            raise ImportError(
                "MomentModel requires the 'momentfm' package. "
                "Install it with `pip install momentfm` or recreate the conda environment."
            ) from exc

        model_kwargs = {"task_name": self.task_name, **self.model_kwargs}
        model = MOMENTPipeline.from_pretrained(
            self.model_id,
            model_kwargs=model_kwargs,
        )
        model.init()

        if self.device is not None:
            model = model.to(self.device)

        return model

    def _prepare_input(self, x: Any) -> Any:
        if self.device is not None and hasattr(x, "to"):
            x = x.to(self.device)

        if self.channel_first:
            return x

        if hasattr(x, "ndim") and x.ndim == 3:
            return x.transpose(1, 2)

        return x
