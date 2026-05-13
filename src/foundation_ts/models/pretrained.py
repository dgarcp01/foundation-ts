from pathlib import Path
from typing import Any, Callable

from foundation_ts.models.base import BaseTimeSeriesModel


class PretrainedTimeSeriesModel(BaseTimeSeriesModel):
    """Adapter for already-trained models with a predict-like callable."""

    def __init__(self, model: Any, predict_fn: Callable[[Any], Any] | None = None):
        self.model = model
        self.predict_fn = predict_fn

    def predict(self, x: Any) -> Any:
        if self.predict_fn is not None:
            return self.predict_fn(x)

        if hasattr(self.model, "predict"):
            return self.model.predict(x)

        if callable(self.model):
            return self.model(x)

        raise TypeError("The wrapped pretrained model must be callable or expose predict().")

    @classmethod
    def load(
        cls,
        path: str | Path,
        loader: Callable[[str | Path], Any],
        predict_fn: Callable[[Any], Any] | None = None,
        **_: Any,
    ) -> "PretrainedTimeSeriesModel":
        model = loader(path)
        return cls(model=model, predict_fn=predict_fn)
