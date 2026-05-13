from collections.abc import Mapping
from typing import Any

from foundation_ts.models.base import BaseTimeSeriesModel
from foundation_ts.models.moment import MomentModel
from foundation_ts.models.pretrained import PretrainedTimeSeriesModel


MODEL_REGISTRY: dict[str, type[BaseTimeSeriesModel]] = {
    "moment": MomentModel,
    "pretrained": PretrainedTimeSeriesModel,
}


def register_model(name: str, model_cls: type[BaseTimeSeriesModel]) -> None:
    if not name:
        raise ValueError("Model name cannot be empty.")

    if not issubclass(model_cls, BaseTimeSeriesModel):
        raise TypeError("model_cls must inherit from BaseTimeSeriesModel.")

    MODEL_REGISTRY[name] = model_cls


def build_model(name: str, params: Mapping[str, Any] | None = None) -> BaseTimeSeriesModel:
    if not name:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Model name is required. Available models: {available}")

    try:
        model_cls = MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Available models: {available}") from exc

    return model_cls(**dict(params or {}))
