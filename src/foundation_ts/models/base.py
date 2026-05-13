from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseTimeSeriesModel(ABC):
    """Common interface for trainable and pretrained time-series models."""

    @abstractmethod
    def predict(self, x: Any) -> Any:
        """Run inference on a batch of time-series samples."""
        raise NotImplementedError

    def fit(self, train_data: Any, val_data: Any | None = None) -> None:
        """Train the model when the implementation supports it."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support training.")

    def save(self, path: str | Path) -> None:
        """Persist the model when the implementation supports it."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support saving.")

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> "BaseTimeSeriesModel":
        """Load a model from disk when the implementation supports it."""
        raise NotImplementedError(f"{cls.__name__} does not support loading.")
