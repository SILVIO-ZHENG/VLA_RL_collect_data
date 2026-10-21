"""Plugin interfaces and dependency-free ``module:object`` loading."""

from __future__ import annotations

import importlib
import inspect
from abc import ABC, abstractmethod
from typing import Any, Mapping

from .contracts import ActionChunk, InferenceRequest, Observation, PolicyCapabilities, TensorSpec


class PluginError(RuntimeError):
    pass


class ModelAdapter(ABC):
    """The only interface a VLA implementation must provide."""

    @property
    @abstractmethod
    def capabilities(self) -> PolicyCapabilities: ...

    def load(self) -> None:
        """Load weights and allocate device resources before serving."""

    @abstractmethod
    def infer(self, request: InferenceRequest) -> ActionChunk: ...

    def reset(self, session_id: str) -> None:
        """Clear optional per-session recurrent/RTC state."""

    def close(self) -> None:
        """Release optional model resources."""


class RobotAdapter(ABC):
    """The only interface a robot implementation must provide."""

    @property
    @abstractmethod
    def state_spec(self) -> TensorSpec: ...

    @property
    @abstractmethod
    def action_spec(self) -> TensorSpec: ...

    @property
    @abstractmethod
    def camera_names(self) -> tuple[str, ...]: ...

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def observe(self, task: str) -> Observation: ...

    @abstractmethod
    def execute(self, action: Any) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


def load_plugin(target: str, config: Mapping[str, Any] | None = None) -> Any:
    """Instantiate ``package.module:Class`` with one configuration mapping."""
    module_name, separator, attribute = target.partition(":")
    if not separator or not module_name or not attribute:
        raise PluginError(f"Plugin must use 'module:object' syntax, got {target!r}")
    try:
        factory = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as error:
        raise PluginError(f"Cannot load plugin {target!r}: {error}") from error
    if not callable(factory):
        raise PluginError(f"Plugin target is not callable: {target!r}")
    parameters = inspect.signature(factory).parameters
    try:
        return factory(dict(config or {})) if parameters else factory()
    except Exception as error:
        raise PluginError(f"Could not initialize plugin {target!r}: {error}") from error
