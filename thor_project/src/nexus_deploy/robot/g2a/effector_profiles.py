"""Explicit end-effector semantics for the standalone G2A client.

The policy action and the GDK joint target are not assumed to share a direction.
All model outputs are converted once, before filtering, kinematic limiting and
the 100 Hz collision guard.  Those realtime stages therefore operate only in
physical GDK coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np

from .constants import ACTION_DIM, EFFECTOR_CANONICAL_INDICES

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EffectorProfile:
    name: str
    display_name: str
    unit: str
    hardware_state_range: tuple[float, float]
    model_action_range: tuple[float, float]
    model_action_clip_tolerance: float
    action_mapping: str
    state_semantics: str
    action_semantics: str
    default_max_velocity: float
    default_max_acceleration: float
    default_tracking_error: float
    default_relative_target_warning: float
    collision_geometry: str
    collision_geometry_status: str
    saturate_model_actions: bool = False

    def model_actions_to_hardware_targets(self, actions: np.ndarray) -> np.ndarray:
        values = _finite_actions(actions)
        result = values.copy()
        indices = list(EFFECTOR_CANONICAL_INDICES)
        model_low, model_high = self.model_action_range
        requested = result[..., indices]
        tolerance = self.model_action_clip_tolerance
        observed_low = float(np.min(requested))
        observed_high = float(np.max(requested))
        if not self.saturate_model_actions and (
            observed_low < model_low - tolerance or observed_high > model_high + tolerance
        ):
            raise ValueError(
                f"{self.name} model effector action is outside "
                f"[{model_low}, {model_high}]: observed=[{observed_low}, {observed_high}], "
                f"clip_tolerance={tolerance}"
            )
        clipped_count = int(np.count_nonzero((requested < model_low) | (requested > model_high)))
        if self.saturate_model_actions and clipped_count:
            # Once per model response, not once per 100 Hz servo tick. Preserve
            # raw input for traceability; never clip arm/head/waist coordinates.
            logger.warning(
                "%s model effector targets clipped to [%g, %g] %s before direction mapping: "
                "observed=[%.9g, %.9g], clipped_values=%d; action chunk retained",
                self.name, model_low, model_high, self.unit, observed_low, observed_high, clipped_count,
            )
        # Dahuan intentionally saturates every finite predicted opening. Legacy
        # profiles retain their tolerance-based rejection above. Shape and
        # non-finite checks happen before clipping; hardware limits and the
        # kinematic/collision guards still run after direction conversion.
        requested = np.clip(requested, model_low, model_high)
        if self.action_mapping == "identity":
            hardware = requested
        elif self.action_mapping == "invert_about_endpoints":
            hardware_low, hardware_high = self.hardware_state_range
            ratio = (requested - model_low) / (model_high - model_low)
            hardware = hardware_high - ratio * (hardware_high - hardware_low)
        else:  # Defensive: profiles are static but this must remain fail-closed.
            raise ValueError(f"Unsupported effector action mapping: {self.action_mapping}")
        result[..., indices] = hardware
        return result.astype(np.float32, copy=False)

    def hardware_targets_to_model_actions(self, targets: np.ndarray) -> np.ndarray:
        values = _finite_actions(targets)
        result = values.copy()
        indices = list(EFFECTOR_CANONICAL_INDICES)
        hardware_low, hardware_high = self.hardware_state_range
        requested = np.clip(result[..., indices], hardware_low, hardware_high)
        if self.action_mapping == "identity":
            model = requested
        elif self.action_mapping == "invert_about_endpoints":
            model_low, model_high = self.model_action_range
            ratio = (requested - hardware_low) / (hardware_high - hardware_low)
            model = model_high - ratio * (model_high - model_low)
        else:
            raise ValueError(f"Unsupported effector action mapping: {self.action_mapping}")
        result[..., indices] = model
        return result.astype(np.float32, copy=False)


def _finite_actions(actions: np.ndarray) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float32)
    if (values.ndim not in {1, 2} or values.shape[-1] != ACTION_DIM
            or not values.size or not np.isfinite(values).all()):
        raise ValueError(f"Expected finite (..., {ACTION_DIM}) actions, got {values.shape}")
    return values


EFFECTOR_PROFILES = {
    "dahuan": EffectorProfile(
        name="dahuan",
        display_name="大寰夹爪",
        unit="m",
        hardware_state_range=(0.0, 0.025),
        model_action_range=(0.0, 0.025),
        model_action_clip_tolerance=0.001,
        action_mapping="invert_about_endpoints",
        state_semantics="0=闭合，0.025=张开",
        action_semantics="0=张开指令，0.025=闭合指令",
        default_max_velocity=0.05,
        default_max_acceleration=0.2,
        default_tracking_error=0.01,
        default_relative_target_warning=0.025,
        collision_geometry="dahuan",
        collision_geometry_status="provisional",
        saturate_model_actions=True,
    ),
    "omnipicker_legacy": EffectorProfile(
        name="omnipicker_legacy",
        display_name="智元原装 Omnipicker 夹爪",
        unit="rad",
        hardware_state_range=(-0.785, 0.0),
        model_action_range=(-0.785, 0.0),
        model_action_clip_tolerance=0.02,
        action_mapping="identity",
        state_semantics="沿用旧模型/GDK 关节位置语义",
        action_semantics="模型目标与 GDK 目标同向",
        default_max_velocity=1.0,
        default_max_acceleration=4.0,
        default_tracking_error=0.5,
        default_relative_target_warning=0.05,
        collision_geometry="omnipicker",
        collision_geometry_status="verified",
    ),
}

EFFECTOR_PROFILE_ALIASES = {"dh": "dahuan"}
EFFECTOR_PROFILE_CHOICES = (*EFFECTOR_PROFILES, *EFFECTOR_PROFILE_ALIASES)


def get_effector_profile(name: str) -> EffectorProfile:
    canonical_name = EFFECTOR_PROFILE_ALIASES.get(str(name), str(name))
    try:
        return EFFECTOR_PROFILES[canonical_name]
    except KeyError as error:
        raise ValueError(
            f"Unknown effector profile {name!r}; choices={tuple(EFFECTOR_PROFILES)}"
        ) from error
