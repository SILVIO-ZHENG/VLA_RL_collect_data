#!/usr/bin/env python
"""PI0.5 policy adapters exposed by the G2A WebSocket server."""

from __future__ import annotations

import gc
import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.configs import PreTrainedConfig, RTCAttentionSchedule
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc import reanchor_relative_rtc_prefix
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.processor import NormalizerProcessorStep, RelativeActionsProcessorStep, TransitionKey, create_transition
from lerobot.utils.constants import ACTION, OBS_STATE

from nexus_deploy.robot.g2a.command_prefix import validate_command_prefix
from nexus_deploy.robot.g2a.constants import CANONICAL_STATE_NAMES

from .profiles import infer_profile
from .protocol import PROTOCOL_NAME, PROTOCOL_VERSION, ProtocolError, validate_observation

logger = logging.getLogger(__name__)

CANONICAL_STATE_DIM = 24
CANONICAL_ACTION_DIM = 24
CANONICAL_ACTION_NAMES = tuple(CANONICAL_STATE_NAMES)
CAMERA_KEYS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
HEAD_WAIST_SLICE = slice(16, 24)


def _processor_fingerprint(checkpoint: Path) -> str | None:
    """Identify the on-disk action processors without changing model loading."""
    digest = hashlib.sha256()
    statistics = sorted(checkpoint.glob("policy_*processor_step_*.safetensors"))
    if not statistics:
        return None
    files = [checkpoint / "policy_preprocessor.json", checkpoint / "policy_postprocessor.json", *statistics]
    for path in files:
        if not path.is_file() or path.is_symlink():
            return None
        payload = path.read_bytes()
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _native_action_sha256(actions: np.ndarray) -> str:
    payload = np.ascontiguousarray(actions.astype("<f4", copy=False)).tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class _SessionState:
    next_chunk_number: int = 0
    chunk_id: str | None = None
    raw_actions: torch.Tensor | None = None
    processed_native_actions: torch.Tensor | None = None
    last_inference_s: float | None = None
    rtc_reanchor_mode: str = "none"
    rtc_reanchor_max_abs_rad: float = 0.0
    rtc_prefix_source: str = "model_tail"
    rtc_prefix_committed_steps: int | None = None


def select_native_state(state_24d: np.ndarray, native_dim: int) -> np.ndarray:
    if state_24d.shape != (CANONICAL_STATE_DIM,):
        raise ValueError(f"Expected canonical 24D state, got {state_24d.shape}")
    if native_dim not in (16, 24):
        raise ValueError(f"Unsupported native state dimension: {native_dim}")
    return np.ascontiguousarray(state_24d[:native_dim], dtype=np.float32)


def expand_actions_to_canonical(
    native_actions: np.ndarray,
    current_state_24d: np.ndarray,
    *,
    command_head_waist: bool,
) -> np.ndarray:
    """Return a canonical 24D absolute chunk, holding head/waist when requested."""
    actions = np.asarray(native_actions, dtype=np.float32)
    state = np.asarray(current_state_24d, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[-1] not in (16, 24):
        raise ValueError(f"Expected [T, 16|24] actions, got {actions.shape}")
    if state.shape != (CANONICAL_STATE_DIM,):
        raise ValueError(f"Expected canonical 24D state, got {state.shape}")

    if actions.shape[-1] == 16:
        canonical = np.empty((len(actions), CANONICAL_ACTION_DIM), dtype=np.float32)
        canonical[:, :16] = actions
        canonical[:, HEAD_WAIST_SLICE] = state[HEAD_WAIST_SLICE]
    else:
        canonical = actions.copy()
        if not command_head_waist:
            canonical[:, HEAD_WAIST_SLICE] = state[HEAD_WAIST_SLICE]
    return canonical


class G2APi05Policy:
    """Load one PI0.5 checkpoint and expose canonical absolute G2A chunks."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        device: str,
        model_fps: float,
        execution_horizon: int,
        max_guidance_weight: float,
        prefix_attention_schedule: RTCAttentionSchedule,
        command_head_waist: bool,
        model_profile: str | None = None,
        fixed_noise_seed: int | None = None,
        fast_model_load: bool = True,
        effector_profile: str = "omnipicker_legacy",
        tokenizer_path: str | Path | None = None,
        rtc_inference_mode: str = "guided",
    ) -> None:
        self.checkpoint = checkpoint.resolve()
        self.device = device
        self.model_fps = model_fps
        self.command_head_waist = command_head_waist
        self.fixed_noise_seed = fixed_noise_seed
        self.fast_model_load = bool(fast_model_load)
        self.effector_profile = "dahuan" if str(effector_profile) == "dh" else str(effector_profile)
        self.model_load_strategy = "not_loaded"
        if rtc_inference_mode not in ("guided", "trained"):
            raise ValueError(f"Unsupported RTC inference mode: {rtc_inference_mode!r}")
        self.rtc_inference_mode = rtc_inference_mode
        self._lock = threading.Lock()
        self._sessions: dict[str, _SessionState] = {}

        config = PreTrainedConfig.from_pretrained(self.checkpoint, local_files_only=True)
        if config.type != "pi05":
            raise ValueError(f"Expected a pi05 checkpoint, got policy type {config.type!r}")
        self.rtc_training_max_delay = getattr(config, "rtc_training_max_delay", 0)
        if self.rtc_inference_mode == "trained":
            delay = self.rtc_training_max_delay
            chunk_size = int(config.chunk_size)
            if type(delay) is not int or delay <= 0:
                raise ValueError("Trained RTC requires a checkpoint with rtc_training_max_delay > 0")
            if not delay <= execution_horizon <= chunk_size - delay:
                raise ValueError(
                    "Trained RTC requires rtc_training_max_delay <= execution_horizon <= "
                    "chunk_size - rtc_training_max_delay; "
                    f"got delay={delay}, horizon={execution_horizon}, chunk_size={chunk_size}"
                )
        if tokenizer_path is not None:
            resource = Path(tokenizer_path).expanduser()
            if not resource.is_dir():
                raise ValueError(f"Tokenizer directory does not exist: {resource}")
            config.tokenizer_name = str(resource.resolve())
        config.device = device
        if hasattr(config, "gradient_checkpointing"):
            config.gradient_checkpointing = False
        if hasattr(config, "compile_model"):
            config.compile_model = False

        self.rtc_config = RTCConfig(
            enabled=True,
            execution_horizon=execution_horizon,
            max_guidance_weight=max_guidance_weight,
            prefix_attention_schedule=prefix_attention_schedule,
        )
        config.rtc_config = self.rtc_config

        self.config = config
        self.native_action_dim = int(config.output_features[ACTION].shape[0])
        if self.native_action_dim not in (16, 24):
            raise ValueError(
                f"G2A deployment only supports 16D or 24D policies, got {self.native_action_dim}"
            )
        self.native_action_names = tuple(config.action_feature_names or ())
        expected_names = CANONICAL_ACTION_NAMES[: self.native_action_dim]
        if self.native_action_names and self.native_action_names != expected_names:
            raise ValueError(
                "Checkpoint action ordering does not match the canonical G2A ordering: "
                f"expected={expected_names}, actual={self.native_action_names}"
            )

        policy_load_started_at = time.perf_counter()
        self.policy = self._load_policy()
        logger.info(
            "PI0.5 policy load complete strategy=%s elapsed=%.1fs",
            self.model_load_strategy,
            time.perf_counter() - policy_load_started_at,
        )
        self._fixed_noise = self._make_fixed_noise()
        preprocessor_overrides = {"device_processor": {"device": device}}
        if getattr(config, "tokenizer_name", None):
            preprocessor_overrides["tokenizer_processor"] = {
                "tokenizer_name": str(config.tokenizer_name),
            }
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config,
            pretrained_path=self.checkpoint,
            preprocessor_overrides=preprocessor_overrides,
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self._relative_step = next(
            (
                step
                for step in self.preprocessor.steps
                if isinstance(step, RelativeActionsProcessorStep) and step.enabled
            ),
            None,
        )
        self._normalizer_step = next(
            (step for step in self.preprocessor.steps if isinstance(step, NormalizerProcessorStep)),
            None,
        )
        if bool(config.use_relative_actions) != (self._relative_step is not None):
            raise RuntimeError("Checkpoint relative-action config and preprocessor do not agree")

        inferred_profile = infer_profile(
            self.native_action_dim,
            bool(config.use_relative_actions),
            bool(config.use_peft),
        )
        expected_inferred_profile = (
            "16d_absolute_full" if model_profile == "16d_absolute_full_aug" else model_profile
        )
        if expected_inferred_profile is not None and expected_inferred_profile != inferred_profile:
            raise ValueError(
                "Selected model profile does not match checkpoint configuration: "
                f"selected={model_profile!r}, inferred={inferred_profile!r}"
            )
        if model_profile == "16d_absolute_full_aug" and not bool(
            getattr(config, "use_openpi_image_augmentation", False)
        ):
            raise ValueError(
                "Profile '16d_absolute_full_aug' requires a checkpoint trained with "
                "use_openpi_image_augmentation=true"
            )
        self.profile = model_profile or inferred_profile
        logger.info(
            "Loaded %s checkpoint=%s native_dim=%d relative=%s rtc_available=true fixed_noise_seed=%s",
            self.profile,
            self.checkpoint,
            self.native_action_dim,
            bool(config.use_relative_actions),
            fixed_noise_seed,
        )

    def _make_fixed_noise(self) -> torch.Tensor | None:
        if self.fixed_noise_seed is None:
            return None
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.fixed_noise_seed)
        shape = (1, int(self.config.chunk_size), int(self.config.max_action_dim))
        noise = torch.randn(
            shape,
            generator=generator,
            dtype=torch.float32,
            device=self.device,
        )
        logger.warning(
            "Fixed PI0.5 inference noise is enabled for diagnosis: seed=%d shape=%s",
            self.fixed_noise_seed,
            shape,
        )
        return noise

    def _load_policy(self):
        policy_class = get_policy_class(self.config.type)
        if self.config.use_peft:
            from peft import PeftConfig, PeftModel

            peft_config = PeftConfig.from_pretrained(self.checkpoint)
            base_path = peft_config.base_model_name_or_path
            if not base_path:
                raise ValueError("LoRA adapter_config.json does not define base_model_name_or_path")
            logger.info("Loading PI0.5 base model from %s", base_path)
            policy = policy_class.from_pretrained(
                base_path,
                config=self.config,
                local_files_only=True,
            )
            logger.info("Loading LoRA adapter from %s", self.checkpoint)
            policy = PeftModel.from_pretrained(
                policy,
                str(self.checkpoint),
                config=peft_config,
                is_trainable=False,
            )
            self.model_load_strategy = "peft_legacy"
        else:
            if self.fast_model_load:
                try:
                    policy = policy_class.from_pretrained(
                        self.checkpoint,
                        config=self.config,
                        local_files_only=True,
                        deployment_fast_load=True,
                        load_device=self.device,
                    )
                    self.model_load_strategy = "meta_assign"
                except Exception:
                    logger.exception(
                        "PI0.5 fast model load failed; retrying the verified legacy loader"
                    )
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    policy = policy_class.from_pretrained(
                        self.checkpoint,
                        config=self.config,
                        local_files_only=True,
                    )
                    self.model_load_strategy = "legacy_fallback"
            else:
                policy = policy_class.from_pretrained(
                    self.checkpoint,
                    config=self.config,
                    local_files_only=True,
                )
                self.model_load_strategy = "legacy"

        if self.rtc_config.enabled:
            policy.config.rtc_config = self.rtc_config
            if hasattr(policy, "init_rtc_processor"):
                policy.init_rtc_processor()
        policy.to(self.device)
        policy.eval()
        # RTC computes guidance gradients with respect to the denoising input,
        # never with respect to model weights.  Freezing deployment weights
        # avoids retaining parameter gradients when RTC temporarily re-enables
        # autograd inside its denoising step.
        policy.requires_grad_(False)
        return policy

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "kind": "metadata",
            "protocol": {"name": PROTOCOL_NAME, "version": PROTOCOL_VERSION},
            "checkpoint": str(self.checkpoint),
            "checkpoint_id": str(self.checkpoint),
            "checkpoint_stats_sha256": _processor_fingerprint(self.checkpoint),
            "processor_version": "nexus-pi05-g2a/v1",
            "model_profile": self.profile,
            "policy_type": self.config.type,
            "model_load_strategy": self.model_load_strategy,
            "native_action_dim": self.native_action_dim,
            "native_action_names": list(
                self.native_action_names or CANONICAL_ACTION_NAMES[: self.native_action_dim]
            ),
            "canonical_state_dim": CANONICAL_STATE_DIM,
            "canonical_action_dim": CANONICAL_ACTION_DIM,
            "canonical_action_names": list(CANONICAL_ACTION_NAMES),
            "action_space": "absolute",
            "native_action_space": "absolute_unnormalized",
            "action_capture": {
                "available": True,
                "native_action_space": "absolute_unnormalized",
                "native_action_dim": self.native_action_dim,
                "content_sha256": True,
            },
            "effector_profile": self.effector_profile,
            "chunk_size": int(self.config.chunk_size),
            # Informational training-time cadence.  Wall-clock action timing,
            # stale-step conversion, and buffer scheduling belong to the
            # robot client and use its independently selected execution FPS.
            "model_fps": self.model_fps,
            "camera_keys": list(CAMERA_KEYS),
            "image_transport": {
                "encoded_payloads": True,
                "encodings": ["jpeg", "raw"],
                "decoded_format": "HWC uint8 RGB",
            },
            "head_waist_locked": self.native_action_dim == 16 or not self.command_head_waist,
            "relative_actions_internal": bool(self.config.use_relative_actions),
            "sampling": {
                "num_inference_steps": int(self.config.num_inference_steps),
                "fixed_noise": self._fixed_noise is not None,
                "fixed_noise_seed": self.fixed_noise_seed,
            },
            "rtc": {
                "available": True,
                "request_controlled": True,
                "execution_horizon": self.rtc_config.execution_horizon,
                "inference_mode": self.rtc_inference_mode,
                "training_max_delay": self.rtc_training_max_delay,
                "command_prefix_available": self.rtc_inference_mode == "trained",
                "max_guidance_weight": self.rtc_config.max_guidance_weight,
                "prefix_attention_schedule": self.rtc_config.prefix_attention_schedule.value,
            },
        }

    def reset(self, session_id: str) -> None:
        if not session_id:
            raise ProtocolError("`session_id` is required")
        with self._lock:
            self._sessions[session_id] = _SessionState()
            self.policy.reset()
            self.preprocessor.reset()
            self.postprocessor.reset()

    def drop_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def _prepare_observation(
        self,
        state_24d: np.ndarray,
        images: dict[str, np.ndarray],
        task: str,
    ) -> dict[str, Any]:
        native_state = select_native_state(state_24d, self.native_action_dim)
        observation: dict[str, Any] = {
            OBS_STATE: torch.from_numpy(native_state),
            "task": task,
        }
        for camera_key, image in images.items():
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).to(dtype=torch.float32) / 255.0
            observation[f"observation.images.{camera_key}"] = image_tensor.contiguous()
        return observation

    def _rtc_prefix(
        self,
        session: _SessionState,
        control: dict[str, Any],
        current_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, int, bool, int, int, int, str | None]:
        enable_request_rtc = bool(control.get("enable_rtc", False))
        session.rtc_reanchor_mode = "none"
        session.rtc_reanchor_max_abs_rad = 0.0
        session.rtc_prefix_source = "model_tail"
        session.rtc_prefix_committed_steps = None
        if not enable_request_rtc:
            return None, 0, False, 0, 0, 0, "request_disabled"
        if (
            session.chunk_id is None
            or session.raw_actions is None
            or session.processed_native_actions is None
            or control.get("active_chunk_id") != session.chunk_id
        ):
            return None, 0, False, 0, 0, 0, "no_matching_previous_chunk"

        next_index = int(control.get("next_action_index", 0))
        next_index = max(0, min(next_index, len(session.raw_actions)))
        prefix = session.raw_actions[next_index:].clone().to(self.device)
        if prefix.numel() == 0:
            return None, 0, False, 0, 0, 0, "empty_previous_chunk"
        source_steps = len(prefix)
        if self.rtc_inference_mode == "trained":
            requested_delay = control.get("estimated_delay_steps", 0)
            if (
                type(requested_delay) is not int
                or requested_delay < 0
                or requested_delay > self.rtc_training_max_delay
            ):
                raise ProtocolError(
                    "Trained RTC estimated_delay_steps must be an integer within "
                    f"[0, {self.rtc_training_max_delay}], got {requested_delay!r}"
                )
            if requested_delay > min(source_steps, self.rtc_config.execution_horizon):
                raise ProtocolError(
                    "Trained RTC previous chunk has fewer real actions than estimated_delay_steps"
                )

        command_packet = control.get("command_prefix")
        if command_packet is not None:
            if self.rtc_inference_mode != "trained" or self._relative_step is not None:
                raise ProtocolError("Command-prefix conditioning requires absolute-action trained RTC")
            try:
                physical_numpy, committed = validate_command_prefix(
                    command_packet, delay_steps=requested_delay, native_dim=self.native_action_dim)
                if (command_packet.get("active_chunk_id") != control.get("active_chunk_id")
                        or command_packet.get("next_action_index") != next_index):
                    raise ValueError("Command-prefix anchor identity mismatch")
                origin = command_packet.get("time_origin")
                if not isinstance(origin, (int, float)) or not np.isfinite(origin) or origin < 0:
                    raise ValueError("Invalid command-prefix clock")
                if self.effector_profile == "dahuan" and (
                        np.any(physical_numpy[:, [7, 15]] < -1e-7)
                        or np.any(physical_numpy[:, [7, 15]] > .025001)):
                    raise ValueError("Command-prefix gripper action outside model range")
            except (ValueError, TypeError) as error:
                raise ProtocolError(str(error)) from error
            physical = torch.from_numpy(physical_numpy)
            transition = create_transition(action=physical)
            if self._normalizer_step is not None:
                transition = self._normalizer_step(transition)
            prefix = transition[TransitionKey.ACTION].to(self.device)
            session.rtc_reanchor_mode = "robot_command_preview"
            session.rtc_prefix_source = "robot_command_preview"
            session.rtc_prefix_committed_steps = committed
        elif self._relative_step is not None:
            current_state = self._relative_step.get_cached_state()
            processed_leftover = session.processed_native_actions[next_index:]
            if current_state is None or processed_leftover.numel() == 0:
                return None, 0, False, source_steps, 0, 0, "relative_reanchor_unavailable"
            prefix = reanchor_relative_rtc_prefix(
                prev_actions_absolute=processed_leftover,
                current_state=current_state,
                relative_step=self._relative_step,
                normalizer_step=self._normalizer_step,
                policy_device=self.device,
            )
            session.rtc_reanchor_mode = "relative_current_state"
        elif current_state is not None:
            # Absolute-action RTC used to feed the untouched previous model
            # tail back into the denoiser.  Once the robot-side kinematic
            # limiter falls behind that tail, every later request is guided by
            # a trajectory which the robot never actually reached.  Translate
            # the arm-only physical prefix onto the state captured for this
            # request, then normalize it back into model space.  Preserve the
            # gripper events exactly: shifting an opening target would change
            # grasp/release timing and semantics.
            physical = session.processed_native_actions[next_index:].detach().cpu().clone()
            state = current_state.detach().cpu().reshape(-1)[: physical.shape[-1]]
            if state.numel() != physical.shape[-1]:
                return None, 0, False, source_steps, 0, 0, "absolute_reanchor_state_mismatch"
            arm_indices = [index for index in range(min(16, physical.shape[-1])) if index not in (7, 15)]
            if arm_indices:
                indices = torch.as_tensor(arm_indices, dtype=torch.long)
                offset = state[indices] - physical[0, indices]
                physical[:, indices] += offset
                session.rtc_reanchor_max_abs_rad = float(offset.abs().max().item())
            transition = create_transition(action=physical)
            if self._normalizer_step is not None:
                transition = self._normalizer_step(transition)
            prefix = transition[TransitionKey.ACTION].to(self.device)
            session.rtc_reanchor_mode = "absolute_current_state"

        # Keep only real previous-chunk actions. RTCProcessor already pads a
        # short tensor internally *after* clamping its effective horizon to the
        # real prefix length. Padding here would make synthetic zeros look like
        # valid guidance and creates a discontinuity at the padding boundary.
        prefix = prefix[: self.rtc_config.execution_horizon]
        effective_horizon = (min(source_steps, self.rtc_config.execution_horizon)
                             if command_packet is not None else len(prefix))
        estimated_delay = (
            requested_delay
            if self.rtc_inference_mode == "trained"
            else max(0, min(int(control.get("estimated_delay_steps", 0)), effective_horizon))
        )
        blend_steps = max(0, effective_horizon - estimated_delay)
        min_blend_steps = max(0, int(control.get("min_rtc_blend_steps", 0)))
        if blend_steps < min_blend_steps:
            return (
                None,
                0,
                False,
                source_steps,
                effective_horizon,
                blend_steps,
                "insufficient_blend_steps",
            )
        return prefix, estimated_delay, True, source_steps, effective_horizon, blend_steps, None

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = str(request.get("request_id", ""))
        session_id = str(request.get("session_id", ""))
        if not request_id or not session_id:
            raise ProtocolError("Both `request_id` and `session_id` are required")

        state_24d, images, task = validate_observation(
            request.get("observation"),
            state_dim=CANONICAL_STATE_DIM,
            camera_keys=CAMERA_KEYS,
        )
        control = request.get("control", {})
        if not isinstance(control, dict):
            raise ProtocolError("`control` must be a dictionary")
        rtc_requested = bool(control.get("enable_rtc", False))

        with self._lock:
            if bool(request.get("reset", False)) or session_id not in self._sessions:
                self._sessions[session_id] = _SessionState()
                self.policy.reset()
                self.preprocessor.reset()
                self.postprocessor.reset()
            session = self._sessions[session_id]

            preprocess_start = time.perf_counter()
            observation = self._prepare_observation(state_24d, images, task)
            processed_observation = self.preprocessor(observation)
            (
                rtc_prefix,
                inference_delay,
                rtc_prefix_used,
                rtc_prefix_source_steps,
                rtc_effective_horizon,
                rtc_blend_steps,
                rtc_fallback_reason,
            ) = self._rtc_prefix(
                session,
                control,
                torch.from_numpy(select_native_state(state_24d, self.native_action_dim)),
            )
            preprocess_s = time.perf_counter() - preprocess_start

            infer_start = time.perf_counter()
            kwargs: dict[str, Any] = {}
            if self._fixed_noise is not None:
                kwargs["noise"] = self._fixed_noise.clone()
            if rtc_requested and self.rtc_inference_mode == "trained":
                kwargs.update(
                    {
                        "rtc_mode": "trained",
                        "prev_chunk_left_over": rtc_prefix,
                        "inference_delay": inference_delay,
                        "execution_horizon": self.rtc_config.execution_horizon,
                    }
                )
            elif self.rtc_config.enabled:
                kwargs.update(
                    {
                        "prev_chunk_left_over": rtc_prefix,
                        "inference_delay": inference_delay,
                    }
                )
            # RTC guidance uses ``torch.enable_grad()`` internally to compute a
            # correction with respect to x_t.  Unlike ``torch.no_grad()``,
            # ``torch.inference_mode()`` cannot be overridden by enable_grad and
            # makes every request with a previous-chunk prefix fail.  no_grad
            # still keeps ordinary inference gradient-free while allowing that
            # narrowly scoped RTC calculation.
            with torch.no_grad():
                raw_actions = self.policy.predict_action_chunk(processed_observation, **kwargs)
            infer_s = time.perf_counter() - infer_start

            if raw_actions.ndim == 2:
                raw_actions = raw_actions.unsqueeze(0)
            if raw_actions.ndim != 3 or raw_actions.shape[0] != 1:
                raise RuntimeError(f"Unexpected policy output shape: {tuple(raw_actions.shape)}")

            postprocess_start = time.perf_counter()
            raw_chunk = raw_actions.squeeze(0).detach().clone()
            processed_native = self.postprocessor(raw_actions).squeeze(0).detach().cpu()
            native_numpy = processed_native.numpy().astype(np.float32, copy=False)
            canonical_actions = expand_actions_to_canonical(
                native_numpy,
                state_24d,
                command_head_waist=self.command_head_waist,
            )
            if not np.isfinite(canonical_actions).all():
                raise RuntimeError("Policy produced non-finite actions")
            postprocess_s = time.perf_counter() - postprocess_start

            session.next_chunk_number += 1
            chunk_id = f"{session_id}:{session.next_chunk_number}"
            session.chunk_id = chunk_id
            session.raw_actions = raw_chunk
            session.processed_native_actions = processed_native
            session.last_inference_s = infer_s

        return {
            "status": "ok",
            "kind": "prediction",
            "request_id": request_id,
            "session_id": session_id,
            "chunk_id": chunk_id,
            "actions": canonical_actions,
            "native_actions": native_numpy,
            "native_action_space": "absolute_unnormalized",
            "native_action_dim": self.native_action_dim,
            "native_actions_sha256": _native_action_sha256(native_numpy),
            "action_space": "absolute",
            "action_dim": CANONICAL_ACTION_DIM,
            "rtc": {
                "requested": rtc_requested,
                "inference_mode": self.rtc_inference_mode if rtc_requested else "none",
                "prefix_used": rtc_prefix_used,
                "estimated_delay_steps": inference_delay,
                "prefix_source_steps": rtc_prefix_source_steps,
                "effective_execution_horizon": rtc_effective_horizon,
                "blend_steps": rtc_blend_steps,
                "fallback_reason": rtc_fallback_reason,
                "reanchor_mode": session.rtc_reanchor_mode,
                "reanchor_max_abs_rad": session.rtc_reanchor_max_abs_rad,
                "prefix_source": session.rtc_prefix_source,
                "prefix_committed_steps": session.rtc_prefix_committed_steps,
            },
            "server_timing": {
                "preprocess_ms": preprocess_s * 1000,
                "infer_ms": infer_s * 1000,
                "postprocess_ms": postprocess_s * 1000,
                "total_ms": (preprocess_s + infer_s + postprocess_s) * 1000,
            },
        }


class MockG2APolicy:
    """Fast absolute-action policy used for protocol and dry-run tests."""

    def __init__(
        self,
        *,
        model_fps: float = 30.0,
        chunk_size: int = 50,
        execution_horizon: int = 30,
        inference_delay_s: float = 0.0,
        effector_profile: str = "omnipicker_legacy",
    ) -> None:
        self.model_fps = model_fps
        self.chunk_size = chunk_size
        self.execution_horizon = execution_horizon
        self.inference_delay_s = inference_delay_s
        self.effector_profile = "dahuan" if str(effector_profile) == "dh" else str(effector_profile)
        self._lock = threading.Lock()
        self._chunk_numbers: dict[str, int] = {}

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "kind": "metadata",
            "protocol": {"name": PROTOCOL_NAME, "version": PROTOCOL_VERSION},
            "checkpoint": "mock",
            "checkpoint_id": "mock",
            "checkpoint_stats_sha256": None,
            "processor_version": "mock/v1",
            "model_profile": "mock_24d_absolute",
            "policy_type": "mock",
            "native_action_dim": CANONICAL_ACTION_DIM,
            "native_action_space": "absolute_unnormalized",
            "action_capture": {
                "available": True,
                "native_action_space": "absolute_unnormalized",
                "native_action_dim": CANONICAL_ACTION_DIM,
                "content_sha256": True,
            },
            "canonical_state_dim": CANONICAL_STATE_DIM,
            "canonical_action_dim": CANONICAL_ACTION_DIM,
            "canonical_action_names": list(CANONICAL_ACTION_NAMES),
            "action_space": "absolute",
            "effector_profile": self.effector_profile,
            "chunk_size": self.chunk_size,
            "model_fps": self.model_fps,
            "camera_keys": list(CAMERA_KEYS),
            "image_transport": {
                "encoded_payloads": True,
                "encodings": ["jpeg", "raw"],
                "decoded_format": "HWC uint8 RGB",
            },
            "head_waist_locked": True,
            "relative_actions_internal": False,
            "rtc": {
                "available": True,
                "request_controlled": True,
                "execution_horizon": self.execution_horizon,
                "max_guidance_weight": 0.0,
                "prefix_attention_schedule": "mock",
            },
        }

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._chunk_numbers[session_id] = 0

    def drop_session(self, session_id: str) -> None:
        with self._lock:
            self._chunk_numbers.pop(session_id, None)

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        infer_start = time.perf_counter()
        request_id = str(request.get("request_id", ""))
        session_id = str(request.get("session_id", ""))
        state, _, _ = validate_observation(
            request.get("observation"),
            state_dim=CANONICAL_STATE_DIM,
            camera_keys=CAMERA_KEYS,
        )
        with self._lock:
            if bool(request.get("reset", False)):
                self._chunk_numbers[session_id] = 0
            number = self._chunk_numbers.get(session_id, 0) + 1
            self._chunk_numbers[session_id] = number
        if self.inference_delay_s:
            time.sleep(self.inference_delay_s)
        infer_ms = (time.perf_counter() - infer_start) * 1000
        actions = np.repeat(state[None, :], self.chunk_size, axis=0).astype(np.float32)
        if self.effector_profile == "dahuan":
            actions[:, [7, 15]] = 0.025 - actions[:, [7, 15]]
        control = request.get("control", {})
        rtc_requested = bool(control.get("enable_rtc", False))
        previous_chunk_id = f"{session_id}:{number - 1}"
        prefix_matches = bool(
            rtc_requested
            and number > 1
            and control.get("active_chunk_id") == previous_chunk_id
        )
        next_index = max(0, min(int(control.get("next_action_index", 0)), self.chunk_size))
        prefix_source_steps = self.chunk_size - next_index if prefix_matches else 0
        effective_horizon = min(self.execution_horizon, prefix_source_steps)
        estimated_delay = max(
            0,
            min(int(control.get("estimated_delay_steps", 0)), effective_horizon),
        )
        blend_steps = max(0, effective_horizon - estimated_delay)
        min_blend_steps = max(0, int(control.get("min_rtc_blend_steps", 0)))
        prefix_used = prefix_matches and blend_steps >= min_blend_steps
        if not rtc_requested:
            fallback_reason = "request_disabled"
        elif not prefix_matches:
            fallback_reason = "no_matching_previous_chunk"
        elif not prefix_used:
            fallback_reason = "insufficient_blend_steps"
        else:
            fallback_reason = None
        return {
            "status": "ok",
            "kind": "prediction",
            "request_id": request_id,
            "session_id": session_id,
            "chunk_id": f"{session_id}:{number}",
            "actions": actions,
            "native_actions": actions,
            "native_action_space": "absolute_unnormalized",
            "native_action_dim": CANONICAL_ACTION_DIM,
            "native_actions_sha256": _native_action_sha256(actions),
            "action_space": "absolute",
            "action_dim": CANONICAL_ACTION_DIM,
            "rtc": {
                "requested": rtc_requested,
                "prefix_used": prefix_used,
                "estimated_delay_steps": estimated_delay,
                "prefix_source_steps": prefix_source_steps,
                "effective_execution_horizon": effective_horizon,
                "blend_steps": blend_steps,
                "fallback_reason": fallback_reason,
            },
            "server_timing": {
                "preprocess_ms": 0.0,
                "infer_ms": infer_ms,
                "postprocess_ms": 0.0,
                "total_ms": infer_ms,
            },
        }
