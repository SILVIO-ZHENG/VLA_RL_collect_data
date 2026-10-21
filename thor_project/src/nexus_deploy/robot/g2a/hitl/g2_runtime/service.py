"""Production composition of the G2 robot client and FlowPRO DDS runtime.

The process starts the robot client in an acknowledged measured HOLD.  DDS is
attached only after that acknowledgement, so an early VR gesture cannot race
the action thread.  The existing tablet recorder remains externally owned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ...client import (
    RobotClientRunner,
    run_robot_client,
)
from ...client import (
    build_arg_parser as build_client_arg_parser,
)
from ...safety_layer import load_safety_layer
from ..common.collection_manifest import (
    FrozenCollectionManifest,
    load_frozen_collection_manifest,
    validate_runtime_binding,
)
from ..common.runtime_provenance import load_and_verify_runtime_provenance
from ..common.sidecar import (
    VENDOR_META_FINAL,
    VENDOR_META_PATH,
    classify_vendor_meta,
    read_vendor_meta,
)
from .app import FlowPROG2Application
from .dds_runtime import G2DDSRuntimeCoordinator, G2DDSRuntimeService
from .g2_dds import G2ReadOnlyDDSClient, G2RuntimeEvidence
from .rollback import RollbackTrajectory, plan_reverse_measured_trajectory
from .supervisor import EpisodeCoordinator, LogicalInput, SupervisorMode
from .vendor_ee_gate import VendorEffectorGate

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    return build_client_arg_parser()


@dataclass(frozen=True)
class ApprovedSafetyCalibration:
    config_sha256: str


def require_approved_safety_calibration(
    path: Path,
    *,
    expected_robot_id: str,
    expected_effector_profile: str,
) -> ApprovedSafetyCalibration:
    """Verify the selected Nexus execute profile and its robot assets."""
    resolved = path.expanduser().resolve(strict=True)
    raw = resolved.read_bytes()
    profile = json.loads(raw)
    if profile.get("calibrated") is not True or profile.get("execute_approved") is not True:
        raise ValueError("FlowPRO rollback requires an approved calibrated safety profile")
    if not expected_robot_id or not profile.get("execution_approval"):
        raise ValueError("FlowPRO rollback lacks robot identity or execution approval")
    model = profile.get("robot_model") or {}
    if model.get("effector_profile") != expected_effector_profile:
        raise ValueError("FlowPRO rollback safety profile has the wrong effector geometry")
    load_safety_layer(resolved)
    return ApprovedSafetyCalibration(hashlib.sha256(raw).hexdigest())


def build_parser() -> argparse.ArgumentParser:
    parser = build_arg_parser()
    parser.description = __doc__
    parser.add_argument(
        "--sidecar-root",
        type=Path,
        default=Path("/data/logs/flowpro_sidecar"),
        help="Small FlowPRO provenance store on G2; vendor video/H5 remain under --vendor-record-root.",
    )
    parser.add_argument(
        "--vendor-record-root",
        type=Path,
        default=Path("/data/record"),
        help="Tablet-controlled vendor recorder root (read-only to FlowPRO until sealing).",
    )
    parser.add_argument(
        "--initial-vendor-record-name",
        help="Bind this newly detected tablet episode in HOLD before accepting Y.",
    )
    parser.add_argument("--robot-id", default="g2-sorter-01")
    parser.add_argument("--task-id", default="package-sorting")
    parser.add_argument("--dds-queue-size", type=int, default=8192)
    parser.add_argument("--dds-tick-ms", type=float, default=20.0)
    parser.add_argument(
        "--tablet-poll-seconds",
        type=float,
        default=2.0,
        help="Poll the external tablet recorder after FlowPRO exit and seal completed sidecars.",
    )
    parser.add_argument(
        "--flowpro-feedback-root",
        type=Path,
        default=Path("/data/logs/flowpro_operator_feedback"),
        help="Atomic scene-restoration image/status handoff for the tablet/VR display bridge.",
    )
    parser.add_argument(
        "--enable-rollback",
        action="store_true",
        help=(
            "Enable physical deadman rollback in --execute mode. This capability is independent "
            "of optional --safety-config collision checking."
        ),
    )
    parser.add_argument(
        "--enable-vendor-ee-handoff",
        action="store_true",
        help=(
            "Enable the verified Human/Policy gripper ownership handoff in --execute mode. "
            "Before Policy moves, FlowPRO disables vendor teleop ee.enable and waits for "
            "configuration feedback; a trigger held across the boundary is masked until release."
        ),
    )
    parser.add_argument(
        "--collection-manifest",
        type=Path,
        default=None,
        help=(
            "Frozen flowpro.collection_manifest/v1 for formal collection. If omitted, sessions "
            "are treated as pilot/debug data and have no collection-round identity."
        ),
    )
    parser.add_argument(
        "--runtime-provenance",
        type=Path,
        default=Path("RUNTIME_PROVENANCE.json"),
        help="Build-generated G2 runtime file manifest, required and verified for formal collection.",
    )
    return parser


def _policy_manifest(metadata: dict[str, Any]) -> dict[str, Any]:
    required = {
        "checkpoint_id",
        "checkpoint_stats_sha256",
        "native_action_dim",
        "chunk_size",
        "effector_profile",
        "processor_version",
        "model_fps",
    }
    missing = sorted(required - metadata.keys())
    if missing:
        raise RuntimeError(f"Policy server lacks FlowPRO provenance metadata: {missing}")
    stats_sha256 = str(metadata["checkpoint_stats_sha256"])
    if len(stats_sha256) != 64 or any(character not in "0123456789abcdef" for character in stats_sha256):
        raise RuntimeError("Policy server returned an invalid checkpoint_stats_sha256")
    return {key: metadata[key] for key in required}


def _validate_formal_model_fps(metadata: dict[str, Any], execution_fps: float) -> None:
    advertised = metadata.get("model_fps", metadata.get("fps"))
    if advertised is None or not math.isclose(
        float(advertised), float(execution_fps), rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError(
            "Formal FlowPRO collection requires model_fps to equal execution_fps: "
            f"model_fps={advertised!r}, execution_fps={execution_fps!r}"
        )


def _canonical_limit_vector(
    *,
    arm: float,
    effector: float,
    head_waist: float,
) -> np.ndarray:
    return np.asarray(
        [arm] * 7 + [effector] + [arm] * 7 + [effector] + [head_waist] * 8,
        dtype=np.float32,
    )


def _make_rollback_planner(
    runner: RobotClientRunner,
    args: argparse.Namespace,
) -> Any:
    def planner(
        segment: Any,
        intervention_id: str,
        frozen_current_state_24d: Any,
    ) -> RollbackTrajectory:
        if args.hardware_mode != "execute":
            raise RuntimeError("Physical rollback is disabled outside --execute mode")
        if not args.enable_rollback:
            raise RuntimeError("Physical rollback capability was not enabled")
        audit = None
        path_guard = None
        if args.safety_config is not None:
            safety_path = args.safety_config.expanduser().resolve()
            audit = require_approved_safety_calibration(
                safety_path,
                expected_robot_id=args.robot_id,
                expected_effector_profile=str(runner.policy_client.metadata["effector_profile"]),
            )
            # Planning owns a separate FK/safety instance, so its cache cannot
            # race the adapter's 100 Hz realtime guard. Any projection/hold means
            # the frozen reverse path is rejected rather than silently modified.
            planning_safety = load_safety_layer(safety_path)

            def path_guard(start: np.ndarray, end: np.ndarray) -> bool:
                decision = planning_safety.guard(start, end, raw_action=end)
                return bool(
                    not decision.hard_limit_triggered
                    and np.allclose(decision.safe_action, end, rtol=0.0, atol=1e-7)
                )

        return plan_reverse_measured_trajectory(
            segment,
            intervention_id=intervention_id,
            # Planning runs on a background thread. Bind it to the immutable
            # measured snapshot captured after Policy entered HOLD instead of
            # issuing an off-thread GDK read.
            current_state_24d=frozen_current_state_24d,
            servo_fps=runner.servo_fps,
            max_velocity=_canonical_limit_vector(
                arm=args.max_arm_velocity,
                effector=args.max_effector_velocity,
                head_waist=args.max_head_waist_velocity,
            ),
            max_acceleration=_canonical_limit_vector(
                arm=args.max_arm_acceleration,
                effector=args.max_effector_acceleration,
                head_waist=args.max_head_waist_acceleration,
            ),
            fixed_head_waist_8d=(
                runner.flowpro_fixed_head_waist_snapshot()
                if int(runner.policy_client.metadata["native_action_dim"]) == 16
                else None
            ),
            collision_safety_enabled=audit is not None,
            safety_config_sha256=None if audit is None else audit.config_sha256,
            kinematics_calibrated=audit is not None,
            path_guard=path_guard,
        )

    return planner


class FlowPROServiceComposition:
    """Own DDS lifecycle while RobotClientRunner remains the sole GDK writer."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.collection_manifest: FrozenCollectionManifest | None = (
            None
            if args.collection_manifest is None
            else load_frozen_collection_manifest(args.collection_manifest)
        )
        self.stop_event = threading.Event()
        self._closed = False
        self.dds_thread: threading.Thread | None = None
        self.sealer_thread: threading.Thread | None = None
        self.dds_service: G2DDSRuntimeService | None = None
        self.dds_error: BaseException | None = None
        self.sealer_last_error: BaseException | None = None
        self.runner: RobotClientRunner | None = None
        self.application: FlowPROG2Application | None = None

    def attach(self, runner: RobotClientRunner) -> None:
        if self.runner is not None:
            raise RuntimeError("FlowPRO service is already attached")
        self.runner = runner
        policy = _policy_manifest(runner.policy_client.metadata)
        if self.collection_manifest is not None:
            if not self.args.enable_vendor_ee_handoff:
                raise ValueError(
                    "Formal FlowPRO collection requires --enable-vendor-ee-handoff"
                )
            _validate_formal_model_fps(runner.policy_client.metadata, self.args.fps)
            runtime_provenance = load_and_verify_runtime_provenance(self.args.runtime_provenance)
            validate_runtime_binding(
                self.collection_manifest,
                args=self.args,
                policy=policy,
                runtime_source_sha256=str(runtime_provenance["source_sha256"]),
            )
        if self.args.hardware_mode == "execute" and self.args.safety_config is not None:
            require_approved_safety_calibration(
                self.args.safety_config,
                expected_robot_id=self.args.robot_id,
                expected_effector_profile=str(policy["effector_profile"]),
            )
        evidence = G2RuntimeEvidence()
        effector_gate = None
        if self.args.hardware_mode == "execute" and self.args.enable_vendor_ee_handoff:
            # The adapter has already initialized this process's GDK runtime.
            # Reuse that module and create only the audited config pub/sub pair;
            # no second robot or action writer is introduced.
            effector_gate = VendorEffectorGate.from_gdk(
                gdk=getattr(runner.adapter, "_gdk", None),
            )
        self.application = FlowPROG2Application(
            robot_client_runner=runner,
            sidecar_root=self.args.sidecar_root,
            vendor_record_root=self.args.vendor_record_root,
            robot_id=self.args.robot_id,
            task_id=self.args.task_id,
            policy=policy,
            rollback_planner=_make_rollback_planner(runner, self.args),
            arbitration_monitor=evidence.arbitration,
            feedback_root=self.args.flowpro_feedback_root,
            collision_safety_enabled=(
                self.args.hardware_mode == "execute" and self.args.safety_config is not None
            ),
            rollback_enabled=(self.args.hardware_mode == "execute" and self.args.enable_rollback),
            # No time horizon: X duration and the current contiguous Policy
            # segment determine how far rollback can travel.
            max_rollback_ns=None,
            collection_manifest=(
                None if self.collection_manifest is None else self.collection_manifest.identity
            ),
            effector_gate=effector_gate,
        )
        coordinator = G2DDSRuntimeCoordinator(
            application=self.application,
            evidence=evidence,
            cameras_fresh=runner.flowpro_camera_fresh,
            safety_state=self._runtime_safety_state,
        )
        self.dds_service = G2DDSRuntimeService(
            client=G2ReadOnlyDDSClient(),
            coordinator=coordinator,
            queue_size=self.args.dds_queue_size,
            tick_seconds=self.args.dds_tick_ms / 1000.0,
        )
        self.dds_thread = threading.Thread(
            target=self._run_dds,
            name="FlowPRODDS",
            daemon=True,
        )
        self.sealer_thread = threading.Thread(
            target=self._run_tablet_sealer,
            name="FlowPROTabletSealer",
            daemon=True,
        )
        logger.warning(
            "FLOWPRO READY in DISABLED/HOLD-safe startup: tablet recorder remains external; "
            "collision_safety=%s rollback=%s vendor_ee_handoff=%s collection_round=%s "
            "sidecar_root=%s",
            (
                "enabled"
                if self.args.hardware_mode == "execute" and self.args.safety_config is not None
                else "disabled"
            ),
            ("enabled" if self.args.hardware_mode == "execute" and self.args.enable_rollback else "disabled"),
            (
                "enabled"
                if self.args.hardware_mode == "execute" and self.args.enable_vendor_ee_handoff
                else "disabled"
            ),
            (
                "pilot/unbound"
                if self.collection_manifest is None
                else self.collection_manifest.payload["collection_round_id"]
            ),
            self.args.sidecar_root,
        )

    def start(self) -> None:
        """Expose DDS only after the action thread confirms startup HOLD."""
        if self.runner is None or self.dds_thread is None or self.sealer_thread is None:
            raise RuntimeError("FlowPRO service has not been attached to a robot client")
        if not self.runner._pause_active_event.wait(5.0):
            raise TimeoutError("FlowPRO startup measured HOLD was not acknowledged")
        if self.args.hardware_mode == "execute":
            evidence = self.runner.adapter.last_delivery_evidence
            if evidence is None or not evidence.command_accepted:
                raise RuntimeError("FlowPRO startup HOLD has no accepted GDK command")
        assert self.application is not None
        clock_offset_ns = self.runner.adapter.freeze_camera_clock_offset()
        self.application.coordinator.wall_to_vendor_offset_ns = clock_offset_ns
        logger.warning("FLOWPRO CAMERA CLOCK BRIDGE wall_to_vendor_offset_ns=%d", clock_offset_ns)
        if getattr(self.args, "initial_vendor_record_name", None):
            self.application.bind_initial_recording_in_hold(self.args.initial_vendor_record_name)
        self.dds_thread.start()
        self.sealer_thread.start()

    def _runtime_safety_state(self) -> dict[str, bool]:
        if self.runner is None:
            return {"emergency_stop_active": True, "collision_detected": True}
        if self.args.safety_config is None:
            # Collision checking is an explicitly unavailable capability, not
            # a fabricated positive collision. Platform estop/fault/motion
            # evidence remains independently fail-closed in G2RuntimeEvidence.
            return {"emergency_stop_active": False, "collision_detected": False}
        return self.runner.flowpro_safety_state()

    def _run_dds(self) -> None:
        assert self.dds_service is not None and self.runner is not None
        try:
            self.dds_service.run(external_stop=self.stop_event)
        except BaseException as error:
            self.dds_error = error
            logger.exception("FlowPRO DDS runtime failed closed")
            self.runner.shutdown_event.set()

    def _advance_tablet_lifecycle_once(self) -> Path | None:
        """Advance one exited session without touching robot/VR control."""

        if self.application is None:
            return None
        coordinator = self.application.coordinator
        if coordinator.writer is None:
            return None
        state = coordinator._read_state()
        if self.application.session.flowpro_session_id is not None:
            if state["phase"] != coordinator.PHASE_ATTACHED:
                return None
            try:
                recording_finalized = coordinator.attached_recording_finalized()
            except (OSError, RuntimeError, ValueError) as error:
                # Tablet metadata may be temporarily incomplete while Save is
                # rewriting it. Treat that interval as unverified recording;
                # the activity check below keeps Policy fail-closed.
                logger.warning("FLOWPRO TABLET META UNVERIFIED: %s", error)
                recording_finalized = False
            if not recording_finalized:
                if (
                    self.application.session.supervisor.mode
                    in {SupervisorMode.AUTO, SupervisorMode.FORWARD_RUN}
                    and not coordinator.attached_recording_active()
                ):
                    # A recorder which stopped producing files is no longer
                    # authoritative capture evidence. Stop Policy while the
                    # tablet's Save/Discard outcome remains unresolved.
                    self.application._handle_input(
                        LogicalInput(
                            "y_press", time.time_ns(),
                            {"source": "tablet_recording_inactive"},
                        )
                    )
                    logger.warning(
                        "FLOWPRO TABLET RECORDING INACTIVE: Policy requested HOLD pending Save/Discard"
                    )
                return None
            mode = self.application.session.supervisor.mode
            if mode in {SupervisorMode.AUTO, SupervisorMode.FORWARD_RUN}:
                # Tablet Save is an authoritative episode boundary. Stop the
                # Policy immediately, then let the next poll finalize from the
                # acknowledged HOLD. Use the same Y-stop path so generation
                # and actually-executed prefixes retain normal lineage.
                self.application._handle_input(
                    LogicalInput(
                        "y_press",
                        time.time_ns(),
                        {"source": "tablet_recording_finished"},
                    )
                )
                return None
            if mode != SupervisorMode.HOLD:
                logger.warning(
                    "FLOWPRO TABLET FINAL META WAITING FOR HOLD: mode=%s",
                    mode.value,
                )
                return None
            self.application.finish_from_tablet(timestamp_ns=time.time_ns())
            state = coordinator._read_state()
        if state["phase"] == coordinator.PHASE_ATTACHED:
            # Session shutdown has not durably recorded its recorder-commit
            # boundary yet. Never infer an outcome from vendor files here.
            return None
        state = coordinator.observe_vendor_completion()
        if state["phase"] != coordinator.PHASE_READY:
            return None
        return coordinator.finalize()

    def _bind_next_tablet_recording_in_hold(self) -> None:
        """Bind later tablet recordings without restarting the Nexus deployment."""

        if self.application is None:
            return
        app = self.application
        if app.coordinator.writer is not None or app.session.supervisor.mode != SupervisorMode.DISABLED:
            return
        active = app.coordinator.list_active_recordings(app.vendor_record_root)
        if len(active) != 1:
            return
        app.bind_initial_recording_in_hold(active[0].name)
        logger.warning("FLOWPRO TABLET RECORDING BOUND IN HOLD: %s", active[0].name)

    def _handoff_replaced_tablet_recording_once(self) -> None:
        """Release an inactive clip when a distinct new tablet clip is live."""

        if self.application is None:
            return
        app = self.application
        if app.coordinator.writer is None or app.session.supervisor.mode not in {
            SupervisorMode.HOLD, SupervisorMode.DISABLED,
        }:
            return
        state = app.coordinator._read_state()
        if state["phase"] not in {
            app.coordinator.PHASE_ATTACHED, app.coordinator.PHASE_AWAITING_SAVE,
        }:
            return
        active = app.coordinator.list_active_recordings(app.vendor_record_root)
        if len(active) != 1 or active[0].name == state["record_name"]:
            return
        parked = app.release_replaced_recording(next_record_name=active[0].name)
        if parked is not None:
            logger.warning(
                "FLOWPRO TABLET PREVIOUS RECORDING UNCONFIRMED: %s; "
                "retained for late Save, next recording=%s",
                parked, active[0].name,
            )

    def _advance_parked_sidecars_once(self) -> list[Path]:
        """Seal a parked clip only if its tablet later publishes final meta."""

        if self.application is None:
            return []
        app = self.application
        current = None if app.coordinator.writer is None else app.coordinator.writer.open_directory
        sealed: list[Path] = []
        for directory in sorted(app.coordinator.sidecar_root.glob("*.open")):
            if directory == current or directory.is_symlink() or not directory.is_dir():
                continue
            state_path = directory / "coordinator_state.json"
            if state_path.is_symlink() or not state_path.is_file():
                continue
            try:
                coordinator = EpisodeCoordinator.open_existing(
                    directory, vendor_record_root=app.vendor_record_root,
                )
                state = coordinator._read_state()
                if state["phase"] not in {
                    coordinator.PHASE_AWAITING_SAVE,
                    coordinator.PHASE_POSTPROCESSING,
                    coordinator.PHASE_READY,
                }:
                    continue
                meta_path = Path(state["vendor_episode_path"]) / VENDOR_META_PATH
                if meta_path.is_symlink() or not meta_path.is_file():
                    continue
                if classify_vendor_meta(read_vendor_meta(meta_path)) != VENDOR_META_FINAL:
                    continue
                state = coordinator.observe_vendor_completion()
                if state["phase"] == coordinator.PHASE_READY:
                    sealed.append(coordinator.finalize())
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
                logger.warning("FLOWPRO PARKED SIDECAR STILL PENDING %s: %s", directory, error)
        return sealed

    def _run_tablet_sealer(self) -> None:
        poll_seconds = float(getattr(self.args, "tablet_poll_seconds", 2.0))
        if poll_seconds <= 0:
            self.sealer_last_error = ValueError("tablet_poll_seconds must be positive")
            logger.error("FLOWPRO TABLET SEALER DISABLED: tablet_poll_seconds must be positive")
            return
        last_error_text: str | None = None
        while not self.stop_event.is_set():
            try:
                sealed = self._advance_tablet_lifecycle_once()
                self._handoff_replaced_tablet_recording_once()
                self._bind_next_tablet_recording_in_hold()
                parked_sealed = self._advance_parked_sidecars_once()
                self.sealer_last_error = None
                last_error_text = None
                if sealed is not None:
                    logger.warning("FLOWPRO DATA SEALED: %s", sealed)
                for parked in parked_sealed:
                    logger.warning("FLOWPRO LATE TABLET SAVE SEALED: %s", parked)
            except (OSError, RuntimeError, ValueError) as error:
                # Recorder postprocessing, temporary ENOSPC, and incomplete
                # vendor files are data-lifecycle failures. They must remain
                # retryable and must never stop the DDS/GDK control process.
                self.sealer_last_error = error
                error_text = f"{type(error).__name__}: {error}"
                if error_text != last_error_text:
                    logger.error("FLOWPRO TABLET SEAL PENDING: %s", error_text)
                    last_error_text = error_text
            self.stop_event.wait(poll_seconds)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.stop_event.set()
        if self.dds_service is not None:
            self.dds_service.stop()
        if self.dds_thread is not None and self.dds_thread.ident is not None:
            self.dds_thread.join(timeout=3.0)
            if self.dds_thread.is_alive():
                raise RuntimeError("FlowPRO DDS thread did not stop")
        if self.sealer_thread is not None and self.sealer_thread.ident is not None:
            self.sealer_thread.join(timeout=30.0)
            if self.sealer_thread.is_alive():
                logger.warning("FlowPRO tablet sealer is still finishing a crash-recoverable seal")
        self._finalize_open_session_as_aborted()
        if self.dds_error is not None:
            raise RuntimeError("FlowPRO DDS runtime terminated with an error") from self.dds_error

    def restore_vendor_effector_control_after_stop(self) -> None:
        """Return gripper ownership only after the robot writer has stopped."""

        application = self.application
        gate = None if application is None else application.effector_gate
        if gate is None:
            return
        try:
            gate.begin_handoff(True)
            gate.wait_confirmed(True, timeout_s=3.0)
        except Exception as error:
            logger.exception(
                "FLOWPRO VENDOR GRIPPER CONTROL RESTORE FAILED after robot stop"
            )
            raise RuntimeError(
                "Vendor gripper control restore was not confirmed after robot stop"
            ) from error
        else:
            logger.warning(
                "FLOWPRO VENDOR GRIPPER CONTROL RESTORED after robot stop: ee.enable=true"
            )

    def _finalize_open_session_as_aborted(self) -> None:
        if self.application is None or self.application.session.flowpro_session_id is None:
            return
        session = self.application.session
        if session.supervisor.mode != SupervisorMode.ABORT:
            session.handle(
                LogicalInput(
                    "fatal_fault",
                    time.time_ns(),
                    {"source": "flowpro_service_shutdown"},
                )
            )
        session.finalize_abort(
            detail={
                "reason": "flowpro_service_shutdown",
                "dds_error": None if self.dds_error is None else type(self.dds_error).__name__,
            }
        )


def main() -> None:
    args = build_parser().parse_args()
    if args.hardware_mode != "execute":
        raise ValueError("FlowPRO G2 service currently requires --execute")
    if args.dds_queue_size <= 0 or args.dds_tick_ms <= 0:
        raise ValueError("FlowPRO DDS queue/tick configuration must be positive")
    if args.enable_rollback and args.hardware_mode != "execute":
        raise ValueError("--enable-rollback requires --execute")
    if args.enable_rollback and (
        args.safety_config is None or not args.async_safety or not args.rolling_safety
        or not math.isclose(args.servo_fps, 100.0, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise ValueError("Physical rollback requires approved 100 Hz async rolling safety")
    if args.enable_rollback and not args.enable_vendor_ee_handoff:
        raise ValueError("Physical rollback requires verified vendor effector handoff")
    if args.enable_vendor_ee_handoff and args.hardware_mode != "execute":
        raise ValueError("--enable-vendor-ee-handoff requires --execute")
    if args.hardware_mode == "execute" and args.safety_config is not None:
        require_approved_safety_calibration(
            args.safety_config,
            expected_robot_id=args.robot_id,
            expected_effector_profile=args.effector_profile,
        )
    composition = FlowPROServiceComposition(args)
    try:
        run_robot_client(
            args,
            runner_setup=composition.attach,
            startup_hook=composition.start,
            shutdown_hook=composition.close,
            start_paused=True,
        )
    finally:
        try:
            composition.close()
        finally:
            composition.restore_vendor_effector_control_after_stop()


def _interrupt(*_args: object) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _interrupt)
    main()
