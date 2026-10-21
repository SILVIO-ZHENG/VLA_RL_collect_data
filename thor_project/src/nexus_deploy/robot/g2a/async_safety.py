"""Bounded 50 Hz collision worker. The child process never imports or calls GDK.

Only the feedback thread reads feedback. Servo submits/polls bounded mailboxes;
neither geometry, IPC waits nor feedback I/O execute in the servo thread.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
import time
from collections import OrderedDict
from dataclasses import replace

import numpy as np

from .safety_layer import URDFForwardKinematics, load_safety_layer

CHECK_PERIOD_S = 0.020
CHECK_BUDGET_MS = 20.0
RESULT_MAX_AGE_MS = 40.0
EFFECTOR_BOUND_M = 0.025001


def install_exact_geometry_cache(layer, *, capacity=128):
    """Memoize pure geometry, never feedback, decisions or dispatch authority.

    Install only on the geometry child's private, immutable profile instance.
    Frozen committed waypoints recur across windows. Their complete float64
    pose bytes (including both measured gripper openings) identify exactly the
    same FK/surface/capsule calculation. Every window still evaluates its new
    measured pose, all original samples, paths and hold decisions, and keeps
    its original feedback timestamp and expiry checks. No approximate keys,
    cross-profile sharing, changed stride, or cached safety certificates.
    """
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        raise ValueError("Exact geometry cache capacity must be a positive integer")
    stats = {"pose_hits": 0, "pose_misses": 0, "pose_entries": 0,
             "inter_arm_hits": 0, "inter_arm_misses": 0, "inter_arm_entries": 0}

    def memoize(function, name, *, with_fk=False):
        entries = OrderedDict()

        def evaluate(q, *args):
            pose = np.asarray(q, dtype=np.float64)
            key = (pose.shape, pose.tobytes(), id(args[0]) if with_fk else None)
            if key in entries:
                stats[name + "_hits"] += 1
                entries.move_to_end(key)
                return entries[key]
            stats[name + "_misses"] += 1
            value = function(q, *args)  # Failed evaluations are never cached.
            entries[key] = value
            if len(entries) > capacity:
                entries.popitem(last=False)
            stats[name + "_entries"] = len(entries)
            return value

        return evaluate

    if layer.inter_arm_collision is not None:
        layer.inter_arm_collision.evaluate = memoize(
            layer.inter_arm_collision.evaluate, "inter_arm", with_fk=True)
    layer._evaluate = memoize(layer._evaluate, "pose")
    return stats

def servo_preparation_time(last_dispatch, period, *, rolling):
    """Start preparation early; final dispatch pacing/expiry remain mandatory."""
    lead = min(.006, period * .6) if rolling else min(.002, period * .2)
    return last_dispatch + period - lead


def braking_sample(reserve, step):
    """Closed-form monotone discrete braking; no online target changes."""
    if not 0 <= step <= reserve["steps"]:
        raise ValueError("Braking step outside finite reserve")
    velocity = np.asarray(reserve["velocity"], dtype=np.float64)
    decrement = np.asarray(reserve["acceleration"], dtype=np.float64) * reserve["dt"]
    count = np.minimum(step, np.floor(np.abs(velocity) / decrement))
    displacement = np.sign(velocity) * reserve["dt"] * (
        count * np.abs(velocity) - decrement * count * (count + 1) / 2)
    position = np.asarray(reserve["anchor"], dtype=np.float64) + displacement
    next_velocity = np.sign(velocity) * np.maximum(np.abs(velocity) - step * decrement, 0)
    return position.astype(np.float32), next_velocity.astype(np.float32)


def braking_clearance(reserve, feedback, coefficients):
    """Global plane bound on the whole measured-to-stop joint-space box.

    Applies at every continuous point, not only sampled waypoints. Coefficients
    are the global URDF displacement bounds already used for hold renewal.
    This intentionally ignores finite tabletop XY extents (more conservative).
    """
    if coefficients is None:
        return -float("inf")
    feedback = np.asarray(feedback, dtype=np.float64)
    if feedback.shape != (24,) or not np.isfinite(feedback).all():
        return -float("inf")
    if np.any(feedback[[7, 15]] < 0) or np.any(feedback[[7, 15]] > EFFECTOR_BOUND_M):
        return -float("inf")
    anchor = np.asarray(reserve["anchor"], dtype=np.float64)
    end = np.asarray(reserve["end"], dtype=np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if any(q.shape != (24,) or not np.isfinite(q).all() for q in (anchor, end, coefficients)):
        return -float("inf")
    if np.any(coefficients < 0):
        return -float("inf")
    # New proofs carry plane-normal clearance directly, while the two legacy
    # fields remain in the record for mixed-version diagnostics. If any
    # present compatibility value is corrupt, reject the whole proof instead
    # of silently preferring a newer field.
    if "anchor_z_min" in reserve or "plane_limit" in reserve:
        if not np.isfinite(
            [reserve.get("anchor_z_min", np.nan), reserve.get("plane_limit", np.nan)]
        ).all():
            return -float("inf")
    if "anchor_surface_clearance" in reserve:
        anchor_clearance = float(reserve["anchor_surface_clearance"])
    elif np.isfinite([reserve["anchor_z_min"], reserve["plane_limit"]]).all():
        # Backward compatibility for horizontal-plane proofs created by older
        # workers. New proofs carry normal-distance clearance directly.
        anchor_clearance = float(reserve["anchor_z_min"] - reserve["plane_limit"])
    else:
        return -float("inf")
    if not np.isfinite(anchor_clearance):
        return -float("inf")
    envelope = np.maximum(np.abs(feedback - anchor), np.abs(end - anchor))
    return float(anchor_clearance - np.dot(coefficients, envelope))


def make_braking_reserve(
    layer, anchor, velocity, specification, measured, coefficients, *, clearance_lower=None
):
    anchor = np.asarray(anchor, dtype=np.float32)
    velocity = np.asarray(velocity, dtype=np.float64)
    acceleration = np.asarray(specification["acceleration"], dtype=np.float64)
    maximum_velocity = np.asarray(specification["max_velocity"], dtype=np.float64)
    if any(a.shape != (24,) or not np.isfinite(a).all()
           for a in (anchor, velocity, acceleration, maximum_velocity)):
        raise ValueError("Invalid braking vectors")
    if np.any(acceleration <= 0) or np.any(maximum_velocity <= 0):
        raise ValueError("Invalid braking limits")
    if np.any(np.abs(velocity) > maximum_velocity + 1e-7) or specification["dt"] != .01:
        raise ValueError("Invalid braking velocity or period")
    steps = int(np.ceil(np.max(np.abs(velocity) / (acceleration * .01))))
    if steps > 128:
        return None  # Bounded reserve, not an unbounded plan or deadline extension.
    evaluated = layer._evaluate(anchor)
    reserve = dict(
        anchor=anchor.copy(),
        velocity=velocity.copy(),
        acceleration=acceleration.copy(),
        dt=.01,
        steps=steps,
        anchor_z_min=evaluated.minimum_z.z,
        plane_limit=max(s.height + s.hard_margin for s in layer.surfaces.surfaces),
        anchor_surface_clearance=(
            layer.global_surface_clearance(anchor)
            if clearance_lower is None
            else float(clearance_lower)
        ),
    )
    reserve["end"], _ = braking_sample(reserve, steps)
    lower = np.asarray(specification["lower"], dtype=np.float32)
    upper = np.asarray(specification["upper"], dtype=np.float32)
    if (lower.shape != (24,) or upper.shape != (24,) or np.any(lower > upper)
            or not np.isfinite(lower).all() or not np.isfinite(upper).all()):
        raise ValueError("Invalid braking position bounds")
    if any(np.any(q < lower) or np.any(q > upper) for q in (anchor, reserve["end"])):
        return None
    return reserve if braking_clearance(reserve, measured, coefficients) > 1e-6 else None


def trajectory_clearance(proof, measured, coefficients):
    """Bound every continuous segment in a frozen trajectory plus fresh feedback.

    This is a global joint-box bound, NOT renewal by timestamp alone. A
    trajectory never changes after certification; only its measured endpoint
    is admitted again. Infinite tabletop planes deliberately over-approximate
    the configured finite surfaces.
    """
    radius = np.asarray(proof["radius"], dtype=np.float64)
    if radius.shape != (24,) or not np.isfinite(radius).all() or np.any(radius < 0):
        return -float("inf")
    # braking_clearance uses abs(end-anchor); encode the certified box radius.
    return braking_clearance(dict(proof, end=np.asarray(proof["anchor"], dtype=np.float64) + radius),
                             measured, coefficients)


def check_rolling_window(layer, measured, anchor, targets, raw, fallback, braking, coefficients):
    """One FK reference and conservative bounds cover the full window and stops.

    Reject if the global bound is too conservative. Never substitute a sampled
    clear result for a missing continuous-path certificate.
    """
    if coefficients is None:
        raise ValueError("Rolling trajectory requires global displacement bounds")
    points = np.asarray([anchor, *targets], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 24 or not np.isfinite(points).all():
        raise ValueError("Invalid frozen trajectory")
    hold = layer.guard(measured, anchor, raw_action=raw)
    fallback_hold = layer.guard(measured, fallback, raw_action=raw)
    evaluated = layer._evaluate(anchor)
    z = evaluated.minimum_z.z
    anchor_clearance = layer.global_surface_clearance(anchor)
    proof = dict(anchor=np.asarray(anchor).copy(), radius=np.max(np.abs(points-anchor), axis=0),
                 anchor_z_min=z,
                 plane_limit=max(s.height+s.hard_margin for s in layer.surfaces.surfaces),
                 anchor_surface_clearance=anchor_clearance)
    clearance = trajectory_clearance(proof, measured, coefficients)
    reserves = [make_braking_reserve(
        layer, q, v, braking, measured, coefficients,
        clearance_lower=anchor_clearance-float(
            np.dot(coefficients, np.abs(np.asarray(q, dtype=np.float64)-anchor))
        ))
        for q, v in zip(targets, braking["target_velocities"], strict=True)]
    # The shared reference can be overly pessimistic for a far-future stop.
    # Refine only those failed stops with exact FK at their own anchor. Keep
    # the SAME global continuous stopping bound; no sampled-only override.
    for i, reserve in enumerate(reserves):
        if reserve is None:
            reserves[i] = make_braking_reserve(layer, targets[i], braking["target_velocities"][i],
                                               braking, measured, coefficients)
    blocked = clearance <= 1e-6 or any(r is None for r in reserves) or hold.hard_limit_triggered
    decision = replace(hold, safe_action=np.asarray(anchor if blocked else targets[-1]).copy(),
                       hard_limit_triggered=blocked, action_scale=0.0 if blocked else 1.0,
                       path_z_min=proof["plane_limit"]+clearance,
                       safe_predicted_z_min=proof["plane_limit"]+clearance,
                       reason="rolling_bound_rejected" if blocked else "rolling_continuous_box_certified")
    target_holds = [replace(decision, safe_action=np.asarray(q).copy()) for q in targets]
    return dict(hold=hold, decision=decision, target_holds=target_holds,
                fallback_hold=fallback_hold, target_brakes=reserves, trajectory_box=proof)


def check_legacy_rolling_window(layer, measured, anchor, targets, raw, fallback):
    """Reuse the legacy sampled path/finite-table predicate without stop gates."""
    original = layer._evaluate
    margins = []
    cache = {}
    def evaluate(q):
        key = np.asarray(q, dtype=np.float64).tobytes()
        if key not in cache:
            cache[key] = layer._evaluate_for_feedback_reuse(q)
        value, margin = cache[key]
        margins.append(margin)
        return value
    layer._evaluate = evaluate
    try:
        hold, decision, target_holds, fallback_hold = check_window(
            layer, measured, anchor, targets, raw, include_proofs=True, fallback_anchor=fallback)
        motion_margin = min(margins)
        def hold_certificate(q):
            margins.clear()
            layer.guard(measured, q, raw_action=raw)
            return dict(reference=np.asarray(measured).copy(), margin=min(margins))
        anchor_certificate = hold_certificate(anchor)
        fallback_certificate = hold_certificate(fallback)
        target_certificates = [hold_certificate(q) for q in targets]
    finally:
        layer._evaluate = original
    return dict(hold=hold, decision=decision, target_holds=target_holds,
                fallback_hold=fallback_hold, target_brakes=None,
                sampled_feedback=dict(reference=np.asarray(measured).copy(), margin=motion_margin),
                anchor_sampled_feedback=anchor_certificate, fallback_sampled_feedback=fallback_certificate,
                target_sampled_feedback=target_certificates)


def sampled_feedback_displacement(proof, feedback, coefficients):
    """Bound movement of legacy sampled poses under a changed feedback endpoint.

    Every linear sample changes by at most the feedback endpoint change.
    The global link-length coefficients bound XYZ displacement as well as Z.
    No new motion, time extension alone, or unseen waypoint is authorized.
    """
    if coefficients is None:
        return float('inf')
    vectors = [np.asarray(x, dtype=np.float64) for x in (proof['reference'], feedback, coefficients)]
    if any(x.shape != (24,) or not np.isfinite(x).all() for x in vectors):
        return float('inf')
    reference, measured, bounds = vectors
    if np.any(bounds < 0) or any(np.any(q[[7,15]]<0) or np.any(q[[7,15]]>EFFECTOR_BOUND_M)
                                for q in (reference, measured)):
        return float('inf')
    displacement = float(np.dot(bounds, np.abs(measured-reference)))
    if not np.isfinite(proof['margin']) or displacement + 1e-6 >= proof['margin']:
        return float('inf')
    return displacement


def feedback_ready_for_check(feedback, task, previous_stamp, now):
    """Use the 50 Hz feedback clock as the geometry cadence, without drift.

    Ordinary checks require a new sample at least one period after the last
    checked sample. Urgent session reanchors may use fresh caller feedback.
    Rolling proofs revalidate feedback at consumption. They may start from the
    latest sample less than one feedback period old; forcing a 5 ms coincidence
    with post-dispatch submission needlessly skips every other sample.
    Non-rolling two-point certificates retain their original 5 ms requirement.
    """
    if feedback is None:
        return False
    stamp = feedback[1]
    freshness = CHECK_PERIOD_S if task.get("rolling") else .005
    return (np.isfinite(stamp) and 0 <= now - stamp < freshness
            and stamp >= task["submitted_at"] - freshness
            and (task.get("urgent") or stamp - previous_stamp >= CHECK_PERIOD_S - 1e-9))


def hold_displacement_coefficients(layer):
    """Global |delta Z| <= sum(coeff[i]*|delta q[i]|) for protected URDF points.

    For each protected frame, bound each ancestor lever arm by the downstream
    chain lengths plus the frame's geometry radius. Sum mimic contributions
    within that chain, then take the componentwise maximum over all frames.
    Other branches cannot move this frame and must not inflate its lever arms.
    Bounds remain global over all joint angles, not a local Jacobian estimate.
    Unknown/unbounded prismatic coordinates disable this fallback.
    """
    fk = layer.fk
    if not isinstance(fk, URDFForwardKinematics):
        return None
    parent_joints = {j.child: j for j in fk._ordered_joints}
    frame_radii = {}
    for point in layer.geometry.points:
        frame_radii[point.frame] = max(frame_radii.get(point.frame, 0.0),
                                      float(np.linalg.norm(point.xyz)) + point.radius)
    coefficients = np.zeros(24, dtype=np.float64)
    for frame, radius in frame_radii.items():
        chain_coefficients = np.zeros(24, dtype=np.float64)
        while frame != fk.root_link:
            joint = parent_joints.get(frame)
            if joint is None:
                return None
            if joint.kind != "fixed":
                rule = fk._rules[joint.name]
                if joint.kind == "prismatic":
                    if rule.action_index is not None and rule.action_index not in (7, 15):
                        return None
                    radius += abs(rule.offset) + abs(rule.multiplier) * EFFECTOR_BOUND_M
                if rule.action_index is not None:
                    factor = 1.0 if joint.kind == "prismatic" else radius
                    chain_coefficients[rule.action_index] += factor * abs(rule.multiplier)
            # This origin is upstream of this joint's axis, but downstream of
            # the next ancestor axis. Fixed origins also contribute to length.
            radius += float(np.linalg.norm(joint.origin[:3, 3]))
            frame = joint.parent
        coefficients = np.maximum(coefficients, chain_coefficients)
    return coefficients


def bounded_hold_is_safe(
    coefficients,
    reference,
    feedback,
    path_z_min,
    plane_limit,
    *,
    path_surface_clearance=None,
):
    if coefficients is None:
        return False
    feedback = np.asarray(feedback, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if feedback.shape != (24,) or reference.shape != (24,) or not np.isfinite(feedback).all():
        return False
    if not np.isfinite(reference).all() or not np.isfinite(path_z_min):
        return False
    if any(np.any(q[[7, 15]] < 0) or np.any(q[[7, 15]] > EFFECTOR_BOUND_M)
           for q in (reference, feedback)):
        return False
    displacement = float(np.dot(coefficients, np.abs(feedback - reference)))
    if path_surface_clearance is not None:
        return float(path_surface_clearance) - displacement > 1e-6
    # Compatibility for an older horizontal-plane proof.
    return path_z_min - displacement > plane_limit + 1e-6


def check_window(layer, measured, anchor, targets, raw, *, include_proofs=False, fallback_anchor=None,
                 braking=None, braking_coefficients=None, allow_unsafe_anchor=False):
    # Exact per-window memoization only. Shared endpoints need one FK each;
    # no tolerance, vertex reduction or reuse across feedback snapshots.
    original = layer._evaluate
    cache = {}

    def evaluate(configuration):
        key = np.asarray(configuration, dtype=np.float64).tobytes()
        if key not in cache:
            cache[key] = original(configuration)
        return cache[key]

    layer._evaluate = evaluate
    try:
        hold, motion = _check_window(
            layer, measured, anchor, targets, raw,
            allow_unsafe_future_anchor=(allow_unsafe_anchor or (fallback_anchor is not None
                                        and not np.array_equal(fallback_anchor, anchor))),
        )
        target_holds = [layer.guard(measured, q, raw_action=raw) for q in targets]
        if any(d.hard_limit_triggered for d in target_holds):
            failed = next(d for d in target_holds if d.hard_limit_triggered)
            motion = replace(failed, safe_action=anchor.copy(),
                             safe_predicted_z_min=hold.safe_predicted_z_min,
                             reason="collision_risk_hold_previous_safe_action", action_scale=0.0)
        motion = replace(motion, current_z_min=hold.current_z_min)
        if include_proofs and fallback_anchor is not None:
            # Prefetch can use an as-yet unexecuted future anchor. Independently
            # certify the command actually accepted at submission, so retiring
            # that future motion does not discard an otherwise usable hold.
            fallback_hold = layer.guard(measured, fallback_anchor, raw_action=raw)
            if braking is not None:
                reserves = [make_braking_reserve(layer, q, v, braking, measured, braking_coefficients)
                            for q, v in zip(targets, braking["target_velocities"], strict=True)]
                return hold, motion, target_holds, fallback_hold, reserves
            return hold, motion, target_holds, fallback_hold
        return (hold, motion, target_holds) if include_proofs else (hold, motion)
    finally:
        layer._evaluate = original


def _check_window(layer, measured, anchor, targets, raw, *, allow_unsafe_future_anchor=False):
    """Independently certify the fallback hold and both limited motion steps."""
    hold = layer.guard(measured, anchor, raw_action=raw)
    if hold.hard_limit_triggered:
        if allow_unsafe_future_anchor:
            # An unexecuted prefetched waypoint is a candidate, not the actual
            # hold command. Let the independently checked fallback decide.
            return hold, hold
        raise RuntimeError("Measured-to-hold path is unsafe; no certified fallback target")
    motion = layer.guard_trajectory(anchor, targets, raw_action=raw)
    # The robot can lag behind the accepted anchor. Check its direct path too.
    tracking = layer.guard_trajectory(measured, targets, raw_action=raw)
    if tracking.hard_limit_triggered:
        motion = tracking
    if motion.hard_limit_triggered:
        motion = replace(
            motion, safe_action=anchor.copy(), safe_predicted_z_min=hold.safe_predicted_z_min,
            reason="collision_risk_hold_previous_safe_action", action_scale=0.0,
        )
    return hold, motion


def stride_sample_indices(length, stride=2):
    """Fixed per-segment sampling, retaining both boundary waypoints."""
    if stride not in (1, 2):
        raise ValueError("Only stride 1 (offline baseline) and fixed stride 2 are supported")
    return sorted(set(range(0, length, stride)) | ({length - 1} if length else set()))


def sampled_escape_is_improving(layer, measured, task, committed_indices, target_indices):
    """Sampled egress: every vertex's penetration must not increase.

    Not a swept-volume proof. Check command geometry AND measured-offset
    geometry with actual gripper openings; never assume an untracked closure.
    """
    if not task.get("allow_sampled_escape"):
        return False
    measured = np.asarray(measured, dtype=np.float64)
    if layer._evaluate(measured).surface.clearance >= -layer.tolerance:
        return False
    anchor = np.asarray(task["fallback_anchor"], dtype=np.float64)
    points = [task["committed"][i] for i in committed_indices] + [task["targets"][i] for i in target_indices]

    def penetration(q):
        groups = layer.geometry.world_points(q, layer.fk)
        values = []
        for surface in layer.surfaces.surfaces:
            for _, _, xyz, radii in groups:
                depth = np.maximum(
                    0.0,
                    surface.hard_margin - surface.signed_distances(xyz, radii),
                )
                if surface.xy_bounds is not None:
                    xmin, xmax, ymin, ymax = surface.xy_bounds
                    inside = ((xyz[:, 0]+radii >= xmin) & (xyz[:, 0]-radii <= xmax)
                              & (xyz[:, 1]+radii >= ymin) & (xyz[:, 1]-radii <= ymax))
                    depth = np.where(inside, depth, 0.)
                values.append(depth)
        return np.concatenate(values)

    improved_actual = False
    for variant in ("command", "actual_opening", "feedback_offset"):
        def configuration(q):
            q = np.asarray(q, dtype=np.float64).copy()
            if variant == "feedback_offset":
                q += measured-anchor
            if variant != "command" and len(q) == 24:
                q[[7, 15]] = measured[[7, 15]]
            return q
        initial = previous = penetration(configuration(anchor))
        for q in points:
            current = penetration(configuration(q))
            if not np.isfinite(current).all() or np.any(current > previous + 1e-9):
                return False
            previous = current
        if variant == "feedback_offset":
            improved_actual = float(initial.max()-previous.max()) > 1e-6
    return improved_actual


def check_sampled_segment_window(layer, measured, task, *, stride=2):
    """Check the committed prefix and new suffix against one real snapshot.

    This is bounded-age sampled checking, not a continuous displacement proof.
    Production uses fixed stride 2, including near-table and recovery motion.
    Skipped waypoints are NOT geometrically certified. No midpoint sampling
    secretly reintroduces them. Stride 1 exists only for offline comparison.
    """
    committed = task.get("committed", [])
    # Stride reduces geometry only, never basic validation of skipped commands.
    for q in [*committed, *task["targets"]]:
        q = np.asarray(q, dtype=np.float64)
        if q.shape != np.asarray(measured).shape or not np.isfinite(q).all():
            raise ValueError("Invalid waypoint in sampled segment (including skipped points)")
    committed_indices = stride_sample_indices(len(committed), stride)
    target_indices = stride_sample_indices(len(task["targets"]), stride)
    sampled_committed = [committed[i] for i in committed_indices]
    sampled_targets = [task["targets"][i] for i in target_indices]
    original = layer._evaluate
    original_path_samples = layer.path_samples
    cache = {}
    inter_arm = layer.inter_arm_collision
    original_inter_arm_evaluate = None if inter_arm is None else inter_arm.evaluate
    inter_arm_cache = {}

    def evaluate_inter_arm(q, fk):
        """Reuse cross-chain FK/capsules within one immutable safety window.

        The generic surface evaluation already includes inter-arm geometry at
        stride-sampled points.  The explicit full-rate pass below covers every
        100 Hz waypoint.  Without this cache, sampled waypoints were evaluated
        twice, adding CPU jitter without adding geometric coverage.
        """
        key = np.asarray(q, dtype=np.float64).tobytes()
        if key not in inter_arm_cache:
            inter_arm_cache[key] = original_inter_arm_evaluate(q, fk)
        return inter_arm_cache[key]

    def evaluate(q):
        key = np.asarray(q, dtype=np.float64).tobytes()
        if key not in cache:
            cache[key] = original(q)
        return cache[key]

    layer._evaluate = evaluate
    if inter_arm is not None:
        inter_arm.evaluate = evaluate_inter_arm
    if stride == 2:
        layer.path_samples = 1  # Endpoints only, not the skipped middle action.
    try:
        fallback = task["fallback_anchor"]
        live = layer.guard(measured, fallback, raw_action=task["raw"])
        if sampled_committed and not live.hard_limit_triggered:
            _, live = check_window(
                layer, measured, fallback, sampled_committed, task["raw"],
                allow_unsafe_anchor=True)
        hold, motion, holds, fallback_hold = check_window(
            layer, measured, task["anchor"], sampled_targets, task["raw"],
            include_proofs=True, fallback_anchor=fallback, allow_unsafe_anchor=True)
        # Table and cross-arm geometry share the fixed stride-2 cadence.  The
        # two-stage capsule model keeps each selected check conservative while
        # avoiding a second full-rate geometry pass.
        inter_arm_report = None
        inter_arm_escape = False
        if layer.inter_arm_collision is not None:
            full_points = [
                *[committed[index] for index in committed_indices],
                *[task["targets"][index] for index in target_indices],
            ]
            evaluations = [
                layer.inter_arm_collision.evaluate(np.asarray(point), layer.fk)
                for point in full_points
            ]
            current_inter_arm = layer.inter_arm_collision.evaluate(
                np.asarray(measured), layer.fk
            )
            worst = min(evaluations, key=lambda item: item.clearance)
            inter_arm_report = {
                "minimum_separation_m": worst.separation,
                "left_geometry": worst.left_name,
                "right_geometry": worst.right_name,
                "phase": getattr(worst, "phase", "narrow"),
                "checked_waypoints": len(evaluations),
                "warning": worst.separation < layer.inter_arm_collision.warning_margin,
                "collision": worst.clearance < -layer.tolerance,
                "hard_margin_m": layer.inter_arm_collision.hard_margin,
                "warning_margin_m": layer.inter_arm_collision.warning_margin,
            }
            if worst.clearance < -layer.tolerance:
                clearances = [item.clearance for item in evaluations]
                inter_arm_escape = (
                    current_inter_arm.clearance < -layer.tolerance
                    and all(
                        later >= earlier - 1e-9
                        for earlier, later in zip(
                            [current_inter_arm.clearance, *clearances[:-1]],
                            clearances,
                            strict=True,
                        )
                    )
                    and clearances[-1] > current_inter_arm.clearance + 1e-6
                )
                if inter_arm_escape:
                    motion = replace(
                        motion,
                        hard_limit_triggered=False,
                        safe_action=np.asarray(task["targets"][-1]).copy(),
                        action_scale=1.0,
                        reason="inter_arm_escape_increases_clearance",
                    )
                    live = replace(
                        live,
                        hard_limit_triggered=False,
                        reason="inter_arm_escape_increases_clearance",
                    )
                else:
                    motion = replace(
                        motion,
                        hard_limit_triggered=True,
                        safe_action=np.asarray(task["anchor"]).copy(),
                        safe_predicted_z_min=hold.safe_predicted_z_min,
                        action_scale=0.0,
                        limiting_point=f"{worst.left_name}<->{worst.right_name}",
                        limiting_frame="dual_arm",
                        limiting_surface="inter_arm",
                        distance_to_surface=worst.separation,
                        reason="inter_arm_collision_hold_previous_safe_action",
                    )
        escape = inter_arm_escape
        if (live.hard_limit_triggered or motion.hard_limit_triggered) and motion.limiting_surface != "inter_arm":
            surface_escape = sampled_escape_is_improving(
                layer, measured, task, committed_indices, target_indices
            )
            escape = escape or surface_escape
            if surface_escape:
                motion = replace(motion, hard_limit_triggered=False, action_scale=1.,
                                 safe_action=np.asarray(task["targets"][-1]).copy(),
                                 reason="sampled_escape_reduces_penetration")
                live = replace(live, hard_limit_triggered=False, reason="sampled_escape_reduces_penetration")
        # Do not fabricate a hold certificate for an unchecked odd waypoint.
        target_holds = [None] * len(task["targets"])
        for index, proof in zip(target_indices, holds, strict=True):
            target_holds[index] = proof
        return dict(hold=hold, decision=motion, target_holds=target_holds,
                    fallback_hold=fallback_hold, target_brakes=None,
                    live_decision=live, sampled_snapshot=True, escape_approved=escape,
                    collision_action_stride=stride, sampled_target_indices=target_indices,
                    sampled_committed_indices=committed_indices,
                    evaluated_configurations=len(cache),
                    inter_arm=inter_arm_report,
                    target_sampled_feedback=[None] * len(task["targets"]))
    finally:
        layer._evaluate = original
        if inter_arm is not None:
            inter_arm.evaluate = original_inter_arm_evaluate
        layer.path_samples = original_path_samples


def _geometry_process(connection, config_path, path_samples, effector_envelope_ranges=None):
    try:
        layer = load_safety_layer(config_path, effector_envelope_ranges=effector_envelope_ranges).realtime_guard(path_samples=path_samples)
        cache_stats = install_exact_geometry_cache(layer)
        braking_coefficients = hold_displacement_coefficients(layer)
        affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        connection.send({"ready": True})
        while True:
            task = connection.recv()
            received_ns = time.perf_counter_ns()
            if task is None:
                return
            cpu_started_ns = time.process_time_ns()
            cache_before = cache_stats.copy()
            try:
                if task.get("rolling_legacy_geometry"):
                    result = check_sampled_segment_window(layer, task["measured"], task)
                elif task.get("rolling"):
                    result = check_rolling_window(layer, task["measured"], task["anchor"], task["targets"],
                                                  task["raw"], task["fallback_anchor"], task["braking"],
                                                  braking_coefficients)
                else:
                    checked = check_window(
                        layer, task["measured"], task["anchor"], task["targets"], task["raw"],
                        include_proofs=True, fallback_anchor=task.get("fallback_anchor", task["anchor"]),
                        braking=task.get("braking"), braking_coefficients=braking_coefficients,
                    )
                    hold, motion, target_holds, fallback_hold = checked[:4]
                    result = {"hold": hold, "decision": motion, "target_holds": target_holds,
                              "fallback_hold": fallback_hold,
                              "target_brakes": checked[4] if len(checked) == 5 else None}
            except Exception as error:
                result = {"error": f"{type(error).__name__}: {error}"}
            finished_ns = time.perf_counter_ns()
            result["geometry_timing"] = {
                "child_received_ns": received_ns,
                "child_finished_ns": finished_ns,
                "child_wall_ms": (finished_ns - received_ns) / 1e6,
                "child_cpu_ms": (time.process_time_ns() - cpu_started_ns) / 1e6,
                "child_affinity": affinity,
                "cache_delta": {key: value - cache_before[key] for key, value in cache_stats.items()},
            }
            connection.send(result)
    except (EOFError, BrokenPipeError):
        pass
    except Exception as error:
        try:
            connection.send({"error": f"Geometry worker: {type(error).__name__}: {error}"})
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


class AsyncSafetyWorker:
    def __init__(self, config_path, path_samples, read_state, on_result=None, *, effector_envelope_ranges=None):
        self.read_state = read_state
        self.on_result = on_result
        self.requests = queue.Queue(maxsize=1)
        self.results = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.error = None
        self.feedback = None
        self.feedback_error = None
        self.feedback_requested = threading.Event()
        self._feedback_publish_lock = threading.Lock()
        self.feedback_changed = threading.Condition(self._feedback_publish_lock)
        context = mp.get_context("spawn")  # Never fork an initialized vendor SDK.
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_geometry_process,
                                       args=(child, config_path, path_samples, effector_envelope_ranges), daemon=True,
                                       name="nexus-collision-50hz")
        self.process.start()
        child.close()
        self.thread = threading.Thread(target=self._run, daemon=True, name="nexus-safety-feedback")
        self.feedback_thread = threading.Thread(target=self._read_feedback, daemon=True, name="nexus-feedback-50hz")
        self.feedback_thread.start()
        self.thread.start()

    @staticmethod
    def _replace(mailbox, value):
        try:
            mailbox.get_nowait()
        except queue.Empty:
            pass
        mailbox.put_nowait(value)

    def submit(self, task):
        task.setdefault("submitted_at", time.perf_counter())
        self.feedback_requested.set()
        self._replace(self.requests, task)
        with self.feedback_changed:
            self.feedback_changed.notify_all()

    def poll(self):
        if self.error or self.feedback_error:
            raise RuntimeError(self.error or self.feedback_error)
        try:
            return self.results.get_nowait()
        except queue.Empty:
            return None

    def publish_feedback(self, state, started):
        snapshot = (np.asarray(state, dtype=np.float32).copy(), started)
        with self.feedback_changed:
            if self.feedback is None or started >= self.feedback[1]:
                self.feedback = snapshot
                self.feedback_changed.notify_all()

    def _wait_response(self, timeout):
        deadline = time.perf_counter() + timeout
        while not self.stop.is_set():
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return False
            if self.connection.poll(min(0.02, remaining)):
                return True
        return False

    def _read_feedback(self):
        while not self.stop.is_set() and not self.feedback_requested.wait(.02):
            pass
        while not self.stop.is_set():
            started = time.perf_counter()
            try:
                measured = np.asarray(self.read_state(), dtype=np.float32).copy()
                if measured.shape != (24,) or not np.isfinite(measured).all():
                    raise ValueError("Invalid safety feedback")
                # Timestamp at read START, so a slow GDK call cannot hide age.
                self.publish_feedback(measured, started)
            except Exception as error:
                self.feedback_error = f"Safety feedback unavailable: {type(error).__name__}: {error}"
                return
            self.stop.wait(max(0, CHECK_PERIOD_S - (time.perf_counter() - started)))

    def _run(self):
        try:
            if not self._wait_response(10):
                raise RuntimeError("Collision process startup timed out")
            ready = self.connection.recv()
            if not ready.get("ready"):
                raise RuntimeError(ready.get("error", "Collision process not ready"))
            previous_feedback_stamp = -float("inf")
            while not self.stop.is_set():
                try:
                    task = self.requests.get(timeout=0.02)
                except queue.Empty:
                    continue
                if self.stop.is_set():
                    return
                try:
                    # A second independent 20 ms compute timer drifted past
                    # feedback publication, causing unnecessary 40 ms gaps.
                    # Wake on publication, once per 50 Hz sample, not polling.
                    with self.feedback_changed:
                        while not self.stop.is_set():
                            try:
                                task = self.requests.get_nowait()
                            except queue.Empty:
                                pass
                            if self.feedback_error:
                                raise RuntimeError(self.feedback_error)
                            feedback = self.feedback
                            if feedback_ready_for_check(feedback, task, previous_feedback_stamp,
                                                        time.perf_counter()):
                                break
                            self.feedback_changed.wait(.02)
                    if self.stop.is_set():
                        return
                    measured, feedback_started = feedback
                    previous_feedback_stamp = feedback_started
                    started = time.perf_counter()
                    if (time.perf_counter() - feedback_started) * 1000 >= RESULT_MAX_AGE_MS:
                        raise RuntimeError("Safety feedback expired before geometric check")
                    self.connection.send(dict(task, measured=measured))
                    if not self._wait_response(0.2):
                        if self.stop.is_set():
                            return  # Owned worker shutdown is not a geometry timeout.
                        raise RuntimeError("Collision process response timed out")
                    result = self.connection.recv()
                    completed = time.perf_counter()
                    timing = result.get("geometry_timing")
                    if timing is not None:
                        timing["dispatch_to_child_ms"] = max(
                            0.0, timing["child_received_ns"] / 1e6 - started * 1000)
                        timing["child_to_parent_ms"] = max(
                            0.0, completed * 1000 - timing["child_finished_ns"] / 1e6)
                    result.update(task=task, measured=measured, started=feedback_started,
                                  compute_started=started,
                                  check_completed=completed,
                                  check_ms=(completed - started) * 1000)
                except Exception as error:
                    result = dict(task=task, error=f"Safety worker: {type(error).__name__}: {error}")
                if self.on_result is not None:
                    self.on_result(result)
                self._replace(self.results, result)
        except Exception as error:
            if not self.stop.is_set():
                self.error = f"Safety worker unavailable: {error}"

    def close(self):
        self.stop.set()
        with self.feedback_changed:
            self.feedback_changed.notify_all()
        self.feedback_thread.join(timeout=1.0)
        self.thread.join(timeout=1.0)
        if self.process.is_alive():
            self.process.terminate()  # This owned, geometry-only child has no hardware handle.
        self.process.join(timeout=1.0)
        self.connection.close()
        if self.thread.is_alive() or self.feedback_thread.is_alive():
            raise RuntimeError("Safety feedback read did not exit; refusing concurrent GDK release")
