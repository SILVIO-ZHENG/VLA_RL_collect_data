"""Offline fixed-clock chunk joins. Not connected to the robot or its queue.

A join changes the arm reference, not collision authority or device events.
Velocity/acceleration feasibility is NOT guaranteed by a cubic polynomial.
"""

import numpy as np


class ChunkJoin:
    def __init__(
        self,
        actions,
        start,
        velocity,
        *,
        fps,
        first=0,
        duration=0.0,
        arm_indices=tuple(range(7)) + tuple(range(8, 15)),
        device_indices=(7, 15),
    ):
        self.actions = np.array(actions, dtype=np.float32, copy=True)
        self.start = np.array(start, dtype=np.float32, copy=True)
        self.velocity = np.array(velocity, dtype=np.float32, copy=True)
        if (
            self.actions.ndim != 2
            or not len(self.actions)
            or self.start.shape != self.actions.shape[1:]
            or self.velocity.shape != self.start.shape
            or not all(np.isfinite(x).all() for x in (self.actions, self.start, self.velocity))
        ):
            raise ValueError("Invalid finite action/anchor dimensions")
        if not np.isfinite(fps) or fps <= 0 or not np.isfinite(duration) or duration < 0:
            raise ValueError("Invalid frequency or join duration")
        if type(first) is not int or not 0 <= first < len(self.actions):
            raise ValueError("Invalid first action")
        self.fps, self.first = float(fps), first
        self.arms, self.devices = list(arm_indices), list(device_indices)
        indices = self.arms + self.devices
        if len(set(indices)) != len(indices) or any(i < 0 or i >= len(self.start) for i in indices):
            raise ValueError("Invalid or overlapping indices")
        self.duration = min(float(duration), (len(self.actions) - first) / fps)
        if self.duration:
            end, end_velocity = self.linear(self.duration)
            T = self.duration
            self.coefficients = np.array(
                [
                    self.start[self.arms],
                    self.velocity[self.arms],
                    3 * (end[self.arms] - self.start[self.arms]) / T**2
                    - (2 * self.velocity[self.arms] + end_velocity[self.arms]) / T,
                    -2 * (end[self.arms] - self.start[self.arms]) / T**3
                    + (self.velocity[self.arms] + end_velocity[self.arms]) / T**2,
                ],
                dtype=np.float64,
            )

    def linear(self, elapsed):
        if not np.isfinite(elapsed):
            raise ValueError("Nonfinite sample time")
        progress = max(0.0, elapsed) * self.fps
        complete = int(np.floor(progress + 1e-9))
        alpha = float(np.clip(progress - complete, 0, 1))
        right_index = self.first + complete
        if right_index >= len(self.actions):
            return self.actions[-1].copy(), np.zeros_like(self.start)
        left = self.start if complete == 0 else self.actions[right_index - 1]
        right = self.actions[right_index]
        q = np.asarray(left + (right - left) * alpha, dtype=np.float32)
        v = (right - left) * self.fps
        q[self.devices] = right[self.devices]
        v[self.devices] = 0
        return q, v

    def sample(self, elapsed):
        q, v = self.linear(elapsed)
        if self.duration and 0 <= elapsed < self.duration:
            a, b, c, d = self.coefficients
            q[self.arms] = a + elapsed * (b + elapsed * (c + elapsed * d))
            v[self.arms] = b + elapsed * (2 * c + 3 * elapsed * d)
        return q, v


class HandoffOffset:
    """Arm-only C1 offset at the actual queued source handoff, not installation.

    The underlying clock/knots stay unchanged. This is still only an offline
    candidate: a nonzero offset changes the physical path and needs rechecking.
    """

    def __init__(self, reference, *, elapsed, position, velocity, duration):
        if not np.isfinite(duration) or duration <= 0 or not np.isfinite(elapsed):
            raise ValueError("Invalid handoff time")
        self.reference = reference
        self.elapsed = elapsed
        self.duration = duration
        self.arms = reference.arms
        q, v = reference.sample(elapsed)
        position, velocity = np.asarray(position), np.asarray(velocity)
        if (
            position.shape != q.shape
            or velocity.shape != v.shape
            or not np.isfinite(position).all()
            or not np.isfinite(velocity).all()
        ):
            raise ValueError("Invalid handoff state")
        self.delta = (position - q)[self.arms].astype(np.float64)
        self.delta_v = (velocity - v)[self.arms].astype(np.float64)

    def sample(self, elapsed):
        q, v = self.reference.sample(elapsed)
        u = (elapsed - self.elapsed) / self.duration
        if 0 <= u < 1:
            h00 = 2 * u**3 - 3 * u**2 + 1
            h10 = u**3 - 2 * u**2 + u
            q[self.arms] += h00 * self.delta + h10 * self.duration * self.delta_v
            v[self.arms] += (6 * u**2 - 6 * u) / self.duration * self.delta
            v[self.arms] += (3 * u**2 - 4 * u + 1) * self.delta_v
        return q, v
