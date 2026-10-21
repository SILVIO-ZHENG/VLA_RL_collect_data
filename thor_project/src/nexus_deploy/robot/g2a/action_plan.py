"""Thread-safe, time-sampled action chunks for the standalone robot client."""

# 中文｜执行链位置：client.py 把模型输出映射为控制目标并整块平滑之后，调用 install；
# 伺服线程再 reserve 取样，经调用方的控制／安全处理发送后，以 commit 记录真实发送值。
# 本类只保存一个可被新响应替换的时间轨迹块，并非逐块累积的 FIFO，也不直接调用 GDK。
# 数组契约为单点 (D,) 或整块 (N,D) 的 float32；生产 D=24，具体关节顺序见 constants.py。
# 连续关节通常为 rad；夹爪通道已由 adapter 映射到所选硬件目标单位，本类不再次换单位。
# 时间为 perf_counter 同源的秒；所有 action index、drop、remaining 都按策略帧计数。
# English | client.py maps model outputs to control targets and shapes the full chunk before install.
# The servo thread reserves a sample, applies caller-owned conditioning/safety, sends it, then commits
# the delivered value. This is one replaceable timed chunk, not a FIFO and not a GDK command sender.
# Inputs are float32 (D,) vectors or (N,D) chunks; production D=24 follows constants.py. Continuous
# joints normally use radians; effector channels retain adapter-selected device units. Times use
# perf_counter-compatible seconds; indices, dropped steps, and remaining steps count policy frames.

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PlanSnapshot:
    # 中文｜请求发出时的只读锚点；generation 区分 clear 前后，chunk_id 区分块替换。
    # next_action_index 是已成功提交进度后的下一个策略点，不是伺服循环次数。
    # English | Read-only request anchor: generation separates clear/reset epochs and chunk_id
    # identifies replacement. next_action_index tracks committed policy progress, not servo ticks.
    chunk_id: str | None
    next_action_index: int
    remaining_actions: int
    consumed_since_install: int
    generation: int = 0


@dataclass(frozen=True)
class InstallResult:
    # 中文｜installed=False 时旧计划保留；dropped_steps 是原响应前缀长度，不是发送失败数。
    # installed_at 使用秒，wait_ms 只计等待在途 reservation 释放及取得安装锁的时间。
    # English | Failed installation preserves the old plan. dropped_steps counts skipped response
    # prefix rows. installed_at is seconds; wait_ms measures the installation hand-off wait.
    installed: bool
    dropped_steps: int
    reason: str = ""
    installed_at: float | None = None
    wait_ms: float = 0.0
    fixed_horizon_projection: dict[str, float | int] | None = None
    rtc_handoff: dict[str, float | int | str] | None = None


@dataclass(frozen=True)
class ActionReservation:
    """One interpolated servo sample reserved but not yet delivered."""

    # 中文｜token 标识一次在途取样；action_index 是插值右端点，next_action_index 是提交进度。
    # 高频伺服多次取样可能 advanced_policy_steps=0；慢周期也可能一次跨越多个策略点。
    # English | token identifies one in-flight sample. action_index is the interpolation's right
    # endpoint; next_action_index is committed progress. A servo sample may advance zero or many steps.

    token: int
    chunk_id: str
    action_index: int
    next_action_index: int
    advanced_policy_steps: int
    action: np.ndarray = field(repr=False, compare=False)
    generation: int = 0
    is_hold: bool = False
    sample_time: float = 0.0


class ActionPlan:
    """Own a 30 Hz policy chunk and expose continuous time-sampled actions.

    The policy/action indices always remain in policy-frequency units. A caller
    may reserve samples at any higher servo rate without changing the meanings
    of ``next_action_index``, ``dropped_steps`` or ``replan_after_steps``.
    """

    def __init__(self, action_dim: int, policy_fps: float = 30.0, *,
                 position_target_indices=(), chunk_projector=None,
                 interpolation_mode: str = "linear", cubic_indices=None) -> None:
        # 中文｜position_target_indices 直接输出当前区间目标；cubic_indices 才做三次插值。
        # 两组必须互斥，避免把设备目标寄存器解释为需要连续平滑的物理关节轨迹。
        # English | Position-target channels publish interval goals directly. Cubic channels receive
        # continuous interpolation, so the two sets must be disjoint rather than smoothing device registers.
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if not math.isfinite(policy_fps) or policy_fps <= 0:
            raise ValueError("policy_fps must be positive and finite")
        self.action_dim = action_dim
        self.position_target_indices = tuple(position_target_indices)
        if len(set(self.position_target_indices)) != len(self.position_target_indices) or any(
            not isinstance(i, int) or i < 0 or i >= action_dim for i in self.position_target_indices
        ):
            raise ValueError("Invalid position-target indices")
        if interpolation_mode not in {"linear", "monotone-cubic"}:
            raise ValueError("Invalid action interpolation mode")
        self.cubic_indices = (tuple(i for i in range(action_dim)
                                    if i not in self.position_target_indices)
                              if cubic_indices is None else tuple(cubic_indices))
        if (len(set(self.cubic_indices)) != len(self.cubic_indices)
                or any(not isinstance(i, int) or i < 0 or i >= action_dim
                       or i in self.position_target_indices for i in self.cubic_indices)):
            raise ValueError("Invalid cubic interpolation indices")
        self.policy_fps = float(policy_fps)
        self.interpolation_mode = interpolation_mode
        self.chunk_projector = chunk_projector
        self._condition = threading.Condition()
        self._actions: np.ndarray | None = None
        self._chunk_id: str | None = None
        self._first_index = 0
        self._next_index = 0
        self._valid_end_index = 0
        self._consumed_since_install = 0
        self._trajectory_start_time: float | None = None
        self._trajectory_start_action: np.ndarray | None = None
        self._trajectory_start_velocity = np.zeros(action_dim, dtype=np.float32)
        self._last_sent_action: np.ndarray | None = None
        self._last_sent_velocity = np.zeros(action_dim, dtype=np.float32)
        self._last_sent_time: float | None = None
        self._next_token = 0
        self._reservation: ActionReservation | None = None
        self._generation = 0

    def snapshot(self) -> PlanSnapshot:
        # 中文｜锁内一次性读取块身份与进度，避免通信线程拿到两个不同安装时刻的数据。
        # English | Read identity and progress under one lock so a request sees one coherent plan state.
        with self._condition:
            remaining = 0 if self._actions is None else max(0, self._valid_end_index - self._next_index)
            return PlanSnapshot(
                self._chunk_id,
                self._next_index,
                remaining,
                self._consumed_since_install,
                self._generation,
            )

    def ready_for_request(
        self,
        mode: str,
        min_replan_after_steps: int,
        request_lead_steps: int,
    ) -> bool:
        # 中文｜空计划／耗尽总是可请求；sync 有剩余就等待。RTC 只看剩余长度是否进入 lead。
        # 其余异步模式还要求已消费 min_replan_after_steps；本函数不发请求也不保证网络成功。
        # English | Empty/exhausted plans request immediately; sync waits while work remains. RTC uses
        # the lead threshold alone; other async modes also require minimum consumed policy steps.
        snapshot = self.snapshot()
        if snapshot.chunk_id is None or snapshot.remaining_actions == 0:
            return True
        if mode == "rtc":
            # Requests are serialized, so the preceding inference already
            # provides the minimum replan interval.  Waiting for another step
            # after installing its response would shorten the default RTC
            # prefix from 30 to 29 before the next request even starts.
            return snapshot.remaining_actions <= request_lead_steps
        return (
            mode != "sync"
            and snapshot.consumed_since_install >= min_replan_after_steps
            and snapshot.remaining_actions <= request_lead_steps
        )

    def preview_at(self, now: float, *, period: float, count: int) -> np.ndarray | None:
        """Read an immutable clock-aligned preview without reserving/consuming.

        Refuse to extend beyond the old trusted horizon. This is a reference
        preview for diagnostics/conditioning, never a collision certificate.
        """
        if not math.isfinite(now) or not math.isfinite(period) or period <= 0 or count < 2:
            raise ValueError("Invalid RTC preview clock")
        with self._condition:
            if self._actions is None or self._trajectory_start_time is None:
                return None
            end = self._trajectory_start_time + (self._valid_end_index - self._first_index) / self.policy_fps
            if now + (count - 1) * period > end + 1e-9:
                return None
            return np.stack([self._sample_locked(now + i * period)[0] for i in range(count)])

    def set_hold_action(self, action: np.ndarray) -> None:
        """Set the safe action sent before the first chunk and during plan gaps."""
        # 中文｜调用者提供安全保持目标；这里仅建立插值起点并清零估计速度，不验证碰撞。
        # English | The caller supplies the hold target. This establishes an interpolation anchor and
        # zero velocity; it does not perform collision validation or transmit a hold command.
        array = self._validate_action(action)
        with self._condition:
            if self._reservation is not None:
                raise RuntimeError("Cannot replace hold action while a servo sample is reserved")
            self._last_sent_action = array.copy()
            self._last_sent_velocity.fill(0.0)
            self._last_sent_time = None

    def reserve(self, now: float | None = None) -> ActionReservation | None:
        """Reserve the interpolated action for ``now`` without advancing progress."""
        # 中文｜先预留再发送再提交：发送期间 install 会等待，使新块从真实最后发送点衔接。
        # 若发送未完成，应由调用者 cancel；同一时刻只允许一个 reservation，不能并发重复消费。
        # English | Reserve -> send -> commit keeps installation behind an in-flight send. Cancel an
        # abandoned send attempt. At most one reservation may exist; reserve itself consumes no progress.
        sample_time = time.perf_counter() if now is None else float(now)
        if not math.isfinite(sample_time):
            raise ValueError("sample time must be finite")
        with self._condition:
            if self._reservation is not None:
                raise RuntimeError("An action is already reserved")
            sampled = self._sample_locked(sample_time)
            if sampled is None:
                return None
            action, next_index, action_index = sampled
            self._next_token += 1
            self._reservation = ActionReservation(
                token=self._next_token,
                chunk_id=self._chunk_id or "",
                action_index=action_index,
                next_action_index=next_index,
                advanced_policy_steps=max(0, next_index - self._next_index),
                action=action,
                generation=self._generation,
                is_hold=self._actions is None or self._next_index >= self._valid_end_index,
                sample_time=sample_time,
            )
            return self._reservation

    def commit(self, reservation: ActionReservation, sent_action: np.ndarray | None = None,
               *, advance: bool = True) -> None:
        """Commit one successfully delivered 100 Hz sample."""
        # 中文｜sent_action 应是经过滤波／限幅后实际成功交付的值，避免新块锚到未执行原值。
        # advance=False 仍更新发送历史但不消费策略点；clear 改代后也不能让旧 reservation 复活。
        # 速度=(本次实际值-上次实际值)/两次采样时间差，单位为各通道单位每秒。
        # English | sent_action should be the successfully delivered conditioned/clipped command.
        # advance=False records output without consuming policy progress. A retired generation cannot
        # revive the plan. Velocity uses delivered differences divided by sample-time seconds.
        with self._condition:
            self._validate_reservation(reservation)
            delivered = reservation.action if sent_action is None else self._validate_action(sent_action)
            if reservation.generation == self._generation and advance:
                self._next_index = reservation.next_action_index
                self._consumed_since_install += reservation.advanced_policy_steps
            # A concurrent clear retires the plan, not a GDK call already in
            # flight. Still record its actual output without reviving the plan.
            if self._last_sent_action is not None and self._last_sent_time is not None:
                dt = reservation.sample_time - self._last_sent_time
                if dt > 0:
                    self._last_sent_velocity = np.asarray(
                        (delivered - self._last_sent_action) / dt, dtype=np.float32
                    )
            self._last_sent_action = delivered.copy()
            self._last_sent_time = reservation.sample_time
            self._reservation = None
            self._condition.notify_all()

    def preview_reserved(self, reservation: ActionReservation, *, now: float,
                         period: float, count: int = 15) -> np.ndarray:
        """Snapshot future samples without consuming or replacing the reservation."""
        # 中文｜返回 (count,D)：首行是已预留目标，后续按 now+i*period 取样；period 单位秒。
        # 只读预览不推进队列，也不等价于安全审核；已退休／保持中的 reservation 不允许预览。
        # English | Return (count,D): reserved first sample plus future samples spaced by period seconds.
        # This read-only preview neither consumes the plan nor certifies safety; hold/retired samples fail.
        samples, _indices = self.preview_reserved_with_indices(
            reservation, now=now, period=period, count=count
        )
        return samples

    def preview_reserved_with_indices(self, reservation: ActionReservation, *, now: float,
                                      period: float, count: int = 15) -> tuple[np.ndarray, tuple[int, ...]]:
        """Return each preview point with its original policy action index."""
        if count < 2 or not math.isfinite(period) or period <= 0 or not math.isfinite(now):
            raise ValueError("Invalid trajectory preview")
        with self._condition:
            self._validate_reservation(reservation)
            if reservation.generation != self._generation or reservation.is_hold:
                raise RuntimeError("Cannot preview a retired or holding reservation")
            future = [self._sample_locked(now + i * period) for i in range(1, count)]
            samples = np.stack([reservation.action.copy()] + [item[0] for item in future])
            indices = (reservation.action_index,) + tuple(item[2] for item in future)
            return samples, indices

    def cancel(self, reservation: ActionReservation) -> bool:
        # 中文｜只撤销匹配的在途 token，不更改已提交动作或策略时钟；唤醒等待安装的线程。
        # English | Release only the matching in-flight token, preserving committed output and time base.
        with self._condition:
            if not self._matches(reservation):
                return False
            self._reservation = None
            self._condition.notify_all()
            return True

    def install(
        self,
        chunk_id: str,
        actions: np.ndarray,
        anchor: PlanSnapshot,
        *,
        stale_steps: int = 0,
        quality_horizon: int | None = None,
        now: float | None = None,
        max_dropped_steps: int | None = None,
        stale_origin_monotonic: float | None = None,
        rtc_handoff_horizon: int | None = None,
        rtc_handoff_indices=(),
        rtc_guard_old_tail: bool = False,
        before_publish: Callable[[InstallResult], None] | None = None,
    ) -> InstallResult:
        """Atomically install a chunk and blend from the last sent command.

        The first selected policy action is reached after one policy period.
        Later points use the selected interpolation at the caller's servo frequency.
        """
        # 中文｜质量窗口限定原响应 [0,valid_end)；安装保留原始索引，实际只执行 [dropped,valid_end)。
        # 这是替换当前块，不是把新块追加到旧块尾部；拒绝任何非有限、空或维度错误的块。
        # English | quality_horizon bounds original response rows [0,valid_end). Installation retains
        # original indices but executes only [dropped,valid_end); this replaces rather than appends a chunk.
        array = np.asarray(actions, dtype=np.float32)
        if array.ndim != 2 or array.shape[-1] != self.action_dim or not np.isfinite(array).all():
            return InstallResult(False, 0, f"invalid action chunk: {array.shape}")
        if len(array) == 0:
            return InstallResult(False, 0, "empty action chunk")
        if not chunk_id:
            return InstallResult(False, 0, "empty chunk_id")
        if stale_steps < 0:
            return InstallResult(False, 0, "stale_steps must be non-negative")
        valid_end = len(array) if quality_horizon is None else min(len(array), quality_horizon)
        if valid_end <= 0:
            return InstallResult(False, 0, "quality_horizon must be positive")
        if now is not None and not math.isfinite(float(now)):
            return InstallResult(False, 0, "non-finite install time")
        if max_dropped_steps is not None and (
            type(max_dropped_steps) is not int or max_dropped_steps < 0
        ):
            return InstallResult(False, 0, "invalid maximum dropped steps")
        if stale_origin_monotonic is not None and not math.isfinite(float(stale_origin_monotonic)):
            return InstallResult(False, 0, "invalid stale time origin")
        if rtc_handoff_horizon is not None and (
            type(rtc_handoff_horizon) is not int or not 0 < rtc_handoff_horizon <= valid_end
        ):
            return InstallResult(False, 0, "invalid RTC handoff horizon")
        if type(rtc_guard_old_tail) is not bool:
            return InstallResult(False, 0, "invalid RTC old-tail guard")
        handoff_indices = tuple(rtc_handoff_indices)
        if len(set(handoff_indices)) != len(handoff_indices) or any(
            type(i) is not int or not 0 <= i < self.action_dim or i in self.position_target_indices
            for i in handoff_indices
        ):
            return InstallResult(False, 0, "invalid RTC handoff indices")
        waiting_started = time.perf_counter()

        with self._condition:
            # Do not splice a new chunk while a GDK call is in flight. This
            # makes the boundary start exactly the last successfully sent value.
            while self._reservation is not None:
                self._condition.wait()
            # 中文｜等待没有本地超时，因此发送方必须最终 commit/cancel；实际安装后才起新时钟。
            # English | This wait has no local timeout: the sender must eventually commit/cancel before
            # the newly selected trajectory starts its clock at actual installation.
            ready_at = time.perf_counter()
            # Production starts the interpolation at actual installation,
            # not response arrival or before an in-flight SDK send completed.
            # Explicit now remains available for deterministic offline clocks.
            install_time = ready_at if now is None else float(now)
            wait_ms = max(0.0, ready_at - waiting_started) * 1000
            if anchor.generation != self._generation:
                return InstallResult(False, 0, "response belongs to a cleared plan generation")
            if anchor.chunk_id is None:
                if self._chunk_id is not None:
                    return InstallResult(False, 0, "initial response arrived after another chunk")
                progress_dropped = 0
            else:
                if self._chunk_id != anchor.chunk_id:
                    return InstallResult(False, 0, "response anchor no longer matches active chunk")
                progress_dropped = max(0, self._next_index - anchor.next_action_index)
            # 中文｜RTC/异步替换的过期前缀取两者最大值，不能相加：两者描述同一段已流逝时间。
            # progress 来自请求锚点到安装时的已提交策略进度，stale_steps 由调用方按观测年龄计算。
            # max_dropped_steps 约束剩余 RTC 混合预算；预算耗尽时保持旧计划并拒绝此次响应。
            # English | Drop max(committed progress since request, caller-provided observation-age steps),
            # not their sum: both describe elapsed time. max_dropped_steps protects the remaining RTC blend
            # budget; exhaustion rejects this response while leaving the previous plan intact.
            dropped = max(progress_dropped, stale_steps)
            # Opt-in state-aligned RTC must also age while waiting for a send
            # to finish. Consumer progress alone misses time spent holding.
            if stale_origin_monotonic is not None:
                if float(stale_origin_monotonic) > install_time:
                    return InstallResult(False, dropped, "stale time origin is in the future", wait_ms=wait_ms)
                dropped = max(dropped, math.ceil(
                    max(0.0, install_time - float(stale_origin_monotonic)) * self.policy_fps
                ))
            if max_dropped_steps is not None and dropped > max_dropped_steps:
                return InstallResult(False, dropped, "RTC blend budget exhausted while waiting to install", wait_ms=wait_ms)
            if dropped >= valid_end:
                return InstallResult(False, dropped, "entire trusted response became stale")

            projection = None
            installed_actions = np.ascontiguousarray(array.copy())
            start = self._last_sent_action
            handoff = None
            if (rtc_handoff_horizon is not None and handoff_indices
                    and anchor.chunk_id is not None and self._actions is not None
                    and self._trajectory_start_time is not None and start is not None):
                # Blend only still-trusted old/new futures at the SAME wall
                # time. Never append expired rows, stretch time, extrapolate
                # beyond the old quality horizon, or touch device events.
                old_end_time = (self._trajectory_start_time
                                + (self._valid_end_index - self._first_index) / self.policy_fps)
                old_steps = max(0, math.floor((old_end_time - install_time) * self.policy_fps + 1e-9))
                steps = min(rtc_handoff_horizon - dropped, valid_end - dropped, old_steps)
                if steps >= 2:
                    indices = list(handoff_indices)
                    original = installed_actions[dropped:dropped + steps, indices].copy()
                    old_future = np.stack([
                        self._sample_locked(install_time + (i + 1) / self.policy_fps)[0][indices]
                        for i in range(steps)
                    ])
                    old_delta = float(np.max(np.abs(old_future[0] - start[indices])))
                    new_delta = float(np.max(np.abs(original[0] - start[indices])))
                    # The already-executed output is the physical anchor. In
                    # rare RTC divergence, the old model's future can be much
                    # farther from that output than the new trajectory. Mixing
                    # it back in would recreate an obsolete motion excursion.
                    if rtc_guard_old_tail and old_delta > 0.12 and old_delta > 2.0 * max(new_delta, 1e-6):
                        handoff = dict(
                            mode="bypass-divergent-old-tail", steps=0, old_trusted_steps=old_steps,
                            first_delta_before_rad=new_delta, first_delta_after_rad=new_delta,
                            max_adjustment_rad=0.0, old_tail_delta_rad=old_delta,
                        )
                    else:
                        u = np.arange(1, steps + 1, dtype=np.float64) / steps
                        weight = (u ** 3 * (10.0 - 15.0 * u + 6.0 * u ** 2)).astype(np.float32)
                        blended = old_future + weight[:, None] * (original - old_future)
                        blended[-1] = original[-1]  # exact new endpoint; no tail offset
                        if not np.isfinite(blended).all():
                            return InstallResult(False, dropped, "non-finite RTC handoff", wait_ms=wait_ms)
                        installed_actions[dropped:dropped + steps, indices] = blended
                        handoff = dict(
                            mode="trusted-tail-quintic", steps=steps, old_trusted_steps=old_steps,
                            first_delta_before_rad=new_delta,
                            first_delta_after_rad=float(np.max(np.abs(blended[0] - start[indices]))),
                            max_adjustment_rad=float(np.max(np.abs(blended - original))),
                        )
            if self.chunk_projector is not None:
                # 中文｜可选投影仅处理将执行的后缀，以实际发送位置/速度为起点，保持点数和点间距。
                # English | Optional projection reshapes only the executable suffix from actual output
                # position/velocity, keeping sample count and policy spacing unchanged.
                if start is None:
                    return InstallResult(False, dropped, "fixed-horizon projection has no command anchor")
                projected, projection = self.chunk_projector.project(
                    installed_actions[dropped:valid_end],
                    start_position=start,
                    start_velocity=self._last_sent_velocity,
                )
                installed_actions[dropped:valid_end] = projected
            result = InstallResult(
                True,
                dropped,
                installed_at=install_time,
                wait_ms=wait_ms,
                fixed_horizon_projection=projection,
                rtc_handoff=handoff,
            )
            if before_publish is not None:
                before_publish(result)
            self._actions = installed_actions
            self._chunk_id = chunk_id
            self._first_index = dropped
            self._next_index = dropped
            self._valid_end_index = valid_end
            self._consumed_since_install = 0
            self._trajectory_start_time = install_time
            # 中文｜重新锚定时间不是可变速 retime：每段仍为 1/policy_fps 秒，且不拉长整块追终点。
            # 连续通道从最后发送点走向首个保留点；若从未有发送锚点，则首点同时充当起点。
            # English | Reanchoring the clock is not variable-duration retiming: each interval still lasts
            # 1/policy_fps seconds. Continuous channels start at last output, or at the first row if absent.
            self._trajectory_start_action = (
                installed_actions[dropped].copy()
                if start is None else np.asarray(start, dtype=np.float32).copy()
            )
            self._trajectory_start_velocity = self._last_sent_velocity.copy()
            return result

    def clear(self) -> None:
        """Retire the plan, allowing an in-flight send to commit its hold anchor."""
        # 中文｜递增 generation 使旧请求失效，但保留最后发送状态；在途发送仍可完成并建立保持锚点。
        # English | Increment generation to retire pending responses while keeping the last sent state.
        # A send already in flight may still commit its output as the hold anchor.
        with self._condition:
            self._generation += 1
            self._actions = None
            self._chunk_id = None
            self._first_index = 0
            self._next_index = 0
            self._valid_end_index = 0
            self._consumed_since_install = 0
            self._trajectory_start_time = None
            self._trajectory_start_action = None
            self._trajectory_start_velocity.fill(0.0)
            self._condition.notify_all()

    def _sample_locked(self, now: float) -> tuple[np.ndarray, int, int] | None:
        # 中文｜调用者已持锁。没有计划就返回最后发送值；连该值都没有则返回 None。
        # 输出三元组为 (D维目标, 下一个策略索引, 当前插值右端索引)，不在此提交任何进度。
        # English | Called with the condition held. With no plan, return the last output or None.
        # Return (D-vector target, next policy index, right endpoint index) without committing progress.
        if self._actions is None or self._trajectory_start_time is None:
            if self._last_sent_action is None:
                return None
            return self._last_sent_action.copy(), self._next_index, max(0, self._next_index - 1)

        available = self._valid_end_index - self._first_index
        elapsed = max(0.0, now - self._trajectory_start_time)
        elapsed_policy_steps = elapsed * self.policy_fps
        # 中文｜s=(now-start)*fps，k=floor(s+1e-9)，alpha=clip(s-k,0,1)。首段左端为发送锚点，
        # 后续左端为前一策略点；这让 30Hz 策略可由 100Hz 等任意频率采样，索引仍属于 30Hz。
        # English | s=(now-start)*fps; k=floor(s+1e-9); alpha=clip(s-k,0,1). The first left endpoint is
        # the output anchor, later ones are preceding policy rows. Servo rate does not redefine policy indices.
        completed = min(available, int(math.floor(elapsed_policy_steps + 1e-9)))
        alpha = float(np.clip(elapsed_policy_steps - completed, 0.0, 1.0))
        # Absolute timestamps can place an exact 30 Hz boundary a few ulps
        # below its integer value (for example 0.99999999999998).
        next_index = self._first_index + completed

        if completed >= available:
            # 中文｜首次跨过末端时给末行；末端已被提交后保持实际发送值，防止滤波值回跳到原末行。
            # English | First exhaustion yields the final row; after exhaustion is committed, hold the
            # actual delivered output rather than jumping back from its conditioned value to the raw endpoint.
            last_index = self._valid_end_index - 1
            if self._next_index >= self._valid_end_index and self._last_sent_action is not None:
                return self._last_sent_action.copy(), self._valid_end_index, last_index
            return self._actions[last_index].copy(), self._valid_end_index, last_index

        right_index = self._first_index + completed
        left = self._trajectory_start_action if completed == 0 else self._actions[right_index - 1]
        right = self._actions[right_index]
        action = np.asarray(left + (right - left) * alpha, dtype=np.float32)
        if self.interpolation_mode == "monotone-cubic" and self.cubic_indices:
            # 中文｜按通道计算 Hermite 三次插值：斜率单位为目标单位/秒，乘段长 1/fps 后进入基函数。
            # 内点斜率用同号割线的调和平均；拐向处为零。首段继承速度仅保留同向分量且截到 3*割线。
            # 这里只改善插值形状与速度连续性，不构成加速度、jerk 或碰撞安全约束。
            # English | Channelwise Hermite interpolation uses target-units/second tangents multiplied by
            # 1/fps. Interior tangents use the harmonic mean of same-sign secants, zero at reversals; the
            # initial tangent keeps only matching direction and is capped at three times the segment slope.
            # This interpolation is not an acceleration, jerk, or collision-safety certificate.
            current_slope = (right - left) * self.policy_fps
            if completed == 0:
                # Only carry a previous velocity pointing into the new
                # segment, and bound the tangent to prevent overshoot.
                start_slope = np.where(
                    self._trajectory_start_velocity * current_slope > 0,
                    np.sign(current_slope) * np.minimum(
                        np.abs(self._trajectory_start_velocity), 3.0 * np.abs(current_slope)
                    ), 0.0,
                )
            else:
                previous_left = (self._trajectory_start_action if completed == 1
                                 else self._actions[right_index - 2])
                previous_slope = (left - previous_left) * self.policy_fps
                start_slope = self._monotone_slope(previous_slope, current_slope)
            if right_index + 1 < self._valid_end_index:
                next_slope = (self._actions[right_index + 1] - right) * self.policy_fps
                end_slope = self._monotone_slope(current_slope, next_slope)
            else:
                end_slope = current_slope
            a2 = alpha * alpha
            a3 = a2 * alpha
            cubic_action = np.asarray(
                (2 * a3 - 3 * a2 + 1) * left
                + (a3 - 2 * a2 + alpha) * (start_slope / self.policy_fps)
                + (-2 * a3 + 3 * a2) * right
                + (a3 - a2) * (end_slope / self.policy_fps),
                dtype=np.float32,
            )
            action[list(self.cubic_indices)] = cubic_action[list(self.cubic_indices)]
        # At each policy interval, hold that interval's position goal. The
        # body retains its existing interpolation and the same policy clock.
        if self.position_target_indices:
            # 中文｜设备目标在区间开始立即取 right，不等待 alpha=1；夹爪事件不会被身体插值渐变。
            # English | Device targets take right immediately at the interval start, independent of alpha;
            # their event values therefore bypass continuous body interpolation.
            indices = list(self.position_target_indices)
            action[indices] = right[indices]
        return action, next_index, right_index

    @staticmethod
    def _monotone_slope(before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """Equal-interval PCHIP tangent: continuous velocity without overshoot."""
        # 中文｜同号时 m=2ab/(a+b)，异号或任意一侧为零时 m=0；where 避免在无效位置除零。
        # English | m=2ab/(a+b) for same-sign secants, otherwise zero; masked division avoids invalid terms.
        same_direction = (before * after) > 0
        return np.divide(
            2.0 * before * after, before + after,
            out=np.zeros_like(before), where=same_direction,
        )

    def _validate_action(self, action: np.ndarray) -> np.ndarray:
        array = np.asarray(action, dtype=np.float32)
        if array.shape != (self.action_dim,) or not np.isfinite(array).all():
            raise ValueError(f"Expected one finite {self.action_dim}D action, got {array.shape}")
        return array

    def _matches(self, reservation: ActionReservation) -> bool:
        current = self._reservation
        return (
            current is not None
            and current.token == reservation.token
            and current.chunk_id == reservation.chunk_id
            and current.action_index == reservation.action_index
        )

    def _validate_reservation(self, reservation: ActionReservation) -> None:
        if not self._matches(reservation):
            raise RuntimeError("Action reservation is no longer active")
