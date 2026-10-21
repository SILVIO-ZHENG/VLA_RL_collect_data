"""Narrow robot command boundary; deployment remains the real safety transport."""
from __future__ import annotations

import time
import uuid

from nexus_deploy.presence_gate import GateAction


class RobotPort:
    def __init__(self, plane):
        self.plane = plane
        self.simulated_results = {}  # offline tests may choose succeeded, failed or unknown

    def begin(self, job_id, action, *, simulated=False):
        p = self.plane
        with p._lock:
            job = p._jobs[job_id]
            record = dict(command_id=uuid.uuid4().hex, task_id=job_id,
                          generation=int(job.task_session.get("generation", 0)),
                          action=action, status="accepted", simulated=simulated,
                          accepted_at=time.time())
            job.task_session.setdefault("robot_commands", []).append(record)
            p.tasks.persist()  # no physical command before durable intent
            return record

    def finish(self, job_id, record, status):
        with self.plane._lock:
            record.update(status=status, finished_at=time.time())
            self.plane.tasks.persist()

    def execute(self, job_id, action, operation):
        record = self.begin(job_id, action)
        try:
            result = operation()
        except Exception:
            self.finish(job_id, record, "unknown")
            raise
        self.finish(job_id, record, "succeeded")
        return result

    def start_session(self, robot, model, job_id, config):
        return self.execute(job_id, "start", lambda: self.plane.deployment.start_robot_session(robot, model, job_id, config))

    def stop_session(self, robot, job_id):
        return self.execute(job_id, "stop", lambda: self.plane.deployment.stop_robot_session(robot, job_id))

    def pause_session(self, robot, job_id, paused):
        action = "pause" if paused else "resume"
        return self.execute(job_id, action, lambda: self.plane.deployment.set_robot_session_paused(robot, job_id, paused))

    def reset_body(self, robot, safety_profile_id, job_id, *, timeout_s=70.0):
        return self.execute(job_id, "reset", lambda: self.plane.deployment.reset_robot_with_dahuan_cycle(
            robot, safety_profile_id, timeout_s=timeout_s))

    def apply_simulated_presence(self, job_id, generation, action):
        p = self.plane
        if p.enable_actions:
            raise RuntimeError("simulated robot results require disabled hardware actions")
        with p._lock:
            current = p._gate_action_current_locked(job_id, generation, action)
            if current is None:
                return
            job, gate = current
            record = self.begin(job_id, action.value, simulated=True)
            outcome = self.simulated_results.get(action.value, "succeeded")
            if outcome not in {"succeeded", "failed", "unknown"}:
                raise ValueError("invalid simulated command result")
            self.finish(job_id, record, outcome)
            if outcome != "succeeded":
                gate.acknowledge(action, applied=False)
                job.task_session["status"] = "recovery_required" if outcome == "unknown" else "failed"
                job.batch.update(status="recovery_required" if outcome == "unknown" else "failed",
                                 reason=f"simulated {action.value} {outcome}")
                p._event(job, "recovery_required" if outcome == "unknown" else "failed",
                         f"模拟 {action.value} 回执 {outcome}", job.progress)
                return
            if action is GateAction.PAUSE and job.status == "running":
                gate.acknowledge(action, applied=True)
                job.task_session.update(status="paused", pause_reason="vision")
                job.presence_gate.update(auto_paused=True, pending_action=None, action_status="simulated")
                p._event(job, "paused", "模拟机器人软暂停已确认", 96)
            elif action is GateAction.RESUME and job.status == "paused" and job.task_session.get("pause_reason") == "vision":
                gate.acknowledge(action, applied=True)
                job.task_session.update(status="running", pause_reason=None)
                job.presence_gate.update(auto_paused=False, pending_action=None, action_status="simulated")
                p._event(job, "running", "模拟机器人恢复已确认", 96)
            elif action is GateAction.RESET and job.status == "paused" and gate.auto_paused:
                if job.task_session.get("completion_conflict"):
                    job.status = "recovery_required"
                    job.task_session["status"] = "recovery_required"
                    p.tasks.persist()
                    return
                job.task_session["status"] = "resetting"
                p._event(job, "task_starting", "正在模拟复位回执", 96)
                gate.acknowledge(action, applied=True)
                gate.mark_reset_complete()
                job.robot_reset.update(status="read_only", effectors_commanded=False,
                                       started_at=time.time(), finished_at=time.time())
                p._finish_presence_task(job_id, simulated=True)
