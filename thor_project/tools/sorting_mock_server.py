"""Temporary scheduler integration simulator. Never opens a robot connection."""
from __future__ import annotations

import argparse
import copy
import json
import os
import threading
import time
import uuid
from pathlib import Path

from nexus_deploy.control_api import DeploymentJob, NexusHTTPServer, _handler_factory
from nexus_deploy.sorting_api import SortingController


class MockPlane:
    enable_actions = False
    csrf_token = "simulation-internal-only"

    def __init__(self, state: Path):
        self._lock = threading.RLock()
        self._jobs = {}
        self.state = state
        if state.exists():
            self._jobs = {j["id"]: DeploymentJob.from_dict(j) for j in json.loads(state.read_text())}

    def _persist_state(self):
        self.state.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state.with_suffix(".tmp")
        tmp.write_text(json.dumps([j.to_dict() for j in self._jobs.values()]))
        tmp.replace(self.state)

    def validate(self, config):
        return {"valid": True, "normalized": dict(config, robot_id="simulation-only")}

    def create_deployment(self, config, token, *, batch):
        job = DeploymentJob(uuid.uuid4().hex, dict(config, robot_id="simulation-only"),
                            status="ready_to_start", batch=copy.deepcopy(batch))
        self._jobs[job.id] = job
        self._persist_state()
        return job.to_dict()

    def start_deployment(self, job_id, token, *, sorting_authorized):
        assert sorting_authorized
        def run():
            job = self._jobs[job_id]
            for stage in ("running", "paused", "resetting", "simulated"):
                with self._lock:
                    job.status = "succeeded" if stage == "simulated" else "task_starting" if stage == "resetting" else stage
                    job.task_session["status"] = stage
                    job.updated_at = time.time()
                    if stage == "simulated":
                        job.batch["status"] = "simulated"
                        job.batch["completion_evidence"] = {
                            "physical_execution": False, "source": "synthetic-integration-test",
                            "reset_status": "not_executed",
                        }
                    self._persist_state()
                time.sleep(1)
        threading.Thread(target=run, daemon=True).start()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18768)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--token-file", type=Path, required=True)
    p.add_argument("--callback-url")
    p.add_argument("--timeout", type=float, default=120)
    args = p.parse_args()
    os.environ["G2A_SORTING_TOKEN"] = args.token_file.read_text().strip()
    plane = MockPlane(args.state)
    sorting = SortingController(plane, {"profiles": {"simulation": {"presence_gate": {"enabled": True}}},
                                       "callback_url": args.callback_url})
    server = NexusHTTPServer((args.host, args.port), _handler_factory(
        plane, port=args.port, allowed_hosts=(args.host,), sorting=sorting))
    timer = threading.Timer(args.timeout, server.shutdown)
    timer.daemon = True
    timer.start()
    sorting.start()
    print(f"SIMULATION ONLY http://{args.host}:{args.port}; timeout={args.timeout}s", flush=True)
    try:
        server.serve_forever()
    finally:
        sorting.close()
        server.server_close()
        timer.cancel()


if __name__ == "__main__":
    main()
