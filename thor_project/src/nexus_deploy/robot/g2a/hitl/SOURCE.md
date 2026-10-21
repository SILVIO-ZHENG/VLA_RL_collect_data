# HITL source snapshot

The `common/`, `g2_runtime/`, `offline/ingest.py`, and `schemas/` files were
copied on 2026-10-16 from
`5090:/home/luoxiaolong/workspace/lerobot-flowpro` at commit `f61b3bfd`.
The remote working tree contained uncommitted changes, including
`common/collection_manifest.py`; this snapshot reflects the working tree, not
only the commit. The original implementation targets the older FlowPRO G2
client; the Nexus safety and dispatch adaptations are described below.

The sidecar contract accepts the selected model cadence and Dahuan profile;
`offline/__main__.py` pulls a sealed episode to Thor, verifies its join, and
archives it by content hash. The Nexus G2 runner now gives rollback and Human
handoff an exclusive action slot. Its service waits for accepted startup HOLD
before DDS starts. Deployment selects that service only for `human_in_loop`
physical execute sessions with enforced workspace safety; read-only task
preparation stays on the ordinary client. The selected inference mode and task
parameters pass through unchanged. An independent, read-only G2 VR subscriber watches for two
complete Y clicks only while a saved manual task is startable. Thor ends that
subscriber before calling the same `start_deployment` admission path as the
web button. One tablet recording is one parcel/sidecar; a single Nexus task
can bind and seal several recordings in sequence, waiting for each seal before
the next Y activation.

The Nexus safety adapter now exposes per-call dispatch receipts and a way to
retire unissued motion. Its preplanned-trajectory branch validates exact
rollback waypoints against the final velocity, acceleration, position, and
existing rolling collision checks. `g2_runtime/dispatch_ledger.py` tracks
accepted points by queue provenance, rejects a skipped or modified point, and
caps the path after X release. Simulated rolling-queue and startup-order tests
cover these components. Physical rollback still needs a controlled G2 hardware
acceptance run covering vendor takeover, X release/repress, tablet sealing, and
Thor offline archive before operators rely on the new entry point.
