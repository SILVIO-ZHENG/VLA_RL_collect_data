# LeRobot dependency

This directory keeps the reproducible PI0.5 dependency recipe in one place.

- `pi05-deployment.patch` contains the reviewed inference-only changes.
- `requirements-5090.lock` pins the validated x86_64 package set.
- `source/` is a generated checkout of the pinned upstream revision. It is
  ignored by Git and must be recreated with `scripts/prepare_pi05_lerobot.py`.

Verify an existing checkout without downloading or starting a service:

```bash
.venv/bin/python scripts/prepare_pi05_lerobot.py --verify
```
