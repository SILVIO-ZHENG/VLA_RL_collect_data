# Third-party notices and external dependencies

This repository publishes project source code and documentation. It excludes vendored framework implementations, installed dependencies, binary packages, model weights, recordings, and binary robot meshes. Text dependency declarations and a project-specific patch are retained so the integration can be inspected.

## Project license

The GitHub repository was initialized with an Apache License 2.0. That existing root license is preserved in `LICENSE`. Original copyright, license, and source notices in retained files remain intact.

## LeRobot

LeRobot is developed by Hugging Face and its contributors and distributed under Apache-2.0. Its upstream source is not bundled in this publication. The retained `thor_project/third_party/lerobot` directory contains dependency information, constraints, and the project's deployment patch.

- Upstream: <https://github.com/huggingface/lerobot>
- Preparation recipe: `thor_project/scripts/prepare_pi05_lerobot.py`
- Project patch: `thor_project/third_party/lerobot/pi05-deployment.patch`
- Model integration/reconstruction limits: `docs/VLA_TRAINING_AND_DEPLOYMENT.md`

The original archived runtime contained changes beyond the retained preparation patch. Recreating that exact runtime requires separately restoring or reconciling those changes and recording their provenance.

## AGIBOT Genie Sim robot descriptions

Retained robot descriptions, derived textual models, and geometry provenance refer to AGIBOT Genie Sim resources distributed under Mozilla Public License 2.0. Their original license/source files are preserved. Binary meshes are not included in this source-only export.

- Upstream: <https://github.com/AgibotTech/genie_sim>
- Recorded upstream commit: `e98a17c72840423984db37c062ed29036b3da13a`
- Robot source record: `thor_project/src/nexus_deploy/robot/g2a/assets/agibot_g2/SOURCE.json`
- Robot license: `thor_project/src/nexus_deploy/robot/g2a/assets/agibot_g2/LICENSE`
- Viewer license: `thor_project/src/nexus_deploy/viewer/assets/LICENSE`
- Additional original notices remain alongside offline simulator assets.

## Vendor SDKs and external model artifacts

The AGIBOT GDK, compatible robot services, external model projects, pretrained checkpoints, tokenizers, GPU wheels, and other installation resources are separate dependencies. Their licenses and acquisition requirements are not replaced by this repository's root license.

References to FlowPRO, pi0.5, SAC, IQL, or other research explain the integration context. They do not imply authorship of those methods, distribution of their model artifacts, or a completed reproduction of their training results.
