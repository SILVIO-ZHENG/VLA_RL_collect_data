# From HITL Recordings to VLA Training and Deployment

This guide connects the repository's G2 collection runtime to downstream learning.
It distinguishes implemented deployment code from conversion and training work that must be supplied externally.
All commands below are examples for an isolated Linux environment; no training run or robot deployment is implied.

## 1. What this source release provides

The main project is under [`thor_project`](../thor_project).
Its G2 adapter, HITL state machine, sidecar contracts, offline ingestion and pi05 serving adapter are project source.
Third-party libraries, pretrained checkpoints, tokenizers, datasets, wheel archives and binary robot meshes are excluded.
An importable external LeRobot installation and platform-specific robot assets remain deployment prerequisites.

| Stage | Present in this repository | Additional work or assets |
|---|---|---|
| Collect policy observations and interventions | G2 HITL runtime and sidecar writer | Vendor GDK, recorder, VR/DDS infrastructure and robot configuration |
| Verify and ingest a sealed recording | Sidecar validation and immutable raw-store code | Accessible recordings and matching metadata |
| Convert raw recordings to supervised episodes | Data contracts provide inputs | A time-alignment and LeRobot export adapter |
| Train a VLA with supervised targets | No project training executable | An external compatible LeRobot environment and pretrained model |
| Train an actor and critic | No project actor-critic learner | Rewards, transitions, replay, learner and policy adapter |
| Train from preference pairs | Preference-pair validator | Pair construction, objective and optimization implementation |
| Serve pi05 on G2 | WebSocket server and model adapter | Compatible external model code, checkpoint, tokenizer and processors |
| Serve another architecture on G2 | Shared protocol and deployment framework | A model-specific server adapter and acceptance checks |

See [`policy.py`](../thor_project/src/nexus_deploy/vla/pi05/policy.py) for the implemented pi05 boundary.
The runtime does not automatically turn a saved episode into a trained policy.

## 2. Choose the learning objective before exporting

For supervised fine-tuning, each example supplies observations and a desired future action sequence.
Good candidates include successful demonstrations, verified human corrections and accepted autonomous rollouts.
The model learns to reproduce those targets through its own supervised objective.
For pi05, that objective is action flow matching; it does not introduce a critic by itself.

For actor-critic learning, export transitions with observations, actual actions, rewards, next observations and terminal semantics.
Specify whether the learner's action is a joint target, a delta, an action chunk or a residual over a base policy.
The environment transition must correspond to the action representation that the actor and critic actually use.
An intervention flag is evidence of control source; it is not automatically a calibrated scalar reward.

For preference learning, compare compatible action branches under the same observation and representation.
The rejected candidate, physically executed negative and accepted correction have different meanings.
Preserve their provenance rather than substituting one for another to fill a missing label.
The strict pair gate in [`contract.py`](../thor_project/src/nexus_deploy/robot/g2a/hitl/common/contract.py) must remain effective.

## 3. Robot dimensions and the training representation

G2's canonical network vector contains 24 coordinates in this order:

| Canonical indices | Meaning | Native round-1 policy |
|---|---|---|
| 0-6 | Left arm, seven joints | Included |
| 7 | Left effector command/state | Included |
| 8-14 | Right arm, seven joints | Included |
| 15 | Right effector command/state | Included |
| 16-18 | Three head joints | Excluded |
| 19-23 | Five waist joints | Excluded |

The round-1 collection contract requires a 16D arm/effector policy and no policy control of head or waist.
The serving adapter supports compatible 16D and 24D pi05 checkpoints more generally.
For 16D inference, the adapter expands native actions into 24D while preserving the request's head/waist state.
That transport expansion does not turn the model into a 24D learned controller.
Use [`constants.py`](../thor_project/src/nexus_deploy/robot/g2a/constants.py) as the ordering authority.

Arm coordinates are joint positions in radians.
Effector units and direction depend on the selected profile and cannot be inferred from vector length.
A proposed 16D training export should use the first 16 canonical names and explicitly identify the effector profile.
Do not mix Dahuan and legacy Omnipicker samples merely because both have 16 columns.

## 4. Define a supervised export schema

The following is a proposed adapter output, not a schema already emitted by a converter in this repository.
Use the exact camera names expected by the checkpoint, or implement and record an explicit rename mapping.
The existing pi05 adapter constructs these image feature names from its three protocol camera keys:

| Feature | Proposed stored value | Required interpretation |
|---|---|---|
| `observation.state` | `float32[16]` | Measured native arm/effector state at the observation time |
| `observation.images.cam_head` | RGB image/video frame | Physical head camera |
| `observation.images.cam_left_wrist` | RGB image/video frame | Physical left wrist camera |
| `observation.images.cam_right_wrist` | RGB image/video frame | Physical right wrist camera |
| `action` | `float32[16]` | Absolute targets in the model's declared action coordinates |
| `task` | English instruction string | Actual episode/subtask instruction |
| Standard episode/timestamp fields | Generated by the dataset writer | Fixed-rate, ordered episode membership |

Image height and width are dataset decisions that must match the selected checkpoint's preprocessing contract.
Decode to RGB explicitly; OpenCV's default BGR order must not leak into the dataset.
Store physical state and physical/model action units before normalization.
Retain UUIDs, source hashes, control-source intervals and export decisions in a separate provenance index.
That index joins training frames back to sealed raw recordings without treating JSON event records as model features.

For a critic dataset, add a separate documented transition schema rather than overloading the supervised action column.
At minimum record `obs`, `action`, `reward`, `next_obs`, `terminated`, `truncated` and action duration.
For chunk-level decisions, record the executed portion and elapsed duration when a chunk is interrupted.

## 5. Implement alignment before writing training rows

1. Accept only verified, sealed sidecars and their matching vendor recordings.
2. Verify the collection manifest, checkpoint/processor identity and robot/effector profile.
3. Resolve recorder, camera, inference and execution timestamps through the recorded clock mapping.
4. Select a declared target timeline, for example 30 Hz when that matches the model contract.
5. Associate each observation with the corresponding valid camera frames and measured state.
6. Select the intended action label and map it into the model's action coordinates.
7. Cut or label control-source changes, rollback, resets, aborts and invalid spans explicitly.
8. Split episodes before any temporal-window or action-chunk sampling.
9. Write the dataset and a provenance/rejection report, then read it back independently.

Thirty-hertz action data and the 100 Hz robot servo are different timelines.
Do not export every servo tick as a 30 Hz action or assume a new inference happened at each tick.
Choose interpolation rules separately for joints, images, discrete events and gripper transitions.
Avoid blending across a reset, missing execution span or a change in control authority.
The collector's rollback path is a physical recovery operation; it is not automatically a positive demonstration.

The sidecar records exact inference observations and native candidate actions where available.
Those candidates answer what the policy proposed; executed evidence answers what the robot accepted and did.
For human correction labels, use verified human-controlled targets/trajectory evidence with the correct time alignment.
Using measured next-state as a command target is a modeling choice with lag; document it if unavoidable.
Never silently use a rejected candidate as the accepted human correction.

## 6. LeRobot writing API: adapter pseudocode

The original archive's external LeRobot API was inspected for `create`, `add_frame`, `save_episode` and `finalize`.
That library source is intentionally absent from this source-only publication.
The pseudocode below assumes a compatible external version and leaves raw-recording extraction as required adapter work.
It is not a ready-to-run converter and must not be presented as one.

```python
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Adapter supplies exact names, image shapes and validated episode selection.
names16 = canonical_g2_names()[:16]  # Implement from the project's constants.
camera_shapes = validated_camera_shapes()  # RGB (height, width, 3).
features = {
    "observation.state": {"dtype": "float32", "shape": (16,), "names": names16},
    "action": {"dtype": "float32", "shape": (16,), "names": names16},
}
for key, shape in camera_shapes.items():
    features[f"observation.images.{key}"] = {
        "dtype": "video", "shape": shape, "names": ["height", "width", "channels"],
    }

dataset = LeRobotDataset.create(
    repo_id="local/g2_dahuan_hitl_v1",
    root="/data/exports/g2_dahuan_hitl_v1",  # New output directory.
    fps=30,
    robot_type="g2a",
    features=features,
    use_videos=True,
)
for episode in select_verified_episodes():  # Required adapter implementation.
    for sample in align_and_resample(episode, fps=30):  # Required adapter.
        row = {
            "observation.state": np.asarray(sample.state16, dtype=np.float32),
            "action": np.asarray(sample.model_action16, dtype=np.float32),
            "task": sample.english_task,
        }
        for key, rgb in sample.images.items():
            row[f"observation.images.{key}"] = rgb  # HWC uint8 RGB.
        validate_export_row(row, sample.provenance)  # Required adapter.
        dataset.add_frame(row)
    dataset.save_episode()
dataset.finalize()
```

The writer produces dataset metadata, statistics, episode indexing and image/video assets.
The inspected dataset layout uses chunked Parquet under `data/`, episode Parquet under `meta/episodes/`,
`meta/info.json`, `meta/stats.json`, `meta/tasks.parquet` and per-camera files under `videos/`.
Treat `meta/info.json` and the selected library version as authoritative instead of hardcoding filenames in a reader.
Read back complete episodes, verify frame counts and inspect action/image alignment before training.

## 7. Normalization, grippers and relative actions

Normalize only after the physical representation and action direction have been settled.
Fit statistics on the training split and preserve the exact statistics used by every saved checkpoint.
pi05 commonly uses quantile normalization; ensure the required quantiles exist and are appropriate to the chosen processor.
Changing statistics changes the learned and deployed coordinate transform, even if the model weights are unchanged.
The current project pi05 adapter loads saved checkpoint processors and checks relative-action consistency.

The Dahuan profile has a particularly important asymmetry:

| Quantity | Zero | 0.025 | Unit |
|---|---|---|---|
| Hardware state/target | Closed | Open | Metres |
| Model action | Open command | Close command | Metres |

The client converts model action into hardware target once through the effector profile.
For Dahuan, the inverse mapping of a physical gripper target is `model_action = 0.025 - hardware_target`.
Do not apply this inversion to the measured observation state.
When building labels from executed hardware targets, invert only the action coordinates at indices 7 and 15.
Legacy Omnipicker uses a separate radian range and identity direction mapping.
See [`effector_profiles.py`](../thor_project/src/nexus_deploy/robot/g2a/effector_profiles.py).

For an absolute joint policy, the target is an absolute joint position.
For a relative-arm policy, a preprocessing step may form `arm_target - arm_state` at the chunk anchor.
Grippers can remain absolute while arms are relative; use explicit feature names and exclusion rules.
Statistics must describe the transformed action distribution when relative targets are normalized.
At inference, restore absolute actions once; the network protocol to the client remains absolute 24D.
Adding state again in the client would apply the delta twice.

## 8. External training environment and version selection

Use a separate Python 3.12 environment for training, independent of the robot's GDK process.
The project declares runtime dependencies in [`pyproject.toml`](../thor_project/pyproject.toml).
Its historical pi05 recipe pinned Torch 2.11.0, Torchvision 0.26.0, Transformers 5.5.4 and PEFT 0.20.0.
Those are recorded recipe versions, not a promise that every GPU/OS combination has compatible wheels.
The historical x86_64 constraints are distinct from an ARM64/Thor environment.

The external setup pattern below requires a reviewed full LeRobot commit SHA before use:

```bash
uv venv --python 3.12 /opt/g2-training-venv
git clone https://github.com/huggingface/lerobot.git /opt/g2-lerobot
cd /opt/g2-lerobot
export G2_LEROBOT_REV='<reviewed-full-commit-sha>'
git checkout --detach "$G2_LEROBOT_REV"
# Check this revision's extras and CLI before installing or training.
uv pip install --python /opt/g2-training-venv/bin/python -e '.[training,pi,peft-dep]'
/opt/g2-training-venv/bin/python -m lerobot.scripts.lerobot_train --help
```

The shown extras and CLI match the API inspected in the original archive; another revision may differ.
Pin a compatible Torch/CUDA/video-decoder set and record the resolved environment after validation.
Keep model/tokenizer assets outside the source repository and record their revisions and hashes.
If a gated tokenizer requires access, obtain it through its own provider's normal access process.
The [official pi05 guide](https://huggingface.co/docs/lerobot/main/en/pi05) is an external reference;
its `main` documentation evolves and does not establish compatibility with this captured deployment.

### Exact archived dependency reconstruction is incomplete

[`prepare_pi05_lerobot.py`](../thor_project/scripts/prepare_pi05_lerobot.py) records upstream base commit
`a9879e69edefb3c308f7025046d3d12b3cb2fe50` and a reviewed four-file deployment patch.
The historical patch is retained as a dependency recipe under `thor_project/third_party/lerobot/`.
That recipe requires Git metadata and verifies an exact patch; it is not an installer for arbitrary LeRobot releases.
The original October 19 source archive contained additional trained-RTC inference code absent from that patch.
This GitHub publication excludes that third-party source snapshot, so it cannot reconstruct that exact runtime by itself.
Obtain the matching externally maintained dependency snapshot or regenerate, review and validate a complete patch.
Do not bypass source verification or claim that an upstream checkout alone reproduces archived trained-RTC behavior.

## 9. pi05 supervised fine-tuning example

Run this template only after implementing the converter, validating the exported dataset and choosing a compatible checkpoint.
The checkpoint must carry matching 16D features, camera mappings, action semantics and usable pretrained weights.
This example fine-tunes from an existing compatible G2 checkpoint; it does not create compatibility from a filename.

```bash
/opt/g2-training-venv/bin/python -m lerobot.scripts.lerobot_train \
  --policy.path=/data/models/g2_pi05_base/pretrained_model \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/g2_dahuan_hitl_v1 \
  --dataset.root=/data/exports/g2_dahuan_hitl_v1 \
  --dataset.eval_split=0.1 \
  --batch_size=8 \
  --steps=10000 \
  --output_dir=/data/runs/g2_pi05_hitl_v1 \
  --wandb.enable=false
```

The CLI fields above were checked against the archived training configuration API; the command has not been executed here.
Verify them again in the selected external revision and adjust batch size to measured memory use.
Choose episode-held-out splits before fitting statistics; also hold out package instances, collection days or scenes when relevant.
The export adapter must freeze training-only statistics; the CLI's episode split alone does not establish that property.
For an existing dataset, preserve documented split assignments rather than allowing a new split on each experiment.
LoRA, expert-only tuning and full fine-tuning are different configurations with different memory and checkpoint requirements.
Set them explicitly using the chosen library's supported options and validate how deployment loads or merges adapters.

The inspected external training loop uses dataset metadata, a policy factory, processors, backward and optimizer updates.
That generic loop does not read this project's sidecar directly or automatically apply FlowPRO preference losses.
Store source revision, dataset revision, export manifest, seed, optimizer settings and checkpoint/processor hashes with each run.

## 10. Training-time RTC versus inference-time RTC

RTC means real-time chunking: new action predictions must account for action commitments and inference delay.
The project server supports guided/trained routing and checks checkpoint delay metadata and execution horizon.
Its trained route passes the previous prefix, delay and execution horizon to the external pi05 implementation.
The original archived model imported `prepare_inference_prefix` and implemented that trained inference path.

The archive also contained a `sample_training_prefix_mask` helper, but its pi05 training forward did not call that helper.
The presence of that helper and a nonzero `rtc_training_max_delay` field is therefore insufficient evidence of RTC training.
Do not set a checkpoint metadata field and describe an ordinary fine-tuning run as trained-RTC learning.
Continuing trained-RTC training requires the correct training-time prefix/noise/loss construction from a compatible implementation.
Preserve episode-tail masks and the allowed delay distribution, and test both the learning objective and inference route.

For trained inference, the adapter checks `delay <= execution_horizon <= chunk_size - delay` for the configured maximum delay.
The captured Thor registrations use an execution horizon of 29 for their trained-RTC checkpoints.
These historical settings must be checked against a new checkpoint rather than copied as universal constants.

## 11. How actor-critic improvements can reach a VLA

An actor-critic policy and a VLA usually have different architectures, parameter shapes and input/output interfaces.
There is no meaningful general operation that copies critic/actor weights into pi05 or another VLA.
Use an explicit data or objective bridge instead:

| Bridge | Training material | Additional implementation |
|---|---|---|
| Rollout distillation | Verified successful actor rollouts with images, state, task and actions | Dataset export and ordinary VLA supervised training |
| Human correction replay | Accepted intervention segments mixed with prior demonstrations | Selection, weighting and regression safeguards |
| Advantage-weighted fitting | Action targets plus comparable advantage estimates | Weight calculation, clipping, provenance and supported weighted loss |
| Preference optimization | Same-observation accepted/rejected branches | Valid pair builder and a suitable VLA preference objective |
| Direct VLA RL | VLA policy, rewards and on-policy/off-policy interaction | A model-specific RL learner and deployment integration |

The simplest data bridge is to execute the improved actor, validate task outcomes, and export successful behavior as demonstrations.
Record whether the label came from the actor, the VLA or a human so failures and biases remain traceable.
Keep the original demonstrations and mix deliberately to reduce forgetting and narrow-scene overfitting.
If using critic-based weights, calibrate them within the same task, reward scale and action representation.
Offline action-value estimates outside the replay distribution are not automatically reliable teacher labels.
For residual RL, reconstruct the final target action from base plus residual before supervised export, or document a residual VLA target.

## 12. Other model families

| Family | Reusable export material | Required adaptation |
|---|---|---|
| ACT | Images, state and future joint-action chunks | Dataset feature config, chunk horizon and a G2 serving adapter |
| Diffusion policy | Observation history and future action windows | History/window sampling, normalization and sampler latency handling |
| SmolVLA | Images, task language, state and actions | Its processor/tokenizer, model features and G2 serving adapter |
| Other VLAs | Synchronized robot observations and actions | Model-specific embodiment, camera, language and action conventions |
| LingBot-VLA | Potentially compatible robot episodes after adaptation | External implementation, environment, checkpoint and protocol adapter |
| LingBot-VA | Temporally ordered image clips and action sequences | Its video/action schema and representation-specific conversion |

The captured 5090 configuration's LingBot entry references an external `lingbot-vla-v2` repository and server module.
It is not supplied by this source release and is not made available by selecting a model name in the UI.
The separately inspected LeRobot LingBot-VA implementation used a fixed 30-channel multi-embodiment action layout.
Its released EEF-oriented examples are not interchangeable with this G2 16D joint-target dataset.
Mapping to EEF poses requires a declared frame, orientation convention, FK/IK policy and explicit gripper-channel mapping.
Checkpoint-specific camera order and temporal clip construction must also match training.

## 13. Register and validate a trained checkpoint

Keep each trained artifact in an immutable external directory with config, weights, processors and statistics.
Use [`configs/thor.toml`](../thor_project/configs/thor.toml) as the registration structure, replacing site-specific paths.
The captured Dahuan default is `newdata_training_rtc_004500`; this identifier is a historical registration, not a bundled model.
Add a new version under the correct model and effector variant rather than silently replacing an old checkpoint directory.

```toml
[models.pi05.effector_variants.dahuan.versions.hitl_sft_v1_010000]
display_name = "G2 Dahuan HITL SFT v1 - step 10000"
normalization_source = "checkpoint_embedded"
normalization_label = "G2 Dahuan HITL v1 checkpoint processors"
real_args = ["--checkpoint=/data/models/g2_pi05_hitl_v1/pretrained_model"]
```

This is a registration fragment; inspect inherited arguments, tokenizer path and RTC routing before use.
For a trained-RTC artifact, add trained routing and a compatible horizon only after obtaining the matching external implementation.
For an ordinary supervised checkpoint, do not inherit trained-only settings accidentally from an existing version.
The pi05 adapter rejects incompatible policy type, native dimensions, declared action order and processor semantics.
It returns native absolute actions plus canonical `(T, 24)` absolute actions through the shared WebSocket protocol.
Other architectures must implement equivalent metadata, observation handling, action conversion and session/reset behavior.

## 14. Evaluate before promoting a policy

| Validation layer | Useful evidence |
|---|---|
| Dataset integrity | UUID joins, hashes, synchronization error, rejected spans, frame counts and control-source coverage |
| Representation | Per-joint ranges, units, gripper endpoints/direction and absolute-relative round trips |
| Offline learning | Episode-held-out loss, task-specific metrics and comparison with the unchanged baseline |
| Chunk behavior | Horizon, stale-prefix behavior, interruption boundaries and action continuity |
| Runtime performance | Inference/transport percentiles, queue delay, image age and deadline misses |
| Closed-loop tasks | Success rate, intervention rate, completion time, aborts and recovery outcomes |
| Deployment traceability | Source, checkpoint, tokenizer, processor, configuration and asset identities |

Report the number of tasks/episodes and the held-out conditions with every success rate.
Lower offline loss alone does not establish better task completion or acceptable robot behavior.
Validate processors and dimensions offline, then model serving, then the read-only robot chain, before a controlled physical trial.
Source-only publication validation cannot substitute for target GPU, vendor SDK, collision-asset or physical acceptance.
Promote a checkpoint only with its exact processors, compatible dependencies and recorded evaluation evidence.
