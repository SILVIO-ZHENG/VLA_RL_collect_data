# From HITL Recordings to Actor-Critic Learning and VLA Training

This guide explains how G2 teleoperation and human-in-the-loop recordings can
become learning datasets, separating included collection code from proposed adapters.
All network sizes, batch sizes, rewards, and training schedules below are
illustrative engineering choices unless explicitly identified as source facts.

## 1. Implementation status

| Component | Status in this source release |
| --- | --- |
| Tablet recording discovery and unique episode binding | Implemented |
| Exact inference observation and native candidate capture | Implemented |
| Human ownership, executed-history, and rollback evidence | Implemented |
| Sidecar validation, sealing, transfer, and immutable archive | Implemented |
| Preference-pair interchange schema and validators | Implemented |
| Raw H5/video/sidecar to RL transition exporter | Requires implementation |
| Task reward annotation or trained success detector | Requires implementation |
| G2 actor-critic environment and online replay bridge | Requires implementation |
| G2-specific SAC, IQL, or FlowPRO optimizer | Not included |
| Automatic SAC checkpoint to VLA checkpoint conversion | Not provided |

This source-only publication excludes the original snapshot's upstream LeRobot
checkout, third-party libraries, weights, tokenizers, binary robot assets, and datasets.
An externally installed, compatible learning library is an optional dependency;
its training capabilities do not constitute an integrated G2 training pipeline.

The implemented archive command is documented in the
[offline entry point](../thor_project/src/nexus_deploy/robot/g2a/hitl/offline/__main__.py).
The project does not currently provide a single command that records a parcel,
exports its transitions, trains an actor, and deploys the resulting policy.

## 2. Preserve the source evidence first

One saved tablet recording corresponds to one vendor episode and one sealed
FlowPRO sidecar; several recordings can occur within one Nexus deployment.
Keep originals immutable and write derived datasets under a versioned destination.

The vendor episode contains:

~~~text
<record_name>/
  meta_info.json
  record/aligned_joints.h5
  camera/head_color/head_color.h265
  camera/head_color/head_color.txt
  camera/hand_left_color/hand_left_color.h265
  camera/hand_left_color/hand_left_color.txt
  camera/hand_right_color/hand_right_color.h265
  camera/hand_right_color/hand_right_color.txt
~~~

The sidecar contains a manifest, append-only events, exact inference inputs,
native candidate arrays, and intervention-related execution artifacts.
See the [writer and sealer](../thor_project/src/nexus_deploy/robot/g2a/hitl/common/sidecar.py)
and [contract validators](../thor_project/src/nexus_deploy/robot/g2a/hitl/common/contract.py).
Not every servo sample is necessarily a separate durable event:
intervention windows can be frozen from in-memory execution history.
A complete dense episode exporter must therefore inspect vendor H5 coverage
as well as sidecar evidence before claiming full command provenance.

The archive verifies sealed manifests and hashes, then stores content-addressed
objects, a per-bundle receipt, and an ACK.
Record the source receipt, source manifest hash, exporter revision,
annotation revision, normalization statistics, and split manifest in every
derived dataset release.

## 3. Reconstruct one shared timeline

The join clock is vendor_timestamp_ns, in nanoseconds.
The recorded monotonic clock is local to a process and must not be used as a
universal timestamp across the robot client, server, and archive machine.
The manifest can record a wall-to-vendor offset; apply the declared clock
mapping once, then retain the original timestamps alongside converted values.

The existing validator checks contiguous H5 frame keys starting at zero and
strictly increasing main_timestamp values.
It also checks increasing camera timestamps and maximum camera alignment skew.
Captured inference inputs use the exact three camera keys:
cam_head, cam_left_wrist, and cam_right_wrist.
Each captured JPEG has an identity, hash, timestamp, width, and height.
The inference contract permits at most 40 ms of camera-to-anchor skew.

A proposed exporter should perform these steps:

1. Verify the archive receipt and sealed bundle before reading tensors.
2. Read the declared checkpoint, effector profile, cadence, and clock contract.
3. Decode each camera stream against its timestamp text file.
4. Read measured robot state and available command/ownership fields from H5.
5. Join events by session, observation, request, candidate, generation, and action IDs.
6. Reconstruct dispatch lineage and reject ambiguous or conflicting joins.
7. Select a documented learning cadence and construct state/action intervals.
8. Attach reward, termination, provenance, and quality masks.
9. Split by original recording or collection session before fitting statistics.

Use a bounded timestamp association rule, with measured skew stored per sample.
Do not pair a future image with an earlier state without recording that choice.
Do not interpolate an image, language label, ownership change, or contact event.
If camera decoding loses a frame, preserve the gap and invalidate affected
transitions rather than silently shifting every later image by one index.

## 4. Keep state, action, and command spaces distinct

The exact inference input is canonical hardware float32[24].
Native policy candidates are finite float32[T,D], with D equal to 16 or 24.
Their declared space is absolute_unnormalized.
The general runtime supports both widths, while formal collection manifest v1
requires a 16D arm/effector policy and excludes head/waist control.

For the 16D model convention, dimensions 0:7 are the seven left-arm joints,
dimension 7 is the left gripper, dimensions 8:15 are the right-arm joints,
and dimension 15 is the right gripper.
Use the live effector profile and model adapter for unit conversion and gripper
direction; the model gripper coordinate and the hardware coordinate can differ.
Do not assume that every 24D hardware vector is already a 24D model action.
For Dahuan, hardware opening is 0 to 0.025 m, while the model gripper command
has the opposite direction: hardware_target = 0.025 - model_action.
See [effector profiles](../thor_project/src/nexus_deploy/robot/g2a/effector_profiles.py)
and the selected model processor before building a feature map.

For each interval, retain these separate quantities when available:

| Quantity | Meaning |
| --- | --- |
| measured_state_24d | Sensor observation of the physical robot |
| policy_native_actions | Generated model-space proposal |
| safety_targets_24d | Target after the safety/execution path |
| gdk_commands_24d | Command submitted through the hardware interface |
| command_accepted | Evidence that a submitted command was accepted |
| safety_modified | Whether the safety path changed the proposal |
| measured_model_actions | Measured trajectory represented in model coordinates |

The action in a Bellman transition must describe the action of the chosen
environment interface.
For a target-position actor, that usually means the accepted target mapped into
the actor's action space; next_state is the later measured outcome.
Measured joint position is useful supervision, but is not automatically the
issued action: actuator lag and filtering make those quantities different.
When a human target command cannot be reconstructed, a measured human path can
support an explicitly labeled imitation dataset; its use as RL action needs a
defined environment interpretation and uncertainty treatment.

For normalization, fit arm and gripper transforms on the training split only.
Save the per-feature units, bounds, mean/std or quantile transform, clipping,
and inverse transform with the checkpoint.
Use the same transforms for offline data, online collection, and deployment.

## 5. Choose the decision cadence before building transitions

A 100 Hz servo loop, a model execution FPS, and a policy inference rate describe
different clocks.
A candidate chunk can contain many future actions while the executor consumes
only an initial portion before replacing it with a newer chunk.
Never count unexecuted candidate suffixes as environment experience.

For a per-step learner, choose decision times t[k] and define:

~~~text
observation[k] = measured state + aligned images + task context at t[k]
action[k]      = accepted interface action governing the interval
next_obs[k]    = measured observation after that interval
reward[k]      = verified reward accumulated over that interval
discount[k]    = discount for the actual interval duration
~~~

An interval that contains several materially different targets needs either
finer segmentation or an explicitly defined action aggregation rule.
Nearest-time resampling alone does not resolve that ambiguity.
Store delta_t and, when using a fixed physical discount time base, compute
discount[k] = gamma_ref ** (delta_t / reference_delta_t).

For a chunk-action learner, action has shape [H,D] and needs an executed mask.
Define the transition duration, accumulated reward, and next observation at the
end of the executed chunk prefix.
Use a semi-Markov target with the corresponding multi-step discount.
Do not train a standard one-step target against a whole chunk without defining
these semantics.
Padding masks, execution masks, and quality masks serve different purposes.

## 6. Ownership, safety, rollback, and masks

Retain control_owner values policy, human, hold, and rollback.
Button intent alone does not prove human control: the runtime checks vendor
authority and fresh evidence.
See [supervisor](../thor_project/src/nexus_deploy/robot/g2a/hitl/g2_runtime/supervisor.py)
and [execution history](../thor_project/src/nexus_deploy/robot/g2a/hitl/g2_runtime/executed.py).

For an initial task learner, use contiguous valid policy/human task segments.
Store HOLD and rollback in the archive, then handle them under explicit rules:

- Policy samples require accepted commands and attributable dispatch lineage.
- Human samples require confirmed ownership and a declared action reconstruction.
- Operator HOLD may be useful context but should not dominate imitation batches.
- Queue-underrun HOLD signals an execution fault; keep a separate quality flag.
- Rollback is recovery behavior; exclude it from forward-task demonstrations.
- Scene restoration, reset, and manual object replacement break task continuity.
- Stale observations, missing command evidence, and timing gaps invalidate joins.

A safety-modified proposal cannot be paired with the executed next state while
pretending the original proposal was executed.
Either learn the composite actor-plus-safety environment and preserve its state,
or train on accepted transformed actions with consistent interface semantics.
Keep both proposal and executed action for diagnostics.
If an action is rejected before movement, record the rejection; do not label it
as a successful task action.

Useful proposed masks are transition_valid, observation_valid, command_valid,
reward_valid, actor_bc_valid, and executed_action_mask.
Define masks in exporter configuration and dataset metadata.
An intervention can express safety, exploration assistance, or correction;
it is not inherently a negative task reward.

## 7. Define reward and episode endings independently

EpisodeOutcome can contain success, failure, aborted, unknown, or pending_recorder_commit.
The runtime can promote pending_recorder_commit to success on finalized tablet metadata,
recording successful commitment with outcome_authority=external_tablet_recorder.
It does not independently establish that the parcel manipulation task succeeded.

Never assign reward +1 solely from that recorder-success field.
A proposed reward layer must use task annotations or a validated success detector,
with a reward definition version and evidence attached to each label.
For example, a sorting task could define success as the specified parcel inside
the intended destination with the gripper released and the outcome verified.
That example is a proposed task specification, not a detector included here.

An illustrative sparse reward is +1 for verified task completion and 0 otherwise.
Optional failure/contact/time penalties require separate verified labels,
recorded units, and calibration; changing reward scale changes learning behavior.
Missing labels remain unknown and receive reward_valid=false.
Do not silently replace unknown outcomes with failure.
Do not propagate a terminal +1 to every earlier frame without defining a shaped
reward; credit assignment already occurs through the value function.

Use terminated=true for an actual terminal state of the task.
Use truncated=true for an external collection/time budget boundary when the task
could have continued.
For a valid time-limit truncation, bootstrapping may continue from a valid final
observation; use bootstrap_mask = 1 - terminated.
For corrupted data, emergency abort, reset discontinuity, or a missing next
observation, cut the segment and set an explicit bootstrap_valid=false.
Keep the final observation separate from a subsequent reset observation.
Episode boundaries prevent joining two different tablet recordings.

## 8. Proposed transition schema

This design target has no included exporter. Arrays may live in Parquet/Arrow,
NPZ, or another learning format, with video references stored separately.

~~~yaml
schema_version: g2.learning_transition/proposed-v1
identity:
  bundle_id: <sealed-session-uuid>
  record_name: <original-vendor-recording>
  segment_id: <contiguous-forward-task-segment>
  step_index: 42
  source_manifest_sha256: <sha256>
  exporter_revision: <git-commit>
observation:
  timestamp_ns: <vendor-clock>
  state: <float32[24], canonical hardware coordinates>
  images: <three timestamped frame references>
  task: <language instruction>
action:
  values: <float32[16], declared target-position interface>
  native_proposal: <optional float32[16]>
  space: <versioned coordinate/unit convention>
  normalization_id: <training-statistics-hash>
next_observation: <same structure as observation>
reward: 0.0
reward_definition_id: <task-label-specification-version>
reward_evidence: <annotation-or-detector-reference>
terminated: false
truncated: false
bootstrap_valid: true
delta_t_seconds: 0.0333333333
discount: 0.99
control_owner: policy
quality:
  transition_valid: true
  reward_valid: true
  command_accepted: true
  safety_modified: false
  camera_skew_ms: <per-camera-values>
  queue_underrun: false
lineage:
  observation_id: <uuid>
  candidate_id: <uuid>
  generation_id: <uuid>
  action_id: <uuid>
  chunk_id: <source-chunk-id>
  action_index: <executed-index-in-candidate>
~~~

For human segments, candidate lineage may be null; retain intervention identity,
ownership evidence, and the source of the reconstructed action instead.
Hash the feature specification so a loader can reject incompatible datasets.

## 9. Example tensor batch

This proposed visual actor-critic batch uses illustrative batch size 128 and
image resolution 224; these are not runtime requirements.

~~~python
batch = {
    "state": {
        "observation.state": Tensor[128, 24],
        "observation.images.cam_head": Tensor[128, 3, 224, 224],
        "observation.images.cam_left_wrist": Tensor[128, 3, 224, 224],
        "observation.images.cam_right_wrist": Tensor[128, 3, 224, 224],
    },
    "action": Tensor[128, 16],
    "reward": Tensor[128, 1],
    "next_state": {...},  # Same observation keys and shapes.
    "discount": Tensor[128, 1],
    "terminated": BoolTensor[128, 1],
    "truncated": BoolTensor[128, 1],
    "bootstrap_valid": BoolTensor[128, 1],
    "transition_valid": BoolTensor[128, 1],
    "is_human": BoolTensor[128, 1],
}
~~~

This shape notation is not executable Python. Images become channel-first tensors
after a declared decode, resize, color, and augmentation pipeline.
Language can be a fixed task ID or encoded tokens for a language-conditioned
actor; a state-only ablation may omit images but cannot infer unseen objects.
For H-step chunks, action becomes [B,H,16] with a matching execution mask.

## 10. Illustrative SAC architecture and objectives

A possible G2 experiment uses three visual encoders, a 24D state MLP, and task
conditioning concatenated into a shared observation representation.
An actor head outputs 16 means and 16 log standard deviations.
A bounded distribution is mapped back into the declared physical action range.
Two independent critic heads receive the observation representation plus action
and return scalar Q estimates; target critics provide the backup.
Encoder sharing, freezing, and update schedules must be tested on the dataset.

SAC learns an entropy-regularized stochastic policy using off-policy experience.
For a modern twin-critic formulation, a schematic backup and actor objective are:

~~~text
a_next ~ pi(. | next_obs)
y = r + discount * bootstrap_mask *
        (min(Q1_target, Q2_target)(next_obs, a_next) - alpha * log_pi(a_next))
L_Q = mean((Q1(obs,a)-y)^2 + (Q2(obs,a)-y)^2)
L_actor = mean(alpha * log_pi(a_pi) - min(Q1,Q2)(obs,a_pi))
~~~

Losses must also respect transition/reward/bootstrap validity masks.
Temperature tuning and target averaging are algorithm settings, not robot
safety mechanisms. See [SAC](https://arxiv.org/abs/1801.01290) and its
[algorithms and applications paper](https://arxiv.org/abs/1812.05905).
Pure offline SAC can query unsupported actions; choose and validate an offline
learning method rather than assuming a replay buffer alone resolves that issue.

## 11. Offline IQL and behavior cloning

IQL is another proposed route; no G2 IQL implementation is included.
It fits an expectile state-value function from dataset actions, backs that value
into Q, and extracts a policy with advantage-weighted behavior cloning.
A schematic form is:

~~~text
L_V = expectile_loss(Q_target(obs,a) - V(obs))
L_Q = (Q(obs,a) - [r + discount * bootstrap_mask * V(next_obs)])^2
weight = clip(exp((Q_target(obs,a)-V(obs))/temperature), max_weight)
L_policy = -weight * log_pi(dataset_action | obs)
~~~

The weight scale, clipping, and expectile are experiment parameters.
For deterministic actors, use a compatible weighted regression objective.
IQL reduces reliance on evaluating unseen actions during offline improvement;
it does not correct mislabeled rewards, timing errors, or missing task coverage.
See [Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169).

Before value learning, establish a behavior-cloning baseline from verified
human demonstrations and successful forward task segments.
Use active-dimension and temporal padding masks in the imitation loss.
Keep failure trajectories for critic learning when their rewards and commands
are valid; they need not be positive imitation targets.
Split all derivatives of one source recording into the same partition.
Evaluate by held-out task success and intervention rate, not training loss alone.

## 12. Offline-to-online lifecycle and optional libraries

Publish a validated dataset and reproducible BC baseline, then compare offline
value learning against that baseline.
Check held-out returns, action bounds, calibration, and rollout behavior.
Before online learning, implement the robot environment, reward source, replay
bridge, checkpoint handoff, and the existing deployment admission path.
Keep updates outside the servo loop and version the actor for each collection segment.
Mix demonstrations and online transitions deliberately and log the mixture.
Evaluate fixed snapshots before promoting them to subsequent collection runs.

An externally installed LeRobot version may provide generic actor/learner tools:

~~~bash
# Generic external-library examples; require a compatible G2 adapter/config.
python -m lerobot.rl.learner --config_path /path/to/g2_rl_config.json
python -m lerobot.rl.actor --config_path /path/to/g2_rl_config.json
~~~

These commands are not turnkey commands for this repository.
Confirm the selected library version's API, feature names, action transforms,
reward keys, termination handling, and dependencies before integration.
Some LeRobot replay routes use action, next.reward, next.done, episode_index,
and configured observation keys; support for truncation varies by version.
Do not collapse the proposed termination schema merely to fit an older loader.

## 13. Connecting learned experience to VLA training

Actor parameters and critic return estimates do not directly convert into a
vision-language-action backbone.
The transferable artifact is usually verified experience with synchronized
images, measured state, instruction, and correctly represented action targets.

A proposed workflow is to collect and evaluate the improved actor, curate its
successful forward trajectories and human corrections, export a VLA dataset,
and fine-tune the chosen VLA with its own processor and training objective.
Keep source identity, reward evidence, and actor version on every trajectory.
Construct future action chunks within one contiguous segment and mask padding.
Use the VLA's required state/action convention and training statistics.
A flow-matching model needs its native supervised objective; a tokenized action
model needs its tokenizer and sequence objective.
Critic-ranked or advantage-weighted examples are an optional new experiment,
requiring calibration and held-out checks; high predicted Q is not ground truth.

FlowPRO takes a different learning route: its published method uses paired
preferred/rejected trajectories and a proximal preference objective for
flow-matching VLAs, rather than requiring task rewards and an actor-critic loop.
See the [FlowPRO paper](https://arxiv.org/abs/2606.05468).
This repository includes collection evidence and preference contracts, not the
paper's complete training optimizer or a claim of reproduced paper results.

Strict preference examples require executed negative evidence, a common anchor,
verified scene context, synchronized sensors, compatible action conventions,
and valid temporal/dimension masks.
The current execution contract rejects queue-underrun/gapped histories for
strict preference eligibility even when measured holds remain useful for rewind.
Synthetic interpolation needs explicit provenance and contact-aware validation;
it must not be mislabeled as a measured demonstration.
See the [preference schema](../thor_project/src/nexus_deploy/robot/g2a/hitl/schemas/preference-pair-v1.schema.json).

Remaining work includes a raw exporter, reward/termination annotation layer,
learning-library adapter, G2 environment/replay bridge, VLA dataset adapter,
and reproducible evaluation.
Collection contracts supply evidence; these components still require implementation.
