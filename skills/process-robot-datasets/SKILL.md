---
name: process-robot-datasets
description: Audit, resume, annotate, segment, visually review, export, and validate task-pluggable robot datasets. Use when Codex is asked to process a batch of recordings, continue an interrupted annotation job, review every source segment, add a new manipulation task, produce LeRobot-compatible outputs, or verify batch completeness without assuming machine-specific paths.
---

# Process Robot Datasets

Run annotation as an artifact-driven state machine. Treat automatic annotations as suggestions,
preserve source data, and make every environment path explicit.

## Resolve the environment first

Read the active repository instructions. If the user has not already supplied all paths, inspect
only enough local context to find plausible candidates, then ask the user to confirm:

1. framework checkout containing `pyproject.toml` and the `rda` command;
2. source-recording root;
3. review-artifact root;
4. dataset-output root;
5. task-spec JSON;
6. runtime or container command when source adapters need ROS or other system dependencies.

Do not assume the current directory, a home directory layout, camera names, container names, or
output folders. Convert confirmed paths to absolute paths and create a session with:

    rda configure --output <session.json>

Missing configure arguments intentionally prompt the user. Reuse the session file during one
campaign; do not bake its paths back into this Skill.

## Audit before changing anything

Run:

    rda audit --config <session.json>

Trust files and validation reports over watcher PIDs or stale campaign summaries. For each item,
perform only the next missing transition:

| Evidence-derived stage | Next action |
|---|---|
| `DISCOVERED` | Prepare review evidence |
| `REVIEW_READY` | Review or repair decisions |
| `REVIEW_DECIDED` | Export the dataset |
| `EXPORTED` | Run internal and official checks |
| `INTERNAL_VALID` | Finish remaining external checks |
| `COMPLETE` | Skip without overwriting |

Use atomic temporary outputs followed by rename. Never overwrite a complete dataset.

Preview the configured next transition:

    rda resume --config <session.json>

Inspect the rendered argv and paths. Add `--execute` only after they match the confirmed scope.
Commands are argv arrays stored in the session and do not use a host shell. Leave human-review
actions unconfigured so the runner stops and reports the required manual work.

## Review exhaustively

Read [references/review-standard.md](references/review-standard.md) before writing decisions.

- Inspect every source segment and every required camera.
- Use low-rate overviews for complete coverage and full-rate views near uncertain boundaries.
- Treat task-plugin candidates and rejection reasons as hints only.
- Record exactly one PASS or FAIL conclusion per source segment.
- Allow multiple non-overlapping episodes inside one PASS source segment.
- Require monotonically increasing task boundaries and the configured minimum frames per action.
- When QR calibration context must survive segmentation, set `context_start_frame` before
  `episode_start_frame`; never move the manipulation boundary backward to absorb calibration.

For a task that defines a suggester, an Insight synchronized review manifest can be used without
first producing parquet:

    rda suggest --task-spec <task.json> --observations <review-manifest.json> \
      --format insight-review

`cup-pick-place` intentionally has no automatic splitter. Review it manually, keep the QR calibration
prefix with `context_start_frame`, and record the overall atomic boundaries plus independent
left/right hand subtask boundaries in original source-frame coordinates. A waiting right hand must
remain `right_hand_wait`; do not relabel it as acquisition or transport merely because the left hand
is moving.

`screw-nut-sorting` also intentionally requires manual boundaries. Treat one complete mixed batch as
an episode: nuts go in the fixed left box and screws in the fixed right box. Do not split overlapping
left/right single-item cycles into separate episodes. End the sorting phase at the final release, keep
the subsequent completion and retreat phase, exclude human reset intervals, and independently mark
when each gripper stops sorting and waits or retreats. Human intervention, a wrong box, an item outside
both boxes, or an item left in the central workspace prevents a PASS episode.

After its decisions are valid, `screw-nut-sorting` requires a native-VIO versus Insight-Global audit
for both hands and for each episode:

    rda audit-insight-episode-poses --source <recording-dir> \
      --review-manifest <review-dir>/manifest.json \
      --decisions <review-dir>/decisions.json --task-spec <task.json> \
      --output <review-dir>/episode_pose_quality_audit.json

Accept a native jump only when Insight Global cancels it at the same frame. A shared Global jump is
uncorrected; a later alignment update is only partially corrected; missing or discontinuous evidence
remains `NEEDS_REVIEW`. Evaluate every later episode from its own boundaries, so an earlier drift does
not invalidate it. The adapter must reproduce the review manifest within 1 mm, selecting its actual
linear-interpolation or legacy nearest-neighbor pose sampling and applying the same sampler to paired
native VIO. Keep rejected source evidence, revise final decisions if necessary, and rerun the audit.
Desktop-world QR, gripper-state markers, and rear multi-face camera calibration are separate
evidence and must not replace this temporal comparison.

The review adapter exposes synchronized hand poses. Gripper width remains absent until export unless
an explicit wrist-marker calibration is supplied. A task plugin may return multiple ordered
candidates in `episodes`; confirm each one against all views.

Create a schema-v2 template when needed:

    rda init-decisions --manifest <manifest.json> --output <decisions.json> \
      --task-id <task-id>

Validate before an expensive export:

    rda validate-decisions --manifest <manifest.json> \
      --decisions <decisions.json> --task-spec <task.json>

For a task with preserved QR context, measure the printed black-square edge and estimate the QR
pose in the global tracking frame before export:

    rda calibrate-qr --source <recording-dir> \
      --review-manifest <review-dir>/manifest.json --marker-size-m <meters> \
      --frame-start <context-start> --frame-end-exclusive <episode-start> \
      --output <review-dir>/qr_transform.json

When the printed square is ArUco rather than a standard QR code, add
`--marker-type aruco --aruco-dictionary <dictionary> --aruco-marker-id <id>`.
For the standard UMI `DICT_4X4_50` marker ID 4, use the configured 0.06 m black-square edge; do not
substitute the 0.16 m workspace-marker size used by other UMI IDs.

The JSON stores `global_from_qr` and `qr_from_global`. It is calibration evidence only: do not
rewrite camera poses or discard the source context after producing it.
Static transforms may be split across multiple latched `/tf_static` messages. The adapter merges
messages until it finds the complete head-IMU-to-RGB path; do not treat the first static message as
the complete calibration set.
If that path is genuinely absent but another recording from the same, unchanged head device has it,
pass `--head-static-calibration-source <recording-dir>` to both QR calibration and export. This only
borrows `tf_static`; current images, intrinsics, and global poses remain authoritative. Confirm the
device was not remounted. The QR JSON and export manifest must identify the reference recording and
set `static_calibration_borrowed`; never hardcode or silently borrow the transform.

Before QR calibration or export, audit global pose continuity:

    rda correct-pose-drift --review-manifest <review-dir>/manifest.json \
      --output-manifest <review-dir>/manifest_pose_corrected.json \
      --audit <review-dir>/pose_drift_audit.json \
      --decisions <review-dir>/decisions.json

Use the corrected manifest only when the audit is `PASS`. The command may interpolate paired short
spikes, stitch a high-confidence persistent coordinate jump and its later stable residual jumps,
progressively stitch a short, directional coordinate-drift transition whose cumulative displacement is physically implausible,
or stitch a 2--4-step settling jump followed by stable tracking. A confirmed large coordinate jump
may also establish the instability chain for independent medium stable jumps earlier in that pose stream.
Subsequent relative motion remains intact. It writes a new manifest with raw arrays and correction
masks; ambiguous jumps remain unchanged and force `NEEDS_REVIEW`.
When final decisions intentionally exclude a later ambiguous interval, pass `--decisions`. The PASS
gate then applies only to the exact selected episode ranges while preserving full-stream status and
all out-of-scope unresolved frames. The corrected manifest and audit must bind the decisions hash
and selected ranges. Reproduce the source bag with the preserved raw pose arrays, then evaluate
native-VIO correction against the corrected pose arrays during the per-episode audit.

For batch trajectory inspection, generate a self-contained interactive 3D dashboard and machine-
readable QC report from the reviewed recording root:

    rda visualize-trajectories --input-root <recordings-root> \
      --output <trajectory-qc-dir> --write-png

Select the corrected manifest referenced by a final PASS audit; do not select files by a filename
guess when earlier `NEEDS_REVIEW` or pre-settling artifacts coexist. When `qr_transform.json` is
available, visualize both raw and corrected trajectories in the QR frame so recordings share a
meaningful origin. Render only context and training frames selected by the final decisions; never
let a discarded post-drift suffix reappear in 3D tracks or per-take correction plots. Rate temporal
pose continuity and validity inside those selected ranges. Keep hand-to-head distance as navigation
evidence only, never as a rating gate. QR quality and robust-centroid checks may independently flag
calibration or cross-take spatial consistency, but they are not temporal VIO-drift evidence. Use
`PASS_AFTER_CORRECTION` only when the selected ranges contain corrected frames. Inspect every
`REVIEW_REQUIRED` plot before excluding or reprocessing data; the visualizer must never rewrite
poses or decisions.

For a reviewed single-segment Insight ROS 2 bag in MCAP or SQLite3 storage, export three source
camera streams and synchronized poses to LeRobotDataset v3.0 with explicit paths. The adapter lets
rosbag2 select the storage plugin from the bag metadata:

    rda export-lerobot --source <recording-dir> \
      --review-manifest <review-dir>/manifest.json \
      --annotation-manifest <review-dir>/annotation_manifest.json \
      --decisions <review-dir>/decisions.json --task-spec <task.json> \
      --gripper-calibration <gripper-calibration.json> \
      --output <dataset-dir> --repo-id <namespace/name>

For any task with `episode_pose_quality`, add
`--episode-pose-audit <review-dir>/episode_pose_quality_audit.json`. Export must stop unless the audit
is `PASS`, every selected episode is usable, and its manifest, decisions, and task-spec hashes match.

To place accepted episodes from multiple recordings in one dataset, create a schema-v1 recordings
manifest whose ordered `recordings` array supplies `source`, `review_manifest`, `annotation_manifest`,
`decisions`, and the required `episode_pose_audit` for every recording. Then run:

    rda export-lerobot-batch --recordings-manifest <recordings.json> \
      --task-spec <task.json> --gripper-calibration <gripper-calibration.json> \
      --maximum-gripper-interpolation-gap-frames 3 \
      --output <dataset-dir> --repo-id <namespace/name>

This is one atomic writer operation, not a filesystem merge. Require identical FPS and video geometry,
renumber episodes globally, and carry both `annotation.source_recording_index` and the original source
frame on every row. Bind each recording index to its bag name, episode range, input hashes, pose audit,
synchronization, and head calibration in `rda/export_manifest.json`.

The adapter refuses an existing output and atomically promotes a temporary directory only after
LeRobot finalization. Without a gripper calibration, its action remains the next-frame 18D
dual-hand pose and provenance states that no gripper command exists. With an explicit calibration,
detect each hand camera's two configured ArUco jaw markers, map their center distance to physical
jaw width, insert width after each 9D hand pose, and export a next-frame 20D action. Require a unique
detection of both markers for direct measurement. A calibration may explicitly define a symmetric
marker midpoint and reference image geometry; when exactly one jaw marker is unique, infer the total
distance as twice its distance from that midpoint. Never enable this fallback without calibration,
and invalidate ambiguous, fully missing, or geometry-mismatched frames. Record direct and inferred
coverage, the visible marker used for inference, paired-midpoint error, clipping, calibration
parameters, and the calibration hash. Generate
`meta/manifest.json` and `meta/modality.json` and validate their 20D indices (left width 9, right
width 19) before delivery.
For the batch exporter only, a configured short-gap resolver may linearly interpolate at most three
invalid frames when valid measurements bound the gap inside the same episode. Never interpolate across
episodes, at an unbounded episode edge, or across a longer dropout. Store per-frame state/action source
codes for invalid, paired-marker direct, single-marker symmetric inference, and temporal interpolation;
validation must prove that those codes agree with validity and next-frame action semantics.
The export also carries deterministic left/right subtask IDs, task/subtask progress, the original
head tracking pose, and the head RGB-camera global pose obtained from the recorded static transform.
Camera frames use direct streaming encoding by default to avoid temporary PNG disk traffic. If a
runtime encoder falls briefly behind source decoding, keep the bounded encoder queue large enough
to avoid frame loss; the default is 256 frames per camera and can be adjusted with
`--encoder-queue-maxsize`. Each camera encoder defaults to two threads so three-camera or batched
exports do not oversubscribe a modest CPU; adjust this with `--encoder-threads` and retain its value
in export provenance. If a runtime encoder is incompatible with streaming, retry explicitly
with `--video-encoding-mode staged-png` and retain the mode and queue size in export provenance.

Validate the completed local dataset in the same pinned environment:

    rda validate-lerobot --dataset <dataset-dir>

For Python 3.10, build that isolated environment from
`configs/lerobot-validator-py310.txt`; do not install it into the device Python environment.
This command must completely decode every video file before it writes internal PASS evidence, then
construct the official LeRobot loader, index representative rows, and read one DataLoader batch
before it writes official PASS evidence.

After validation, distinguish the working dataset from a strict LeRobot delivery package. The
working dataset keeps `rda/` because the exporter and validators store provenance and PASS evidence
there; `rda/` is an RDA extension, not part of the LeRobot v3 schema. The official writer may also
leave empty `images/<video-key>/` staging directories after encoding `dtype: video` features. Check
that `images/` contains no files before treating it as disposable; never remove it when files remain.
For this video-backed export, create a separate strict delivery copy containing `data/`, `meta/`,
and `videos/`; an explicitly required immutable calibration sidecar such as `qr_transform.json` may
also be copied at the delivery root. Keep the `rda/` evidence beside that copy and do not modify the
validated working dataset. Confirm the official loader still opens the package with any approved
sidecar. The `observation.images.*` feature names in `meta/info.json` still point through
`video_path` to MP4 files under `videos/`; they do not require the empty root `images/` directory.

## Resume and validate

Read [references/recovery-and-validation.md](references/recovery-and-validation.md) after a
reboot, interrupted encoder, missing validator, or inconsistent report.

Require source-review coverage, internal dataset verification, task-specific semantic checks,
and the official loader/decoder check configured for the target format. Record code revision,
task-spec hash, decision hash, and validator version in final provenance.

Run the audit again. Completion means every in-scope item is `COMPLETE` and no relevant worker is
still running. Report item, episode and frame totals, failures, remaining disk space, and the
confirmed output paths.

Do not delete recordings, decisions, review evidence, or datasets without explicit authorization.

## Add another task

Read [references/task-extension.md](references/task-extension.md). Keep source-format code in an
adapter and task semantics in a plugin. Add labeled regression examples and report both candidate
precision and recall; never optimize an automatic gate solely by reducing the number of candidates.
