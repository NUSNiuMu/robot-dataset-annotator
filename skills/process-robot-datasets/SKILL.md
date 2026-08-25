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
prefix with `context_start_frame`, and record all four manipulation boundaries in original
source-frame coordinates.

The adapter exposes synchronized hand poses and marks absent gripper-width fields invalid. A task
plugin may return multiple ordered candidates in `episodes`; confirm each one against all views.

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

The JSON stores `global_from_qr` and `qr_from_global`. It is calibration evidence only: do not
rewrite camera poses or discard the source context after producing it.

For a reviewed single-segment Insight MCAP, export three source camera streams and synchronized
poses to LeRobotDataset v3.0 with explicit paths:

    rda export-lerobot --source <recording-dir> \
      --review-manifest <review-dir>/manifest.json \
      --annotation-manifest <review-dir>/annotation_manifest.json \
      --decisions <review-dir>/decisions.json --task-spec <task.json> \
      --output <dataset-dir> --repo-id <namespace/name>

The adapter refuses an existing output and atomically promotes a temporary directory only after
LeRobot finalization. Its action is the next-frame dual-hand pose in the source tracking frame;
provenance must state that it is not robot-retargeted and has no gripper command.
The export also carries deterministic left/right subtask IDs, task/subtask progress, the original
head tracking pose, and the head RGB-camera global pose obtained from the recorded static transform.
Camera frames use direct streaming encoding by default to avoid temporary PNG disk traffic. If a
runtime encoder is incompatible with streaming, retry explicitly with
`--video-encoding-mode staged-png` and retain that mode in export provenance.

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
For this video-backed export, create a separate strict delivery copy containing only `data/`,
`meta/`, and `videos/`, keep the `rda/` evidence beside that copy, and do not modify the validated
working dataset. The `observation.images.*` feature names in `meta/info.json` still point through
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
