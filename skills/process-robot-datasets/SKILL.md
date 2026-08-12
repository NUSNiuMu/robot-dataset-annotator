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

Create a schema-v2 template when needed:

    rda init-decisions --manifest <manifest.json> --output <decisions.json> \
      --task-id <task-id>

Validate before an expensive export:

    rda validate-decisions --manifest <manifest.json> \
      --decisions <decisions.json> --task-spec <task.json>

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
