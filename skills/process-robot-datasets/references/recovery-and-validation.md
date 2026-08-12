# Recovery and validation

After interruption, inspect source, review, decisions, dataset, and validation artifacts in that
order. Do not infer progress from a terminal window or watcher report. Run `rda audit` and resume
the first incomplete transition.

Before restarting an encoder, confirm that no matching process is alive and that no complete
output already exists. Treat temporary directories as untrusted until their expected manifests,
frame counts, and videos pass validation.

Use a pinned validator environment for the target dataset library. Do not install an unconstrained
framework package into a device Python environment: generic wheels may replace vendor-specific
PyTorch, multimedia, or accelerator builds. Prefer a dedicated virtual environment or development
container whose lock file is stored with the adapter.

Validation should include:

1. exhaustive decision-schema coverage;
2. row, episode, timestamp, action, task-index, and validity invariants;
3. complete decoding of every video key and frame-count agreement;
4. official dataset API construction, representative indexing, and one DataLoader batch;
5. artifact hashes and the exact validator version.

If disk space is low, stop accepting new work before deleting evidence. Clean only artifacts the
user explicitly authorizes after both internal and official checks pass.
