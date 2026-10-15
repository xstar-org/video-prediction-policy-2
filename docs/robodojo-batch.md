# RoboDojo batch inference

`RoboDojoVPP2Policy._infer_action_chunks(observations, instructions)` returns one
action chunk per input in the same order. Set `VPP2_INFERENCE_BATCH_SIZE=8` on the
policy server to cap real model batches at eight; the default cap is one. Final
partial batches are supported. Each environment keeps its own replan counter,
history and action/video generators. `reset_env` clears only that environment.

The scalar `_infer_action_chunk` API remains available. No weight offload, CPU
frame migration, prompt cache, reduced precision or inference-step reduction is
included. Batch visual-plan export is explicitly unsupported. Batch numerical
rounding may differ; simulator task-score validation is required in addition to
the CPU seed/order contract tests.
