# Optional RoboDojo inference weight offload

The default keeps model weights on their original device. For memory-limited
inference, the server can move the text encoder and a tail of video blocks to CPU:

```bash
export VPP2_TEXT_ENCODER_OFFLOAD=1
export VPP2_VIDEO_BLOCK_OFFLOAD=32
```

Set both values to `0` to keep the original behavior. The video count must be
between zero and the model's layer count, and requires evaluation mode.

The text encoder returns to GPU when encoding a new prompt and returns to CPU
afterward. Offloaded video blocks retain pinned CPU backing and are transferred
to GPU one block at a time during video prefill, then restored to CPU. Parameters
and buffers retain their dtype. Full history, video K/V, action horizon, Euler
steps and the actual requested tensor batch are preserved.

This trades host memory and PCIe transfers for GPU memory. It does not guarantee
that a particular batch fits, or that larger batches improve throughput. Check
capacity and steady-state latency on the target device. Do not change offload
configuration during an in-flight call, move the entire model to GPU afterward,
or concurrently invoke a shared mutable policy instance. Training with offloaded
video blocks is unsupported.

The controls apply to the RoboDojo policy loader; other release recipes retain
their existing default behavior. Checkpoints and task settings are unchanged.
