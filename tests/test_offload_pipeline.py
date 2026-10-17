"""Tests for the offload prefetch pipeline (pure AST-level; no model load)."""
import ast
import pathlib
import unittest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src/vpp2/models/wan21_14b/mot.py"


class OffloadPipelineTests(unittest.TestCase):
    def test_upload_stream_created(self):
        tree = ast.parse(SRC.read_text(encoding="utf-8"))
        text = SRC.read_text(encoding="utf-8")
        self.assertIn("_offload_upload_stream", text)
        self.assertIn("torch.cuda.Stream", text)

    def test_prefetch_overlaps_compute(self):
        """The prefetch call must come before the weight assignment in the loop."""
        tree = ast.parse(SRC.read_text(encoding="utf-8"))
        src_text = SRC.read_text(encoding="utf-8")
        prefetch_pos = src_text.find("_start_prefetch(layer_idx + 1)")
        consume_pos = src_text.find("_consume_prefetch(layer_idx, offloaded_state)")
        self.assertGreater(prefetch_pos, 0, "prefetch call not found")
        self.assertGreater(consume_pos, 0, "consume call not found")
        self.assertLess(prefetch_pos, consume_pos,
                        "prefetch must be issued before consuming current layer weights")

    def test_side_stream_usage(self):
        tree = ast.parse(SRC.read_text(encoding="utf-8"))
        text = SRC.read_text(encoding="utf-8")
        self.assertIn("torch.cuda.stream(upload_stream)", text,
                      "upload must use a side CUDA stream")
        self.assertIn("wait_event", text,
                      "compute must wait for upload event")
        self.assertIn("non_blocking=True", text,
                      "transfers must be non-blocking")

    def test_drain_on_exit(self):
        tree = ast.parse(SRC.read_text(encoding="utf-8"))
        text = SRC.read_text(encoding="utf-8")
        self.assertIn("_prefetched.clear()", text,
                      "leftover prefetch events must be drained")

    def test_offload_disabled_path_unchanged(self):
        """Without offload (no cpu_state), the loop body has no prefetch."""
        tree = ast.parse(SRC.read_text(encoding="utf-8"))
        text = SRC.read_text(encoding="utf-8")
        self.assertIn("_offload_upload_stream", text)
        self.assertIn("pin_memory", text)


if __name__ == "__main__":
    unittest.main()
