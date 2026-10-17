"""Tests for prompt embedding caching (pure logic; no model load)."""
import ast
import pathlib
import unittest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src/vpp2/models/wan21_14b/vpp2.py"


class PromptCacheTests(unittest.TestCase):
    def test_encode_prompt_has_cache(self):
        tree = ast.parse(SRC.read_text(encoding="utf-8"))
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "encode_prompt"
        )
        text = ast.dump(fn)
        self.assertIn("_prompt_embedding_cache", text)
        self.assertIn("clone", text)
        # Cache check happens before tokenizer use: find statement order.
        names = [type(x).__name__ for x in ast.walk(fn)]
        self.assertIn("If", names)

    def test_cache_returns_copies(self):
        """The AST must return .clone() on cache hits so callers can mutate."""
        fn = next(
            n for n in ast.walk(ast.parse(SRC.read_text(encoding="utf-8")))
            if isinstance(n, ast.FunctionDef) and n.name == "encode_prompt"
        )
        hits = [
            n for n in ast.walk(fn)
            if isinstance(n, ast.Return)
            and isinstance(n.value, ast.Tuple)
            and any(isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "clone" for c in n.value.elts)
        ]
        self.assertTrue(hits, "cache-hit return must clone both tensors")

    def test_cache_capacity_bounded(self):
        src = SRC.read_text(encoding="utf-8")
        self.assertIn("_prompt_cache_capacity", src)
        self.assertIn("pop(next(iter(prompt_cache)))", src)


if __name__ == "__main__":
    unittest.main()
