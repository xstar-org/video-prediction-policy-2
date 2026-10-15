"""Exercise production batch dispatch without loading model weights."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch

source = Path(__file__).resolve().parents[1] / 'src/vpp2/policy.py'
tree = ast.parse(source.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RoboDojoVPP2Policy')
cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_infer_action_chunks']
ns = {'torch': torch}
exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), ns)


class DispatchTests(unittest.TestCase):
    def test_microbatch_tail_preserves_prompt_seed_and_result_order(self):
        policy = ns['RoboDojoVPP2Policy']()
        policy.inference_batch_size = 2
        policy.visual_planning_output_root = None
        policy.output_action_dim = 2
        policy._prepare_inference = lambda o, i: dict(
            prompt=i, input_image=torch.zeros(3, 16, 16),
            proprio=torch.tensor([[o, o]]), seed=o, video_seed=o + 100,
            condition_video=torch.zeros(1, 3, 9, 16, 16))
        calls = []
        def infer(**kwargs):
            calls.append(kwargs)
            return {'action': torch.tensor(kwargs['seed'])[:, None, None].expand(-1, 3, 2)}
        policy.model = SimpleNamespace(infer_action=infer)
        policy._denormalize_action = lambda x: x.numpy()
        outputs = policy._infer_action_chunks([7, 2, 9], ['a', 'b', 'c'])
        self.assertEqual([x['input_image'].shape[0] for x in calls], [2, 1])
        self.assertEqual([x['prompt'] for x in calls], [['a', 'b'], ['c']])
        self.assertEqual([x['video_seed'] for x in calls], [[107, 102], [109]])
        self.assertEqual([x[0, 0] for x in outputs], [7, 2, 9])
        self.assertEqual(policy._infer_action_chunks([], []), [])
        with self.assertRaises(ValueError):
            policy._infer_action_chunks([1], [])


if __name__ == '__main__':
    unittest.main()
