"""Exercise production ownership/error paths without checkpoint downloads."""
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Callable, Union, Sequence
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1] / 'src/vpp2/models/wan21_14b'

def extract(file, name, methods):
    node = next(n for n in ast.parse((ROOT/file).read_text()).body if isinstance(n, ast.ClassDef) and n.name == name)
    node.bases = []; node.keywords = []
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    ns = dict(torch=torch, Optional=Optional, Callable=Callable, Union=Union, Sequence=Sequence)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT/file), 'exec'), ns)
    return ns[name]

MoT = extract('mot.py', 'MoT', {'set_video_block_offload', 'prefill_video_cache'})
VPP = extract('vpp2.py', 'VPP2', {'encode_prompt'})

class OffloadTests(unittest.TestCase):
    def model(self):
        m=MoT(); m.training=False; m.num_layers=2
        m.mixtures={'action':torch.nn.Linear(2,2), 'video':SimpleNamespace(blocks=torch.nn.ModuleList([torch.nn.Linear(2,2),torch.nn.Linear(2,2)]),num_heads=1)}
        m.use_fixed_video_layers=None; m.use_video_tokens=False
        return m

    def test_disabled_and_tail_selection(self):
        m=self.model(); before=[p.clone() for b in m.mixtures['video'].blocks for p in b.parameters()]
        m.set_video_block_offload(1)
        self.assertEqual(set(m._video_block_cpu_state),{1})
        m.set_video_block_offload(0)
        self.assertEqual(m._video_block_cpu_state,{})
        after=[p for b in m.mixtures['video'].blocks for p in b.parameters()]
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,after)))
        for n in (-1,3):
            with self.assertRaises(ValueError): m.set_video_block_offload(n)
        m.training=True
        with self.assertRaises(ValueError): m.set_video_block_offload(1)

    def test_failure_restores_backing(self):
        m=self.model(); m.set_video_block_offload(2)
        def fail(**kw): raise RuntimeError('injected compute failure')
        m._build_expert_attention_io=fail
        with self.assertRaisesRegex(RuntimeError,'injected'):
            m.prefill_video_cache(video_tokens=torch.ones(1,2,2),video_freqs=torch.ones(2,1,2),video_t_mod=torch.ones(1),video_context_payload=None,video_attention_mask=torch.ones(2,2))
        for state in m._video_block_cpu_state.values():
            for tensor, backing in state:
                self.assertEqual(tensor.data_ptr(),backing.data_ptr())

    def test_text_output_and_exception_restoration(self):
        m=VPP(); m.device='cpu'; moves=[]
        class Encoder:
            def to(self,device): moves.append(device); return self
            def __call__(self,ids,mask): return ids.float()[...,None].expand(-1,-1,2).clone()
        m.text_encoder=Encoder(); m.tokenizer=lambda *a,**kw:(torch.tensor([[1,2,3]]),torch.tensor([[1,1,0]]))
        m.offload_text_encoder=False; ref=m.encode_prompt('test')
        m.offload_text_encoder=True; got=m.encode_prompt('test')
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(ref,got)))
        self.assertEqual(moves,['cpu','cpu'])
        class Broken(Encoder):
            def __call__(self,*a): raise RuntimeError('injected encoder failure')
        m.text_encoder=Broken(); moves.clear()
        with self.assertRaisesRegex(RuntimeError,'injected'): m.encode_prompt('test')
        self.assertEqual(moves[-1],'cpu')

if __name__=='__main__': unittest.main()
