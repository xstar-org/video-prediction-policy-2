import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Sequence, Union
import unittest
import torch

source=Path(__file__).resolve().parents[1]/'src/vpp2/models/wan21_14b/vpp2.py'
tree=ast.parse(source.read_text())
cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='VPP2')
cls.bases=[];cls.keywords=[]
cls.body=[x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name in {'infer_action','_sample_action_video_latents'}]
ns=dict(torch=torch,Optional=Optional,Any=Any,Sequence=Sequence,Union=Union)
exec(compile(ast.Module(body=[cls],type_ignores=[]),str(source),'exec'),ns)
Core=ns['VPP2']

class BatchContract(unittest.TestCase):
 def model(self):
  m=Core();m.device='cpu';m.torch_dtype=torch.float32;m.eval=lambda:None
  m.video_expert=SimpleNamespace(video_attention_mask_mode='bidirectional',fuse_vae_embedding_in_latents=False)
  m.proprio_dim=2;m.action_chunk_size=1;m.action_visible_video_frames=1
  m.action_expert=SimpleNamespace(action_dim=2);m.vae=SimpleNamespace(temporal_downsample_factor=4)
  m.train_video_scheduler=SimpleNamespace(num_train_timesteps=1000)
  m.infer_action_scheduler=SimpleNamespace(build_inference_schedule=lambda **kw:(torch.tensor([1.,.5]),torch.tensor([.1,.1])),step=lambda pred,delta,x:x-delta*pred)
  m._encode_separate_condition_latents=lambda x,**kw:torch.zeros((x.shape[0],2,1,2,2))
  m._encode_clip_fea_from_video=lambda x:torch.zeros((x.shape[0],1,2))
  m.encode_prompt=lambda p:(torch.ones((len(p) if isinstance(p,list) else 1,3,2)),torch.ones((len(p) if isinstance(p,list) else 1,3),dtype=torch.bool))
  m._append_proprio_to_context=lambda context,context_mask,proprio:(context+proprio[:,None,:],context_mask)
  m._build_frozen_video_cache=lambda **kw:(kw['latents'],None,1)
  m._predict_action_noise_with_cache=lambda **kw:kw['latents_action']*.1+kw['video_kv_cache'].mean((1,2,3,4))[:,None,None]+kw['context'].mean((1,2))[:,None,None]
  m._unpack_action_tokens=lambda x:x
  return m
 def inputs(self,n):
  return dict(prompt=['task']*n,input_image=torch.zeros(n,3,16,16),condition_video=torch.zeros(n,3,9,16,16),proprio=torch.arange(n*2).reshape(n,2).float(),action_horizon=4,num_video_frames=17,num_inference_steps=2)
 def test_batch_matches_serial_with_distinct_seeds(self):
  m=self.model();kw=self.inputs(3);seeds=[3,19,101];videos=[7,42,1003]
  batched=m.infer_action(**kw,seed=seeds,video_seed=videos)['action']
  serial=[]
  for i in range(3):
   row={k:([v[i]] if isinstance(v,list) else v[i:i+1]) if isinstance(v,(list,torch.Tensor)) else v for k,v in kw.items()}
   serial.append(m.infer_action(**row,seed=seeds[i],video_seed=videos[i])['action'])
  self.assertTrue(torch.equal(batched,torch.stack(serial)))
 def test_order_is_independent_and_global_rng_unchanged(self):
  m=self.model();kw=self.inputs(2);before=torch.random.get_rng_state().clone()
  a=m.infer_action(**kw,seed=[2,9],video_seed=[12,19])['action']
  reverse={k:(list(reversed(v)) if isinstance(v,list) else v.flip(0)) if isinstance(v,(list,torch.Tensor)) else v for k,v in kw.items()}
  b=m.infer_action(**reverse,seed=[9,2],video_seed=[19,12])['action']
  self.assertTrue(torch.equal(a,b.flip(0)));self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
 def test_seed_count_mismatch_is_rejected(self):
  with self.assertRaisesRegex(ValueError,'One seed'):
   self.model().infer_action(**self.inputs(2),seed=[1])

if __name__=='__main__':unittest.main()
