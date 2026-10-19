"""Checkpoint-free contracts for request-local KV workspace ownership and values."""
import gc
import importlib.util
from pathlib import Path
import unittest
import weakref

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "action_kv", ROOT / "src/vpp2/models/wan21_14b/action_kv.py"
)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class PrefixTests(unittest.TestCase):
    def tensors(self,dtype=torch.float32):
        torch.manual_seed(19)
        return [torch.randn(2,n,8,dtype=dtype)for n in (7,7,3,3)]

    def test_reuse_changes_only_suffix_and_releases_old_prefix(self):
        for dtype in [torch.float32,torch.float16,torch.bfloat16]:
            k,v,a,b=self.tensors(dtype);reference_k=k.clone();reference_v=v.clone()
            cache=dict(k=k,v=v);old_k=weakref.ref(k);old_v=weakref.ref(v)
            with torch.no_grad():
                joined=helper.join_action_kv(cache,k,v,a,b)
                self.assertTrue(torch.equal(joined[0],torch.cat([k,a],1)))
                self.assertTrue(torch.equal(joined[1],torch.cat([v,b],1)))
                ptrs=[x.data_ptr()for x in joined]
                del k,v;gc.collect()
                self.assertIsNone(old_k());self.assertIsNone(old_v())
                for step in range(3):
                    a=a+step;b=b-step
                    joined=helper.join_action_kv(cache,cache['k'],cache['v'],a,b)
                    self.assertEqual(ptrs,[x.data_ptr()for x in joined])
                    self.assertTrue(torch.equal(joined[0],torch.cat([reference_k,a],1)))
                    self.assertTrue(torch.equal(joined[1],torch.cat([reference_v,b],1)))

    def test_new_prefix_shape_noncontiguous_and_dtype_fallback(self):
        k,v,a,b=self.tensors();cache=dict(k=k,v=v)
        with torch.no_grad():
            helper.join_action_kv(cache,k,v,a,b)
            # Same-sized replacement must refresh the prefix, not reuse old values.
            cache['k']=k+7;cache['v']=v-3
            k2,v2=cache['k'],cache['v']
            out=helper.join_action_kv(cache,k2,v2,a,b)
            self.assertTrue(torch.equal(out[0],torch.cat([k2,a],1)))
            self.assertTrue(torch.equal(out[1],torch.cat([v2,b],1)))
            out=helper.join_action_kv(cache,cache['k'],cache['v'],a[:,:1],b[:,:1])
            self.assertEqual(out[0].shape,(2,8,8));self.assertTrue(out[0].is_contiguous())
            k=k.transpose(1,2).contiguous().transpose(1,2);v=v.transpose(1,2).contiguous().transpose(1,2)
            cache=dict(k=k,v=v);out=helper.join_action_kv(cache,k,v,a,b)
            self.assertTrue(torch.equal(out[0],torch.cat([k,a],1)))
            a=a.double();b=b.double();cache=dict(k=k,v=v)
            out=helper.join_action_kv(cache,k,v,a,b)
            self.assertEqual(out[0].dtype,torch.float64)
            self.assertNotIn('_action_joint_k',cache)

    def test_grad_enabled_keeps_cat_graph_and_cache(self):
        k,v,a,b=self.tensors();a.requires_grad_();cache=dict(k=k,v=v)
        out=helper.join_action_kv(cache,k,v,a,b)
        out[0].sum().backward()
        self.assertTrue(torch.equal(a.grad,torch.ones_like(a)))
        self.assertIs(cache['k'],k);self.assertNotIn('_action_joint_k',cache)

    def test_independent_requests_never_share_a_workspace(self):
        k,v,a,b=self.tensors()
        left=dict(k=k.clone(),v=v.clone());right=dict(k=k.clone(),v=v.clone())
        with torch.no_grad():
            x=helper.join_action_kv(left,left['k'],left['v'],a,b)
            y=helper.join_action_kv(right,right['k'],right['v'],a,b)
            snapshot=[item.clone()for item in y]
            self.assertTrue(all(xi.data_ptr()!=yi.data_ptr()for xi,yi in zip(x,y)))
            helper.join_action_kv(left,left['k'],left['v'],a+9,b-2)
            self.assertTrue(all(torch.equal(yi,old)for yi,old in zip(y,snapshot)))


if __name__ == '__main__':
    unittest.main()
