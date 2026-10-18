"""Compare the production history builder with dense genuine native frames."""
import ast
from collections import deque
import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional
import unittest
import torch

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('native_history',root/'src/vpp2/native_history.py')
protocol=importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)
source=root/'src/vpp2/policy.py'
tree=ast.parse(source.read_text())
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {
    '_normalized_to_uint8','_uint8_to_normalized'}]
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RoboDojoVPP2Policy')
cls.body=[n for n in cls.body if (isinstance(n,ast.FunctionDef) and n.name in {
    'observe','_build_history_condition','reset','reset_env'})
    or (isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='_PER_ENV_STATE' for v in n.targets))]
ns=dict(torch=torch,deque=deque,Any=Any,Dict=Dict,Optional=Optional,
        NATIVE_STEP_FIELD=protocol.NATIVE_STEP_FIELD,
        validate_native_frame=protocol.validate_native_frame,check_configuration=protocol.check_configuration)
exec(compile(ast.Module(body=[*functions,cls],type_ignores=[]),str(source),'exec'),ns)


def policy():
    p=ns['RoboDojoVPP2Policy']()
    for name in p._PER_ENV_STATE: setattr(p,name,{})
    p.history_buffer_size=177
    p.history_condition_frames=8
    p.history_stride=25
    p.replan_steps=24
    p._inference_index=0
    p._compose_observation_image=lambda o:torch.full((3,2,2),((o['tick']*7+o['env_idx'])%255)/127.5-1)
    # The actual dense recorder is included below, without model construction.
    return p


# Retain the shared production history append method too.
original=ast.parse(source.read_text())
method=next(n for n in ast.walk(original) if isinstance(n,ast.FunctionDef) and n.name=='_record_history_frame')
exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),ns)
ns['RoboDojoVPP2Policy']._record_history_frame=ns['_record_history_frame']


class NativeHistoryTests(unittest.TestCase):
    def test_actual_condition_matches_dense_through_many_replans_and_reset(self):
        dense,sparse=policy(),policy()
        for tick in range(2501):
            for env in [3,103]:
                obs={'env_idx':env,'tick':tick}
                dense.observe(obs)
                if protocol.needs_frame(tick):
                    sparse.observe(dict(obs,**{protocol.NATIVE_STEP_FIELD:tick}))
                if tick%24==0:
                    sparse._inference_index_by_env[env]=tick//24
                    protocol.validate_inference(sparse,{protocol.NATIVE_STEP_FIELD:tick},env)
                    a,am=dense._build_history_condition(dense._latest_image_by_env[env],env)
                    b,bm=sparse._build_history_condition(sparse._latest_image_by_env[env],env)
                    assert torch.equal(a,b) and am==bm
        saved=list(sparse._history_by_env[103])
        sparse.reset_env(3)
        assert 3 not in sparse._history_mode_by_env
        sparse.observe({'env_idx':3,'tick':0,protocol.NATIVE_STEP_FIELD:0})
        assert len(sparse._history_by_env[103])==len(saved)
        sparse.reset()
        assert all(not getattr(sparse,n) for n in sparse._PER_ENV_STATE)


    def test_missing_frame_duplicate_wrong_clock_and_protocol_switch_rejected(self):
        p=policy()
        for invalid_step in [None, True, 1.0, -1, 1, 24]:
            with self.assertRaises(ValueError):
                p.observe({'env_idx': 0, 'tick': 0, protocol.NATIVE_STEP_FIELD: invalid_step})
            assert not p._history_by_env and not p._history_step_by_env
        for tick in [0,17,18,19,20,21,22,23,24]:
            p.observe({'env_idx':0,'tick':tick,protocol.NATIVE_STEP_FIELD:tick})
        with self.assertRaises(ValueError):
            p.observe({'env_idx': 0, 'tick': 24, protocol.NATIVE_STEP_FIELD: 24})
        assert p._history_step_by_env[0] == 24 and len(p._history_by_env[0]) == 9
        with self.assertRaisesRegex(ValueError,'switching'): p.observe({'env_idx':0,'tick':25})
        p._inference_index_by_env[0]=1
        protocol.validate_inference(p,{protocol.NATIVE_STEP_FIELD:24},0)
        with self.assertRaisesRegex(ValueError,'current'): protocol.validate_inference(p,{protocol.NATIVE_STEP_FIELD:48},0)
        p._history_by_env[0].clear()
        p._inference_index_by_env[0]=2
        p._history_step_by_env[0]=48
        with self.assertRaisesRegex(ValueError,'Missing required'): protocol.validate_inference(p,{protocol.NATIVE_STEP_FIELD:48},0)
        for field, value in [('replan_steps', 32), ('history_stride', 24), ('history_condition_frames', 7)]:
            invalid = policy()
            setattr(invalid, field, value)
            with self.assertRaisesRegex(ValueError, 'replan24'):
                invalid.observe({'env_idx': 0, 'tick': 0, protocol.NATIVE_STEP_FIELD: 0})

if __name__ == "__main__":
    unittest.main()
