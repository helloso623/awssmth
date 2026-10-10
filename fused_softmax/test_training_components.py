"""Host component parity tests; no optimizer or pretrained CPU training."""
import copy
import json
from pathlib import Path
import unittest
import torch
from torch import nn
from torch.nn.utils import parametrize
from transformers import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts,GptOssTopKRouter
from gpt_oss_experts import StaticExperts,StaticTopKRouter

METRICS = []

class PackedDelta(nn.Module):
    def __init__(self,a,b,scale):
        super().__init__()
        self.A=nn.Parameter(a.detach().clone())
        self.B=nn.Parameter(b.detach().clone())
        self.scale=scale
    def forward(self,w): return w+self.scale*torch.bmm(self.A,self.B)

class ExpertParity(unittest.TestCase):
    def compare(self, nonzero, chunk, enabled=True):
        torch.manual_seed(412)
        cfg=GptOssConfig(hidden_size=8,intermediate_size=6,num_local_experts=4,num_experts_per_tok=2)
        reference=GptOssExperts(cfg).double()
        router=GptOssTopKRouter(cfg).double()
        with torch.no_grad():
            for p in reference.parameters(): p.normal_(0.,.4)
            for p in router.parameters(): p.normal_(0.,.2)
            reference.gate_up_proj_bias[0,0]=9.
            reference.gate_up_proj_bias[0,1]=-9.
            reference.gate_up_proj_bias[1,1]=9.
        candidate=StaticExperts(4,8,6,alpha=32,dtype=torch.float64,expert_chunk_size=chunk)
        candidate.load_base(reference)
        with torch.no_grad():
            if nonzero:
                candidate.gate_lora_B.normal_(0.,.03)
                candidate.down_lora_B.normal_(0.,.03)
        candidate.adapters_enabled=enabled
        # Base derivatives enabled only in this functional parity test.
        for p in candidate.parameters(): p.requires_grad_(True)
        if enabled:
            gate=PackedDelta(candidate.gate_lora_A,candidate.gate_lora_B,candidate.scale)
            down=PackedDelta(candidate.down_lora_A,candidate.down_lora_B,candidate.scale)
            parametrize.register_parametrization(reference,'gate_up_proj',gate)
            parametrize.register_parametrization(reference,'down_proj',down)
        candidate_router=StaticTopKRouter(router)
        x=torch.randn(2,5,8,dtype=torch.float64,requires_grad=True)
        other=x.detach().clone().requires_grad_()
        weights,indices=router(x)
        other_weights,other_indices=candidate_router(other)
        self.assertTrue(torch.equal(indices,other_indices))
        self.assertTrue(torch.equal(weights,other_weights))
        want=reference(x,router_indices=indices,routing_weights=weights)
        got=candidate(other,router_indices=other_indices,routing_weights=other_weights)
        torch.testing.assert_close(got,want,rtol=1e-11,atol=1e-11)
        upstream=torch.randn_like(want)
        (want*upstream).sum().backward()
        (got*upstream).sum().backward()
        comparisons={'input':(other.grad,x.grad),'router_weight':(candidate_router.weight.grad,router.weight.grad),
            'router_bias':(candidate_router.bias.grad,router.bias.grad),
            'gate_bias':(candidate.gate_up_proj_bias.grad,reference.gate_up_proj_bias.grad),
            'down_bias':(candidate.down_proj_bias.grad,reference.down_proj_bias.grad)}
        if enabled:
            comparisons.update(gate_A=(candidate.gate_lora_A.grad,gate.A.grad),
                gate_B=(candidate.gate_lora_B.grad,gate.B.grad),
                down_A=(candidate.down_lora_A.grad,down.A.grad),
                down_B=(candidate.down_lora_B.grad,down.B.grad),
                gate_base=(candidate.gate_up_proj.grad,reference.parametrizations.gate_up_proj.original.grad),
                down_base=(candidate.down_proj.grad,reference.parametrizations.down_proj.original.grad))
            self.assertGreater(candidate.gate_lora_B.grad.abs().max().item(),0.)
            self.assertGreater(candidate.down_lora_B.grad.abs().max().item(),0.)
        else:
            comparisons.update(gate_base=(candidate.gate_up_proj.grad,reference.gate_up_proj.grad),
                               down_base=(candidate.down_proj.grad,reference.down_proj.grad))
        errors={}
        for name,(actual,expected) in comparisons.items():
            torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)
            errors[name]=float((actual-expected).abs().max())
        METRICS.append(dict(nonzero_adapters=nonzero,adapters_enabled=enabled,expert_chunk_size=chunk,
                            max_forward_abs_error=float((got-want).abs().max().detach()),gradient_errors=errors))

    def test_zero_adapters_streamed(self): self.compare(False,1)
    def test_nonzero_adapters_streamed(self): self.compare(True,1)
    def test_nonzero_adapters_vectorized(self): self.compare(True,4)
    def test_disabled_adapters(self): self.compare(True,1,False)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ExpertParity)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    Path(__file__).with_name('expert_parity.json').write_text(json.dumps(dict(
        ok=result.wasSuccessful(),tests=result.testsRun,optimizer_steps=0,
        scope='small deterministic host component parity; no pretrained weights or Trainium execution',
        metrics=METRICS),indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
