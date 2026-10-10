"""Actual-chip component compilation/parity/timing. NOT a 20B training update."""
import copy
import json
import os
from pathlib import Path
import resource
import time
import traceback

def run():
    if os.environ.get('NEURON_RT_VISIBLE_CORES') != '0':
        raise RuntimeError('This single-process component probe uses allocated core0 only')
    import torch
    import torch_neuronx
    import torch_xla.core.xla_model as xm
    from transformers import GptOssConfig
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts,GptOssTopKRouter
    from torch.nn.utils import parametrize
    from test_training_components import PackedDelta
    from gpt_oss_experts import StaticExperts,StaticTopKRouter
    torch.manual_seed(412)
    cfg=GptOssConfig(hidden_size=8,intermediate_size=6,num_local_experts=4,num_experts_per_tok=2)
    original=GptOssExperts(cfg).float()
    router=GptOssTopKRouter(cfg).float()
    with torch.no_grad():
        for p in original.parameters(): p.normal_(0.,.4)
        for p in router.parameters(): p.normal_(0.,.2)
        original.gate_up_proj_bias[0,0]=9.
        original.gate_up_proj_bias[0,1]=-9.
    candidate=StaticExperts(4,8,6,alpha=32,dtype=torch.float32,expert_chunk_size=1)
    candidate.load_base(original)
    with torch.no_grad():
        candidate.gate_lora_B.normal_(0.,.03)
        candidate.down_lora_B.normal_(0.,.03)
    gate=PackedDelta(candidate.gate_lora_A,candidate.gate_lora_B,candidate.scale)
    down=PackedDelta(candidate.down_lora_A,candidate.down_lora_B,candidate.scale)
    parametrize.register_parametrization(original,'gate_up_proj',gate)
    parametrize.register_parametrization(original,'down_proj',down)
    x=torch.randn(2,5,8,requires_grad=True)
    upstream=torch.randn_like(x)
    weights,indices=router(x)
    expected=original(x,router_indices=indices,routing_weights=weights)
    (expected*upstream).sum().backward()
    device=xm.xla_device()
    if device.type != 'xla': raise RuntimeError('No XLA device')
    candidate=candidate.to(device)
    device_router=StaticTopKRouter(router).to(device)
    device_router.zero_grad(set_to_none=True)
    device_x=x.detach().clone().to(device).requires_grad_()
    device_upstream=upstream.to(device)
    # Inspect the original routing operators' autograd metadata on XLA.
    raw_router=copy.deepcopy(router).to(device)
    raw_logits=torch.nn.functional.linear(device_x.reshape(-1,8),raw_router.weight,raw_router.bias)
    raw_values,raw_indices=torch.topk(raw_logits,2,dim=-1)
    raw_prob=torch.softmax(raw_values,dim=1)
    raw_affinity=torch.zeros_like(raw_logits).scatter_(1,raw_indices,raw_prob)
    routing_autograd_audit=dict(logits_requires_grad=raw_logits.requires_grad,
        topk_values_requires_grad=raw_values.requires_grad,
        softmax_requires_grad=raw_prob.requires_grad,scatter_inplace_requires_grad=raw_affinity.requires_grad)
    def step():
        candidate.zero_grad(set_to_none=True)
        device_router.zero_grad(set_to_none=True)
        device_x.grad=None
        weights,indices=device_router(device_x)
        output=candidate(device_x,router_indices=indices,routing_weights=weights)
        loss=(output*device_upstream).sum()
        loss.backward()
        xm.mark_step()
        xm.wait_device_ops()
        return output
    start=time.perf_counter()
    output=step()
    first=time.perf_counter()-start
    torch.testing.assert_close(output.cpu(),expected.detach(),rtol=1e-4,atol=1e-4)
    gradients={'input':(device_x.grad,x.grad),'router_weight':(device_router.weight.grad,router.weight.grad),
        'router_bias':(device_router.bias.grad,router.bias.grad),
        'gate_A':(candidate.gate_lora_A.grad,gate.A.grad),'gate_B':(candidate.gate_lora_B.grad,gate.B.grad),
        'down_A':(candidate.down_lora_A.grad,down.A.grad),'down_B':(candidate.down_lora_B.grad,down.B.grad)}
    errors={}
    mismatches={}
    for name,(actual,want) in gradients.items():
        if actual is None or want is None:
            mismatches[name]=dict(actual_missing=actual is None,expected_missing=want is None)
            continue
        actual=actual.cpu()
        errors[name]=float((actual-want).abs().max())
        try: torch.testing.assert_close(actual,want,rtol=5e-4,atol=5e-4)
        except AssertionError as exc:
            mismatches[name]=dict(detail=str(exc),actual_norm=float(actual.norm()),
                                 expected_norm=float(want.norm()),
                                 dot_ratio=float((actual*want).sum()/(want.square().sum()+1e-12)))
    if mismatches:
        return dict(ok=False,stage='backward_parity',gradient_errors=errors,
                    mismatches=mismatches,optimizer_steps=0,
                    routing_autograd_audit=routing_autograd_audit,
                    max_forward_abs_error=float((output.cpu()-expected.detach()).abs().max()))
    for _ in range(2): step()
    times=[]
    for _ in range(5):
        start=time.perf_counter();step();times.append(time.perf_counter()-start)
    try: memory=xm.get_memory_info(device)
    except Exception as exc: memory={'unavailable':str(exc)}
    return dict(ok=True,component='small deterministic GPT-OSS packed experts, rank32',
        cores=[0],lnc=2,optimizer_steps=0,pretrained_weights_loaded=False,
        compile_and_first_forward_backward_s=first,warm_forward_backward_s=times,
        timing='synchronous XLA forward/backward wall time; excludes first compilation; inputs resident',
        hbm_memory_info=memory,peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        max_forward_abs_error=float((output.cpu()-expected.detach()).abs().max()),gradient_errors=errors,
        routing_autograd_audit=routing_autograd_audit)

if __name__=='__main__':
    root=Path(__file__).parent
    try: result=run()
    except Exception: result=dict(ok=False,failure=traceback.format_exc(),optimizer_steps=0)
    (root/'expert_trainium_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    raise SystemExit(0 if result['ok'] else 1)
