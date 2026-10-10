"""Four-core expert parity and real component Adam update; no pretrained claim."""
import json, os, time, traceback
from pathlib import Path

def run():
    assigned=os.environ.get('NEURON_RT_VISIBLE_CORES')
    if assigned != '0,1,2,3':
        raise RuntimeError('Require exactly the four authorized logical cores at process entry')
    import torch
    import torch_neuronx
    import torch_xla.core.xla_model as xm
    import torch.distributed as dist
    from neuronx_distributed.parallel_layers import parallel_state as ps
    from neuronx_distributed.parallel_layers.mappings import copy_to_tensor_model_parallel_region as copy_tp, reduce_from_tensor_model_parallel_region as reduce_tp
    from gpt_oss_experts import StaticExperts
    # torch_neuronx narrows each torchrun worker to its LOCAL_RANK core.
    dist.init_process_group('xla')
    ps.initialize_model_parallel(tensor_model_parallel_size=4)
    rank=ps.get_tensor_model_parallel_rank()
    torch.manual_seed(812)
    full=StaticExperts(4,8,8,alpha=32,dtype=torch.float32)
    with torch.no_grad():
        for name,p in full.named_parameters():
            if 'lora' not in name: p.normal_(0,.2)
            elif name.endswith('B'): p.normal_(0,.02)
    x=torch.randn(1,3,8,requires_grad=True)
    aff=torch.softmax(torch.randn(3,4),-1).requires_grad_()
    up=torch.randn_like(x)
    expected=full(x,routing_weights=aff)
    (expected*up).sum().backward()
    part=StaticExperts(4,8,8,alpha=32,tp_size=4,tp_rank=rank,dtype=torch.float32,copy_to_tp=copy_tp,reduce_from_tp=reduce_tp)
    lo,hi=rank*2,(rank+1)*2
    slices={
      'gate_up_proj':(slice(None),slice(None),slice(2*lo,2*hi)),
      'gate_up_proj_bias':(slice(None),slice(2*lo,2*hi)),
      'down_proj':(slice(None),slice(lo,hi),slice(None)),
      'gate_lora_B':(slice(None),slice(None),slice(2*lo,2*hi)),
      'down_lora_A':(slice(None),slice(lo,hi),slice(None))}
    source=dict(full.named_parameters())
    with torch.no_grad():
        for name,p in part.named_parameters(): p.copy_(source[name][slices.get(name,Ellipsis)])
    device=xm.xla_device()
    part.to(device)
    dx=x.detach().to(device).requires_grad_()
    da=aff.detach().to(device).requires_grad_()
    start=time.perf_counter()
    output=part(dx,routing_weights=da)
    (output*up.to(device)).sum().backward()
    xm.mark_step();xm.wait_device_ops()
    elapsed=time.perf_counter()-start
    torch.testing.assert_close(output.cpu(),expected.detach(),rtol=5e-4,atol=5e-4)
    errors={}
    for name,p in part.named_parameters():
        if not p.requires_grad: continue
        want=source[name].grad[slices.get(name,Ellipsis)]
        actual=p.grad.cpu()
        errors[name]=float((actual-want).abs().max())
        torch.testing.assert_close(actual,want,rtol=5e-4,atol=5e-4)
    # Input gradient is replicated and reduced by the projection mappings.
    torch.testing.assert_close(dx.grad.cpu(),x.grad,rtol=5e-4,atol=5e-4)
    torch.testing.assert_close(da.grad.cpu(),aff.grad,rtol=5e-4,atol=5e-4)
    before={n:p.detach().cpu().clone() for n,p in part.named_parameters() if p.requires_grad}
    optimizer=torch.optim.Adam([p for p in part.parameters() if p.requires_grad],lr=4e-5,betas=(.9,.95),eps=1e-8)
    optimizer.step();xm.mark_step();xm.wait_device_ops()
    deltas={n:float((p.detach().cpu()-before[n]).abs().max()) for n,p in part.named_parameters() if p.requires_grad}
    if not any(v>0 for v in deltas.values()): raise RuntimeError('No adapter changed')
    result=dict(ok=True,rank=rank,tp=4,optimizer_steps=1,pretrained_weights_loaded=False,
       forward_max_abs=float((output.cpu()-expected.detach()).abs().max()),gradient_errors=errors,
       parameter_changes=deltas,compile_and_backward_s=elapsed,hbm=xm.get_memory_info(device),
       placement={n:list(p.shape) for n,p in part.named_parameters()})
    return result

if __name__=='__main__':
    rank=os.environ.get('RANK','unknown')
    try: result=run()
    except Exception: result=dict(ok=False,stage='tp4_component',error=traceback.format_exc())
    Path(f'tp4_rank{rank}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    raise SystemExit(0 if result['ok'] else 1)
