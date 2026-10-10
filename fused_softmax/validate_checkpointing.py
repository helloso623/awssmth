"""Pending hardware validation: identical small GPT-OSS updates with/without
NxD's installed XLA checkpoint wrapper. Runs sequentially on allocated TP4.
"""
import json,os,time,traceback
from pathlib import Path

def run():
    allocation=os.environ.get('NEURON_RT_VISIBLE_CORES')
    if allocation!='0,1,2,3': raise RuntimeError('Require authorized TP4 allocation')
    import torch,torch_neuronx
    import torch.distributed as dist
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as metrics
    from neuronx_distributed.parallel_layers import parallel_state as ps
    from transformers import GptOssConfig
    from gptoss_tp import build_model,load_pretrained,log_probs,adapter_state,load_adapter,enable_activation_checkpointing
    dist.init_process_group('xla');ps.initialize_model_parallel(tensor_model_parallel_size=4)
    rank=ps.get_tensor_model_parallel_rank()
    cfg=GptOssConfig(hidden_size=32,intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,
        num_key_value_heads=4,head_dim=8,num_local_experts=4,num_experts_per_tok=2,vocab_size=128,
        layer_types=['sliding_attention','full_attention'],max_position_embeddings=128,rope_scaling={'rope_type':'default'})
    torch.manual_seed(612)
    model=build_model(cfg,rank)
    load_pretrained(model,'small_gptoss_fixture',rank)
    device=xm.xla_device();model.to(device);xm.mark_step();xm.wait_device_ops()
    initial=adapter_state(model)
    tokens=torch.arange(16,device=device)[None,:]
    data={}
    for enabled in (False,True):
        if enabled: enable_activation_checkpointing(model)
        load_adapter(model,initial);model.train()
        params={n:p for n,p in model.named_parameters() if p.requires_grad}
        optimizer=torch.optim.Adam(list(params.values()),lr=4e-5,betas=(.9,.95),eps=1e-8)
        def step():
            optimizer.zero_grad(set_to_none=True)
            lp=log_probs(model,tokens)
            loss=-lp.mean();loss.backward();xm.mark_step();xm.wait_device_ops()
            gradient={n:p.grad.detach().cpu().clone() for n,p in params.items()}
            optimizer.step();xm.mark_step();xm.wait_device_ops()
            return lp.detach().cpu(),float(loss.detach().cpu()),gradient,adapter_state(model)
        lp,loss,grad,updated=step()
        if not enabled:
            expected=(lp,loss,grad,updated)
        else:
            e_lp,e_loss,e_grad,e_updated=expected
            torch.testing.assert_close(lp,e_lp,rtol=1e-5,atol=1e-5)
            if abs(loss-e_loss)>1e-5: raise AssertionError('Checkpoint loss mismatch')
            for name in grad:
                torch.testing.assert_close(grad[name],e_grad[name],rtol=2e-3,atol=2e-5)
                torch.testing.assert_close(updated[name],e_updated[name],rtol=1e-6,atol=2e-6)
        for _ in range(2): step()
        metrics.clear_all()
        times=[]
        for _ in range(3):
            start=time.perf_counter();step();times.append(time.perf_counter()-start)
        data[str(enabled)]=dict(loss=loss,warm_update_s=times,memory=xm.get_memory_info(device),
            metrics=metrics.metrics_report(),trainable_parameters=sum(p.numel() for p in params.values()),
            base_gradients=[n for n,p in model.named_parameters() if not p.requires_grad and p.grad is not None])
    result=dict(ok=True,rank=rank,small_only=True,forward_loss_gradient_update_parity=True,measurements=data)
    Path(f'checkpoint_validation_rank{rank}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    try: run()
    except Exception:
        Path(f'checkpoint_validation_failure_rank{os.environ.get("RANK","unknown")}.json').write_text(json.dumps({'error':traceback.format_exc()},indent=2))
        raise
