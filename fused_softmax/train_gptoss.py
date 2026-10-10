"""TP4 full-model execution milestone: step, checkpoint reload, updated sampling.

--small is architectural validation with random weights. --pretrained uses the
pinned real20B checkpoint. Short sequence is an execution test, not a rollout.
"""
import argparse,json,os,time,traceback
from pathlib import Path

def run(args):
    allowed=os.environ.get('NEURON_RT_VISIBLE_CORES')
    if allowed!='0,1,2,3': raise RuntimeError('Require all four allocated cores at entry')
    import torch, torch_neuronx
    import torch.distributed as dist
    import torch_xla.core.xla_model as xm
    from neuronx_distributed.parallel_layers import parallel_state as ps
    from neuronx_distributed.parallel_layers.mappings import gather_from_tensor_model_parallel_region
    from transformers import GptOssConfig,GptOssForCausalLM
    from gptoss_tp import build_model,load_pretrained,log_probs,hidden,adapter_state,load_adapter,adapters_enabled,enable_activation_checkpointing
    dist.init_process_group('xla');ps.initialize_model_parallel(tensor_model_parallel_size=4)
    rank=ps.get_tensor_model_parallel_rank()
    torch.manual_seed(612)
    if args.small:
        cfg=GptOssConfig(hidden_size=32,intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,
            num_key_value_heads=4,head_dim=8,num_local_experts=4,num_experts_per_tok=2,vocab_size=128,
            layer_types=['sliding_attention','full_attention'],max_position_embeddings=128,
            rope_scaling={'rope_type':'default'})
        original=GptOssForCausalLM(cfg).to(torch.bfloat16)
        fixture=Path('small_gptoss_fixture')
        if rank==0:
            from safetensors.torch import save_file
            fixture.mkdir(exist_ok=True)
            state={k:v.contiguous() for k,v in original.state_dict().items()}
            save_file(state,str(fixture/'model.safetensors'))
            (fixture/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{k:'model.safetensors' for k in state}}))
        dist.barrier()
        path=fixture
    else:
        cfg=GptOssConfig(**json.loads(Path('gpt_oss_metadata/config.json').read_text()))
        path=Path(args.checkpoint)
    print(f'RANK{rank} stage=build',flush=True)
    model=build_model(cfg,rank,alpha=args.alpha)
    print(f'RANK{rank} stage=load_weights',flush=True)
    placement=load_pretrained(model,path,rank)
    Path(f'{args.output}_placement_rank{rank}.json').write_text(json.dumps(placement,indent=2))
    device=xm.xla_device()
    model.to(device);xm.mark_step();xm.wait_device_ops()
    if args.activation_checkpoint:
        enable_activation_checkpointing(model,segment_graph=args.segment_graph)
    def token_logps():
        return log_probs(model,tokens,chunk_size=args.logp_chunk,
            checkpoint_head=args.logp_chunk is not None,sync_chunks=args.segment_graph)
    memory=[dict(stage='weights',**xm.get_memory_info(device))]
    tokens=torch.randint(0,cfg.vocab_size,(1,args.sequence),device=device)
    if args.small:
        host_tokens=tokens.cpu()
        with torch.no_grad():
            logits=original(host_tokens,use_cache=False).logits.float()
            expected=torch.log_softmax(logits[:,:-1,:],-1).gather(-1,host_tokens[:,1:,None]).squeeze(-1)
    model.train()
    initial=adapter_state(model)
    print(f'RANK{rank} stage=reference_logps',flush=True)
    with torch.no_grad():
        # Fixed initial adapter, enabled execution path. Disabling adapters
        # changed BF16/compiler accumulation in the architecture test.
        reference=token_logps()
        xm.mark_step();xm.wait_device_ops();reference_host=reference.cpu()
    print(f'RANK{rank} stage=forward_backward',flush=True)
    start=time.perf_counter()
    lp=token_logps()
    # This milestone verifies execution; online RL replaces this diagnostic NLL.
    loss=-lp.mean();loss.backward();xm.mark_step();xm.wait_device_ops()
    if args.small: torch.testing.assert_close(lp.detach().cpu(),expected,rtol=.015,atol=.03)
    torch.testing.assert_close(lp.detach().cpu(),reference_host,rtol=1e-5,atol=1e-5)
    memory.append(dict(stage='backward',**xm.get_memory_info(device)))
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=4e-5,betas=(.9,.95),eps=1e-8)
    base_gradients=[n for n,p in model.named_parameters() if not p.requires_grad and p.grad is not None]
    if base_gradients: raise RuntimeError('Frozen base unexpectedly allocated gradients: '+str(base_gradients))
    optimizer.step();xm.mark_step();xm.wait_device_ops()
    elapsed=time.perf_counter()-start
    memory.append(dict(stage='optimizer',**xm.get_memory_info(device)))
    updated=adapter_state(model)
    changes={n:float((updated[n]-initial[n]).abs().max()) for n in updated}
    if not any(changes.values()): raise RuntimeError('No adapter parameter changed')
    out=Path(args.output);out.mkdir(exist_ok=True)
    checkpoint=out/f'adapter_rank{rank}.pt'
    torch.save(updated,checkpoint)
    load_adapter(model,initial);xm.mark_step();xm.wait_device_ops()
    load_adapter(model,torch.load(checkpoint,weights_only=True));xm.mark_step();xm.wait_device_ops()
    for n,p in adapter_state(model).items(): torch.testing.assert_close(p,updated[n],rtol=0,atol=0)
    print(f'RANK{rank} stage=updated_generation',flush=True)
    model.eval()
    with torch.no_grad():
        local=model.lm_head(hidden(model,tokens)[:,-1:,:]).float()
        logits=gather_from_tensor_model_parallel_region(local)
        next_token=logits.argmax(-1).cpu().tolist()
    result=dict(ok=True,rank=rank,pretrained=not args.small,optimizer_steps=1,sequence=args.sequence,
        test_type='short execution validation, not full kernel rollout',step_s=elapsed,memory=memory,
        parameter_max_change=max(changes.values()),changed_tensors=sum(v>0 for v in changes.values()),
        checkpoint=str(checkpoint),reload_exact=True,next_token_from_updated_checkpoint=next_token,
        base_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
        adapter_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    result.update(base_gradients=base_gradients,optimizer_parameter_count=sum(p.numel() for g in optimizer.param_groups for p in g['params']),
        optimizer_state_parameters=len(optimizer.state),activation_checkpoint=args.activation_checkpoint,
        logp_chunk=args.logp_chunk,segment_graph=args.segment_graph)
    (out/f'evidence_rank{rank}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--small',action='store_true');p.add_argument('--checkpoint');p.add_argument('--sequence',type=int,default=16)
    p.add_argument('--alpha',type=float,default=32);p.add_argument('--output',default='gptoss_step')
    p.add_argument('--activation-checkpoint',action='store_true');p.add_argument('--segment-graph',action='store_true');p.add_argument('--logp-chunk',type=int)
    args=p.parse_args()
    try: run(args)
    except Exception:
        error=traceback.format_exc();print(error,flush=True)
        Path(f'{args.output}_failure_rank{os.environ.get("RANK","unknown")}.json').write_text(json.dumps({'error':error},indent=2))
        raise
