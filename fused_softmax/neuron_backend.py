"""Direct TP4 GPT-OSS backend for fresh-rollout TTT-Discover stages.

Run via run_trainium.sh; generation and updates exit before kernel evaluation.
All LM forward/backward/Adam work executes on Neuron. CPU stores tokens,
scalar logprobs and checkpoints; categorical sampling uses CPU logits only.
"""
import argparse,json,os,time,traceback
from pathlib import Path

def execute(request,output):
    assigned=os.environ.get('NEURON_RT_VISIBLE_CORES')
    if assigned!='0,1,2,3':raise RuntimeError('Require allocated logical cores0–3 at entry')
    if request['model']!='openai/gpt-oss-20b':raise ValueError('GPT-OSS-20B required')
    import torch,torch_neuronx
    import torch.distributed as dist
    import torch_xla.core.xla_model as xm
    from neuronx_distributed.parallel_layers import parallel_state as ps
    from neuronx_distributed.parallel_layers.mappings import gather_from_tensor_model_parallel_region
    from transformers import GptOssConfig,AutoTokenizer
    from gptoss_tp import (build_model,load_pretrained,log_probs,hidden,adapter_state,load_adapter,
        FixedKVCache,cached_hidden,enable_activation_checkpointing)
    from objective import token_advantages,importance_loss
    dist.init_process_group('xla');ps.initialize_model_parallel(tensor_model_parallel_size=4)
    rank=ps.get_tensor_model_parallel_rank();device=xm.xla_device()
    options=request['backend_config'];path=Path(options['pretrained_checkpoint'])
    cfg=GptOssConfig(**json.loads((path/'config.json').read_text()))
    torch.manual_seed(options.get('initial_seed',612))
    print(f'RANK{rank} stage=build_pretrained',flush=True)
    model=build_model(cfg,rank,alpha=options['alpha'])
    placement=load_pretrained(model,path,rank)
    reference_dir=Path(options['reference_dir']);reference_dir.mkdir(parents=True,exist_ok=True)
    ref_file=reference_dir/f'adapter_rank{rank}.pt'
    if ref_file.exists():initial=torch.load(ref_file,weights_only=True)
    else:
        initial=adapter_state(model);torch.save(initial,ref_file)
    load_adapter(model,initial)
    checkpoint=request['checkpoint_id'];saved=None
    if checkpoint!='base':
        saved=torch.load(Path(checkpoint)/f'rank{rank}.pt',map_location='cpu',weights_only=True)
        if saved['alpha']!=options['alpha'] or saved['tp']!=4:raise ValueError('Checkpoint configuration mismatch')
        load_adapter(model,saved['adapter'])
    model.to(device);xm.mark_step();xm.wait_device_ops()
    Path(str(output)+f'.placement_rank{rank}.json').write_text(json.dumps(placement,indent=2))
    base=[p for p in model.parameters() if not p.requires_grad]
    adapters=[p for p in model.parameters() if p.requires_grad]
    if any('lora_' not in n for n,p in model.named_parameters() if p.requires_grad):raise RuntimeError('Trainable base parameter')
    result=dict(ok=True,model='openai/gpt-oss-20b',tp=4,rank=rank,checkpoint_id=checkpoint)
    if request['action']=='generate':
        model.eval();tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        generation=request['generation'];capacity=int(generation['context_length'])
        if generation.get('temperature',1.)!=1.:raise ValueError('Initial verified sampler requires temperature1, no filtering')
        max_new=int(generation['max_new_tokens'])
        eos=set(json.loads((path/'generation_config.json').read_text())['eos_token_id'])
        records=[];sample_index=0
        for group in request['groups']:
            prompt_tokens=tokenizer.apply_chat_template([{'role':'user','content':group['prompt']}],
                tokenize=True,add_generation_prompt=True,reasoning_effort='low')
            if len(prompt_tokens)+max_new>capacity:raise ValueError('Requested full response budget exceeds context; no truncation')
            for _ in range(request['responses']):
                rng=torch.Generator(device='cpu').manual_seed(generation['seed']+request.get('round',0)*10000+sample_index)
                cache=FixedKVCache(cfg,capacity,device)
                ids=prompt_tokens.copy();response=[];sample_logp=[]
                start=time.perf_counter();finish='length'
                with torch.no_grad():
                    input_tokens=torch.tensor([ids],dtype=torch.long,device=device)
                    positions=torch.arange(len(ids),device=device)
                    xm.mark_step()
                    for index in range(max_new):
                        states=cached_hidden(model,input_tokens,positions,cache)
                        local=model.lm_head(states[:,-1:,:]).float()
                        logits=gather_from_tensor_model_parallel_region(local)
                        xm.mark_step();xm.wait_device_ops()
                        logp=torch.log_softmax(logits.cpu().reshape(-1),dim=-1)
                        # All ranks see the same gathered logits and seeded RNG.
                        token=int(torch.multinomial(logp.exp(),1,generator=rng))
                        response.append(token);sample_logp.append(float(logp[token]));ids.append(token)
                        if token in eos:finish='eos';break
                        text=tokenizer.decode(response,skip_special_tokens=False)
                        if '</NKI_SOURCE>' in text:finish='source_complete';break
                        input_tokens=torch.tensor([[token]],dtype=torch.long,device=device)
                        positions=torch.tensor([len(ids)-1],dtype=torch.long,device=device)
                        xm.mark_step()
                text=tokenizer.decode(response,skip_special_tokens=False)
                record=dict(parent_id=group['parent_id'],checkpoint_id=checkpoint,reference_checkpoint=request['reference_checkpoint'],
                    tokens=ids,prompt_tokens=prompt_tokens,response_tokens=response,response=text,
                    sampling_logprobs=[0.]*(len(prompt_tokens)-1)+sample_logp,
                    mask=[0.]*(len(prompt_tokens)-1)+[1.]*len(response),generation=generation,
                    sample_index=sample_index,finish_reason=finish,generation_s=time.perf_counter()-start)
                records.append(record);sample_index+=1
                del cache,states,logits,local
                xm.mark_step();xm.wait_device_ops()
                print(f'RANK{rank} generated sample{sample_index} tokens={len(response)} finish={finish}',flush=True)
        result['records']=records
    elif request['action']=='update':
        records=request['records']
        if any(r['checkpoint_id']!=checkpoint for r in records):raise ValueError('Stale rollout policy')
        if len(records)!=len(request['advantages']):raise ValueError('Advantage count mismatch')
        if options.get('activation_checkpoint',False):enable_activation_checkpointing(model,options.get('segment_graph',False))
        def lp(tokens):return log_probs(model,tokens,chunk_size=options.get('logp_chunk'),
            checkpoint_head=options.get('logp_chunk') is not None,sync_chunks=options.get('segment_graph',False))
        before=adapter_state(model)
        load_adapter(model,initial);xm.mark_step();xm.wait_device_ops()
        reference=[];model.eval()
        with torch.no_grad():
            for record in records:
                tokens=torch.tensor([record['tokens']],dtype=torch.long,device=device)
                ref=lp(tokens);xm.mark_step();xm.wait_device_ops();reference.append(ref.reshape(-1).cpu())
                del ref,tokens
        load_adapter(model,before);xm.mark_step();xm.wait_device_ops();model.train()
        masks=[torch.tensor(r['mask']) for r in records]
        sampler=[torch.tensor(r['sampling_logprobs']) for r in records]
        advantages,average_kl=token_advantages(request['advantages'],sampler,reference,masks,.1)
        optimizer=torch.optim.Adam(adapters,lr=4e-5,betas=(.9,.95),eps=1e-8)
        if saved is not None:optimizer.load_state_dict(saved['optimizer'])
        optimizer.zero_grad(set_to_none=True)
        start=time.perf_counter();losses=[]
        # One fresh batch, microbatch1; sum gradients with the authors' loss scale.
        for record,s,a,m in zip(records,sampler,advantages,masks,strict=True):
            tokens=torch.tensor([record['tokens']],dtype=torch.long,device=device)
            learner=lp(tokens).reshape(-1)
            loss=importance_loss([learner],[s.to(device)],[a.to(device)],[m.to(device)])
            loss.backward();xm.mark_step();xm.wait_device_ops();losses.append(float(loss.detach().cpu()))
            del tokens,learner,loss
        if any(p.grad is not None for p in base):raise RuntimeError('Base gradient unexpectedly allocated')
        optimizer.step();xm.mark_step();xm.wait_device_ops()
        elapsed=time.perf_counter()-start;after=adapter_state(model)
        max_change=max(float((after[n]-before[n]).abs().max()) for n in after)
        changed=sum(not torch.equal(after[n],before[n]) for n in after)
        def cpu_tree(value):
            if isinstance(value,torch.Tensor):return value.detach().cpu()
            if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
            if isinstance(value,list):return [cpu_tree(v) for v in value]
            if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
            return value
        folder=Path(request['checkpoint_dir'])/request['tag'];folder.mkdir(parents=True,exist_ok=True)
        file=folder/f'rank{rank}.pt'
        torch.save(dict(adapter=after,optimizer=cpu_tree(optimizer.state_dict()),tp=4,alpha=options['alpha']),file)
        load_adapter(model,before);xm.mark_step();xm.wait_device_ops()
        restored=torch.load(file,map_location='cpu',weights_only=True)
        load_adapter(model,restored['adapter']);xm.mark_step();xm.wait_device_ops()
        for n,p in adapter_state(model).items():torch.testing.assert_close(p,after[n],rtol=0,atol=0)
        global_change=torch.tensor(max_change,device=device)
        dist.all_reduce(global_change,op=dist.ReduceOp.MAX);xm.mark_step();xm.wait_device_ops()
        result.update(checkpoint_id=str(folder.resolve()),reload_verified=True,adapter_max_abs_change=float(global_change.cpu()),
            local_changed_tensors=changed,optimizer_steps=1,microbatch=1,gradient_accumulation=len(records),
            update_s_including_any_compilation=elapsed,kl_policy_base=average_kl,losses=losses,
            hbm=xm.get_memory_info(device),base_gradients=0,optimizer_trainable_parameter_count=sum(p.numel() for p in adapters))
    else:raise ValueError('Unknown stage')
    Path(str(output)+f'.rank{rank}.json').write_text(json.dumps(result,indent=2)+'\n')
    if rank==0:Path(output).write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    try:execute(json.loads(Path(args.request).read_text()),args.output)
    except Exception:
        error=traceback.format_exc();print(error,flush=True)
        Path(str(args.output)+f'.failure_rank{os.environ.get("RANK","unknown")}.json').write_text(json.dumps({'error':error},indent=2))
        raise
