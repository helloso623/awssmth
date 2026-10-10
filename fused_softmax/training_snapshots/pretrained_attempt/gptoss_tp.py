"""Executable GPT-OSS HF/NxD TP port. No inference endpoint dependency.

Dense LoRA uses AWS LoraParallelLinear; packed experts use our parity-tested
extension. Alpha is explicit: Tinker's internal scaling/placement is opaque.
"""
import json, math
from pathlib import Path
import torch
from torch import nn

def build_model(cfg, rank, tp=4, alpha=32):
    from transformers import GptOssForCausalLM
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding
    from neuronx_distributed.parallel_layers.layers import ParallelEmbedding,ColumnParallelLinear,RowParallelLinear
    from neuronx_distributed.modules.lora.config import LoraConfig
    from neuronx_distributed.modules.lora.tp_layer import LoraParallelLinear
    from neuronx_distributed.parallel_layers.mappings import copy_to_tensor_model_parallel_region,reduce_from_tensor_model_parallel_region
    from gpt_oss_experts import StaticExperts,StaticTopKRouter
    cfg._attn_implementation='eager'
    with torch.device('meta'): model=GptOssForCausalLM(cfg)
    lc=LoraConfig(lora_rank=32,lora_alpha=alpha,lora_dropout=0,bias='none')
    def linear(original,row=False):
        cls=RowParallelLinear if row else ColumnParallelLinear
        opts=dict(input_is_parallel=True) if row else dict(gather_output=False)
        base=cls(original.in_features,original.out_features,bias=original.bias is not None,
                 dtype=torch.bfloat16,device=torch.device('cpu'),init_method=lambda t:t,**opts)
        return LoraParallelLinear(base,lc)
    model.model.embed_tokens=ParallelEmbedding(cfg.vocab_size,cfg.hidden_size,dtype=torch.bfloat16,
        device=torch.device('cpu'),init_method=lambda t:t)
    for layer in model.model.layers:
        for name in ('q_proj','k_proj','v_proj','o_proj'):
            setattr(layer.self_attn,name,linear(getattr(layer.self_attn,name),row=name=='o_proj'))
        layer.self_attn.sinks=nn.Parameter(torch.empty(cfg.num_attention_heads//tp,dtype=torch.bfloat16),requires_grad=False)
        router=layer.mlp.router
        # Replace meta parameters before the static router constructor clones.
        router.to_empty(device='cpu')
        layer.mlp.router=StaticTopKRouter(router)
        layer.mlp.experts=StaticExperts(cfg.num_local_experts,cfg.hidden_size,cfg.intermediate_size,
            alpha=alpha,tp_rank=rank,tp_size=tp,dtype=torch.bfloat16,adapter_dtype=torch.float32,
            copy_to_tp=copy_to_tensor_model_parallel_region,reduce_from_tp=reduce_from_tensor_model_parallel_region,
            expert_chunk_size=4)
    model.lm_head=linear(model.lm_head)
    # Only remaining meta parameters are replicated normalization vectors.
    for module in model.modules():
        for name,p in list(module.named_parameters(recurse=False)):
            if p.device.type=='meta':
                setattr(module,name,nn.Parameter(torch.empty(p.shape,dtype=torch.bfloat16),requires_grad=False))
    model.model.rotary_emb=GptOssRotaryEmbedding(cfg,device='cpu')
    for name,p in model.named_parameters(): p.requires_grad_('lora_' in name)
    return model

class CheckpointReader:
    def __init__(self,path):
        from safetensors import safe_open
        self.path=Path(path)
        index=json.loads((self.path/'model.safetensors.index.json').read_text())['weight_map']
        self.index=index
        self.handles={f:safe_open(str(self.path/f),framework='pt',device='cpu') for f in set(index.values())}
    def tensor(self,key,selection=Ellipsis):
        return self.handles[self.index[key]].get_slice(key)[selection]

def load_pretrained(model,path,rank,tp=4):
    """Shard packed checkpoint BEFORE dequantization; never build full BF16 copy."""
    from neuronx_distributed.quantization.microscaling.transform_weights import get_mxfp4_tensor
    reader=CheckpointReader(path)
    placement={}
    with torch.no_grad():
        for name,p in model.named_parameters():
            if 'lora_' in name: continue
            key=name.replace('.base_layer.','.')
            selection=Ellipsis
            if '.experts.' in name and name.endswith(('.gate_up_proj','.down_proj')):
                if key in reader.index:
                    selection=(slice(None),slice(None),slice(rank*p.shape[2],(rank+1)*p.shape[2])) if name.endswith('gate_up_proj') else (slice(None),slice(rank*p.shape[1],(rank+1)*p.shape[1]),slice(None))
                    p.copy_(reader.tensor(key,selection))
                    placement[name]=dict(shape=list(p.shape),dtype=str(p.dtype),replicated=False)
                    continue
                # HF packed tensors [expert,output,input/32,16]; transpose to
                # [expert,input,output]. Gate outputs paired; down input grouped32.
                blocks=key+'_blocks';scales=key+'_scales'
                local=p.shape[-1] if name.endswith('gate_up_proj') else p.shape[1]
                pieces=[]
                for expert in range(p.shape[0]):
                    if name.endswith('gate_up_proj'):
                        sel=(slice(expert,expert+1),slice(rank*local,(rank+1)*local),slice(None),slice(None))
                    else:
                        # TP4 splits I=2880 into720, which crosses MXFP4 blocks.
                        # Decode overlapping boundary blocks then trim16 values.
                        sel=(slice(expert,expert+1),slice(None),slice(rank*local//32,math.ceil((rank+1)*local/32)),slice(None))
                    b=reader.tensor(blocks,sel)
                    s=reader.tensor(scales,sel[:-1])
                    dequant=get_mxfp4_tensor(b,s,rows_per_chunk=8192).transpose(1,2)
                    if name.endswith('down_proj'):
                        offset=(rank*local)%32
                        dequant=dequant[:,offset:offset+local,:]
                    p[expert:expert+1].copy_(dequant)
                placement[name]=dict(shape=list(p.shape),dtype=str(p.dtype),replicated=False)
                continue
            if key in ('model.embed_tokens.weight','lm_head.weight') or any(f'.{x}.' in key for x in ('q_proj','k_proj','v_proj')):
                selection=(slice(rank*p.shape[0],(rank+1)*p.shape[0]),)+(slice(None),)*(p.ndim-1)
            elif '.o_proj.weight' in key:
                selection=(slice(None),slice(rank*p.shape[1],(rank+1)*p.shape[1]))
            elif key.endswith('.sinks') or key.endswith('.gate_up_proj_bias'):
                selection=(slice(rank*p.shape[0],(rank+1)*p.shape[0]),) if p.ndim==1 else (slice(None),slice(rank*p.shape[1],(rank+1)*p.shape[1]))
            p.copy_(reader.tensor(key,selection).to(p.dtype))
            placement[name]=dict(shape=list(p.shape),dtype=str(p.dtype),replicated=selection is Ellipsis)
    return placement

def hidden(model,tokens):
    """Static masks bypass dynamic HF mask construction; no kernel truncation."""
    seq=tokens.shape[1]
    ar=torch.arange(seq,device=tokens.device)
    delta=ar[:,None]-ar[None,:]
    def mask(keep): return torch.where(keep,0.,torch.finfo(torch.bfloat16).min).to(torch.bfloat16)[None,None,:,:]
    masks={'full_attention':mask(delta>=0),'sliding_attention':mask((delta>=0)&(delta<model.config.sliding_window))}
    return model.model(tokens,attention_mask=masks,use_cache=False).last_hidden_state

def log_probs(model,tokens,chunk_size=None,checkpoint_head=False,sync_chunks=False):
    from neuronx_distributed.parallel_layers.loss_functions import parallel_cross_entropy
    states=hidden(model,tokens)
    targets=tokens[:,1:]
    if chunk_size is None:
        return -parallel_cross_entropy(model.lm_head(states)[:,:-1,:].float(),targets)
    states=states[:,:-1,:]
    from torch_xla.utils.checkpoint import checkpoint
    import torch_xla.core.xla_model as xm
    def selected(h,y):
        result=-parallel_cross_entropy(model.lm_head(h).float(),y)
        if sync_chunks: xm.mark_step()
        return result
    pieces=[]
    for start in range(0,states.shape[1],chunk_size):
        h=states[:,start:start+chunk_size,:]
        y=targets[:,start:start+chunk_size]
        pieces.append(checkpoint(selected,h,y,use_reentrant=True,preserve_rng_state=False)
            if checkpoint_head and torch.is_grad_enabled() else selected(h,y))
    return torch.cat(pieces,dim=1)

def adapters_enabled(model,enabled):
    from gpt_oss_experts import StaticExperts
    from neuronx_distributed.modules.lora.tp_layer import LoraParallelLinear
    for module in model.modules():
        if isinstance(module,StaticExperts): module.adapters_enabled=enabled
        if isinstance(module,LoraParallelLinear): module.merged=not enabled

def adapter_state(model):
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}

def load_adapter(model,state):
    with torch.no_grad():
        for n,p in model.named_parameters():
            if p.requires_grad: p.copy_(state[n].to(p.device))

def enable_activation_checkpointing(model,segment_graph=False):
    """Use XLA's supported reentrant checkpoint path, not unsupported False."""
    from neuronx_distributed.utils.activation_checkpoint import checkpoint_wrapper
    model.enable_input_require_grads()
    for index,layer in enumerate(model.model.layers):
        wrapped=checkpoint_wrapper(layer,preserve_rng_state=True)
        if segment_graph:
            original=wrapped.forward
            def forward(*args,_original=original,**kwargs):
                result=_original(*args,**kwargs)
                import torch_xla.core.xla_model as xm
                xm.mark_step()
                return result
            wrapped.forward=forward
        model.model.layers[index]=wrapped

class FixedKVCache:
    """All layers retain fixed-size cache; explicit masks implement sliding window.

    Fixed cache shapes avoid a new decode graph for every response token.
    This does not change the model's sliding/full attention semantics.
    """
    def __init__(self,config,capacity,device,tp=4):
        self.capacity=capacity
        shape=(1,config.num_key_value_heads//tp,capacity,config.head_dim)
        self.keys=[torch.zeros(shape,device=device,dtype=torch.bfloat16) for _ in range(config.num_hidden_layers)]
        self.values=[torch.zeros_like(k) for k in self.keys]
    def update(self,key,value,layer_idx,cache_kwargs):
        pos=cache_kwargs['cache_position']
        # Router accumulation can make later layers FP32; preserve the actual
        # execution dtype rather than silently quantizing their KV tensors.
        if self.keys[layer_idx].dtype != key.dtype:
            self.keys[layer_idx]=self.keys[layer_idx].to(key.dtype)
            self.values[layer_idx]=self.values[layer_idx].to(value.dtype)
        self.keys[layer_idx]=torch.index_copy(self.keys[layer_idx],2,pos,key)
        self.values[layer_idx]=torch.index_copy(self.values[layer_idx],2,pos,value)
        return self.keys[layer_idx],self.values[layer_idx]

def cached_hidden(model,tokens,positions,cache):
    keys=torch.arange(cache.capacity,device=tokens.device)
    delta=positions[:,None]-keys[None,:]
    def mask(keep): return torch.where(keep,0.,torch.finfo(torch.bfloat16).min).to(torch.bfloat16)[None,None,:,:]
    masks={'full_attention':mask(delta>=0),'sliding_attention':mask((delta>=0)&(delta<model.config.sliding_window))}
    return model.model(tokens,position_ids=positions[None,:],cache_position=positions,
        attention_mask=masks,past_key_values=cache,use_cache=True).last_hidden_state
