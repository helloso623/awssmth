"""Development Trainium port of GPT-OSS's packed expert projections.

Static shapes avoid the HF training path's data-dependent nonzero/where.
Uses differentiable Torch operations and NxD TP collective callbacks, not
private blockwise NKI backward kernels. Gate/up layout and clamped SwiGLU
follow HF GPT-OSS 4.56 and AWS's all_expert_mlps_bf16 reference.

This component does NOT establish complete model training support. The
per-expert LoRA parameterization below is explicit; equivalence to Tinker's
internal expert adapter placement/scaling has NOT been established. Do not
activate it as an exact authors-configuration reproduction.
"""
import torch
from torch import nn

def identity(x): return x

class StaticTopKRouter(nn.Module):
    """Top-k choices with an explicit differentiable selected-logit path.

    Avoids reliance on topk-value/scatter_ autograd on Neuron. Indices are
    nondifferentiable, as in the original router. Selector storage is only
    [tokens,top_k,experts], never [tokens,experts,intermediate].
    """
    def __init__(self, original):
        super().__init__()
        self.top_k,self.num_experts,self.hidden_dim = original.top_k,original.num_experts,original.hidden_dim
        self.weight=nn.Parameter(original.weight.detach().clone(),requires_grad=original.weight.requires_grad)
        self.bias=nn.Parameter(original.bias.detach().clone(),requires_grad=original.bias.requires_grad)

    def forward(self, hidden_states):
        x=hidden_states.reshape(-1,self.hidden_dim)
        logits=torch.nn.functional.linear(x,self.weight,self.bias)
        with torch.no_grad(): _,indices=torch.topk(logits,self.top_k,dim=-1)
        expert_ids=torch.arange(self.num_experts,device=logits.device)
        selectors=(indices.unsqueeze(-1)==expert_ids).to(logits.dtype)
        selected=(logits.unsqueeze(1)*selectors).sum(-1)
        weights=torch.softmax(selected,dim=1)
        affinities=(weights.unsqueeze(-1)*selectors).sum(1)
        return affinities,indices

class StaticExperts(nn.Module):
    def __init__(self, experts, hidden, intermediate, *, rank=32, alpha=None,
                 tp_rank=0, tp_size=1, dtype=torch.bfloat16, device=None,
                 copy_to_tp=identity, reduce_from_tp=identity, expert_chunk_size=1,
                 checkpoint_fn=None, adapter_dtype=None):
        super().__init__()
        if alpha is None:
            raise ValueError('LoRA scaling must be explicitly verified/configured; no undocumented default')
        if rank != 32 or intermediate % tp_size or not 0 <= tp_rank < tp_size:
            raise ValueError('Require rank32 and a valid intermediate TP partition')
        self.hidden, self.intermediate, self.experts = hidden,intermediate,experts
        self.tp_rank,self.tp_size,self.rank = tp_rank,tp_size,rank
        self.scale = float(alpha)/rank
        self.copy_to_tp,self.reduce_from_tp = copy_to_tp,reduce_from_tp
        self.adapters_enabled = True
        if expert_chunk_size < 1: raise ValueError('Expert chunk must be positive')
        self.expert_chunk_size = expert_chunk_size
        self.checkpoint_fn = checkpoint_fn
        local = intermediate//tp_size
        options = dict(dtype=dtype,device=device)
        def base(shape): return nn.Parameter(torch.empty(shape,**options),requires_grad=False)
        self.gate_up_proj = base((experts,hidden,2*local))
        self.gate_up_proj_bias = base((experts,2*local))
        self.down_proj = base((experts,local,hidden))
        self.down_proj_bias = base((experts,hidden))
        options = dict(dtype=adapter_dtype or dtype,device=device)
        # Column adapter: A replicated; B sharded over paired gate/up outputs.
        self.gate_lora_A = nn.Parameter(torch.empty(experts,hidden,rank,**options))
        self.gate_lora_B = nn.Parameter(torch.zeros(experts,rank,2*local,**options))
        # Row adapter: A sharded over intermediate; B replicated over hidden.
        self.down_lora_A = nn.Parameter(torch.empty(experts,local,rank,**options))
        self.down_lora_B = nn.Parameter(torch.zeros(experts,rank,hidden,**options))
        # Zero B gives an unchanged initial policy; initialization/scaling is
        # a port setting, not a claim about undocumented backend defaults.
        nn.init.normal_(self.gate_lora_A,std=hidden**-.5)
        nn.init.normal_(self.down_lora_A,std=intermediate**-.5)

    def load_base(self, hf_experts):
        local = self.intermediate//self.tp_size
        start,end = self.tp_rank*local,(self.tp_rank+1)*local
        with torch.no_grad():
            self.gate_up_proj.copy_(hf_experts.gate_up_proj[...,2*start:2*end])
            self.gate_up_proj_bias.copy_(hf_experts.gate_up_proj_bias[...,2*start:2*end])
            self.down_proj.copy_(hf_experts.down_proj[:,start:end,:])
            self.down_proj_bias.copy_(hf_experts.down_proj_bias)

    def _chunk(self, x, affinities, start, end):
        gate_up = torch.einsum('th,ehi->tei',self.copy_to_tp(x),self.gate_up_proj[start:end])
        gate_up = gate_up+self.gate_up_proj_bias[start:end]
        if self.adapters_enabled:
            low = torch.einsum('th,ehr->ter',x.to(self.gate_lora_A.dtype),self.gate_lora_A[start:end])
            gate_up = gate_up+(self.scale*torch.einsum('ter,eri->tei',self.copy_to_tp(low),self.gate_lora_B[start:end])).to(gate_up.dtype)
        gate = gate_up[...,::2].clamp(max=7.)
        up = gate_up[...,1::2].clamp(min=-7.,max=7.)
        act = gate*torch.sigmoid(1.702*gate)*(up+1.)
        down = torch.einsum('tei,eih->teh',act,self.down_proj[start:end])
        # The router input is replicated, but this branch has partial expert
        # outputs. Sum its affinity gradient across TP ranks, unlike bias and
        # the already-reduced down-adapter branch.
        local_out = torch.einsum('teh,te->th',down,self.copy_to_tp(affinities))
        result = self.reduce_from_tp(local_out)
        result = result+torch.einsum('eh,te->th',self.down_proj_bias[start:end],affinities)
        if self.adapters_enabled:
            low = self.reduce_from_tp(torch.einsum('tei,eir->ter',act.to(self.down_lora_A.dtype),self.down_lora_A[start:end]))
            adapter_out = torch.einsum('ter,erh->teh',low,self.down_lora_B[start:end])
            result = result+(self.scale*torch.einsum('teh,te->th',adapter_out,affinities.to(adapter_out.dtype))).to(result.dtype)
        return result

    def forward(self, hidden_states, router_indices=None, routing_weights=None):
        shape = hidden_states.shape
        x = hidden_states.reshape(-1,self.hidden)
        if routing_weights.shape != (x.shape[0],self.experts):
            raise ValueError('Expected HF4.56 dense expert affinity matrix')
        result = torch.zeros_like(x)
        # Compile-time slices reduce live expert intermediates. Backward saved
        # activations still need a verified checkpoint implementation on XLA.
        for start in range(0,self.experts,self.expert_chunk_size):
            end = min(start+self.expert_chunk_size,self.experts)
            def compute(h,a,lo=start,hi=end): return self._chunk(h,a,lo,hi)
            affinities = routing_weights[:,start:end]
            out = (self.checkpoint_fn(compute,x,affinities) if self.checkpoint_fn is not None
                   and torch.is_grad_enabled() else compute(x,affinities))
            result = result+out
        return result.reshape(shape)

def install_expert_components(model, *, alpha, tp_rank=0,tp_size=1):
    """Developer hook only. Caller must separately port attention, head and I/O.

    Does not silently choose an adapter policy or initialize distributed state.
    """
    if tp_size > 1:
        from neuronx_distributed.parallel_layers.mappings import (
            copy_to_tensor_model_parallel_region,reduce_from_tensor_model_parallel_region)
        callbacks = dict(copy_to_tp=copy_to_tensor_model_parallel_region,
                         reduce_from_tp=reduce_from_tensor_model_parallel_region)
    else: callbacks = {}
    for layer in model.model.layers:
        original = layer.mlp.experts
        component = StaticExperts(original.num_experts,original.hidden_size,
            original.intermediate_size,alpha=alpha,tp_rank=tp_rank,tp_size=tp_size,
            dtype=original.gate_up_proj.dtype,device=original.gate_up_proj.device,**callbacks)
        component.load_base(original)
        layer.mlp.experts = component
    return model
