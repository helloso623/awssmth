"""Explicit per-rank TP memory plan. Estimates are not measured peaks."""
import json
from pathlib import Path

def budget(tp=4, sequence=128):
    c=json.loads(Path('gpt_oss_metadata/config.json').read_text())
    h,i,e,l,r,v=c['hidden_size'],c['intermediate_size'],c['num_local_experts'],c['num_hidden_layers'],32,c['vocab_size']
    q=c['num_attention_heads']*c['head_dim'];kv=c['num_key_value_heads']*c['head_dim']
    # Exact frozen base count with expert/dense/embedding/head projection shards;
    # down and o biases, router, norms are replicated. Sink is head-sharded.
    expert=e*(h*2*i//tp+2*i//tp+i*h//tp+h)
    attention=h*(q+2*kv)//tp+(q+2*kv)//tp+q*h//tp+h+c['num_attention_heads']//tp
    replicated=e*h+e+2*h
    base=l*(expert+attention+replicated)+2*v*h//tp+h
    router_fp32_extra=l*(e*h+e)*2
    expert_adapter=l*e*(h*r+r*2*i//tp+i*r//tp+r*h)
    attention_adapter=l*(3*h*r+r*(q+2*kv)//tp+q*r//tp+r*h)
    head_adapter=h*r+r*v//tp
    adapter=expert_adapter+attention_adapter+head_adapter
    # FP32 adapters and gradients plus two FP32 Adam states (no extra master).
    state=adapter*16
    kv_cache=l*2*sequence*(kv//tp)*2
    # Conservative provisional reserves; must replace with measurements.
    reserves=dict(activations=512*2**20,temporaries=1024*2**20,
                  collective_buffers=256*2**20,runtime=256*2**20)
    return dict(tp=tp,sequence=sequence,microbatch=1,base_parameters_per_core=base,
      frozen_base_bytes_per_core=base*2+router_fp32_extra,router_fp32_extra_bytes=router_fp32_extra,adapter_parameters_per_core=adapter,
      fp32_adapter_gradient_adam_bytes_per_core=state,kv_cache_bytes_per_core=kv_cache,
      reserves_bytes_per_core=reserves,estimated_peak_bytes_per_core=base*2+router_fp32_extra+state+kv_cache+sum(reserves.values()),
      usable_hbm_bytes_per_core=25769803776,
      reference_policy='Sequential restore of immutable initial adapter checkpoint; same enabled-adapter execution path; CPU checkpoint storage, no duplicate base',
      loading='CPU dequantization one expert shard at a time; no full BF16 host model or device dequant copies',
      peak_status='Planning estimate, not measured; compiler scratch and checkpoint overlap remain to be measured',
      placement=dict(expert='gate paired outputs/down intermediate sharded',attention='Q/K/V outputs and O inputs sharded; O bias replicated',
       embedding='vocabulary sharded',head='vocabulary sharded',router='replicated',norms='replicated',
       adapters='column A replicated/B output-sharded; row A input-sharded/B replicated'))

if __name__=='__main__':
    result={str(tp):budget(tp) for tp in (2,4)}
    Path('memory_budget.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
