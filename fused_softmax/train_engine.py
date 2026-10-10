"""Local Neuron RL optimizer engine, awaiting a verified GPT-OSS model port.

This is NOT a GPT-OSS implementation. The missing model/backend integration
is explicit in local_preflight.py and TRAINING.md. No fallback is provided.
An installed port must expose an NxD model/optimizer, XLA next-token logprobs,
fixed-reference logprobs, sampling, synchronization, and adapter metadata.
"""
import hashlib
import math
import time
import torch
from objective import importance_loss, token_advantages

MODEL = 'openai/gpt-oss-20b'

def snapshot(port):
    port.synchronize()
    params = dict(port.adapter_parameters())
    if not params: raise RuntimeError('No trainable adapters')
    if any(p.device.type not in ('xla','neuron') for p in params.values()):
        raise RuntimeError('Trainium backend required; CPU/CUDA training refused')
    return {k:p.detach().cpu().clone() for k,p in params.items()}

def fingerprint(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(str((tensor.dtype,tuple(tensor.shape))).encode())
        h.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def optimizer_update(port, records, entropic, checkpoint_dir, tag):
    """Exactly one fresh-batch optimizer step; port must load sampler checkpoint.

    records contain the FULL token sequence with aligned next-token vectors
    (length sequence-1). Prompt, padding and forced-token positions mask=0.
    """
    if port.model_id != MODEL or port.lora_rank != 32:
        raise RuntimeError('Require GPT-OSS-20B with rank32 LoRA')
    versions = {r['checkpoint_id'] for r in records}
    if versions != {port.checkpoint_id}:
        raise RuntimeError('Mixed or stale sampler checkpoints')
    adapter_params = dict(port.adapter_parameters())
    optimizer_ids = {id(p) for g in port.optimizer.param_groups for p in g['params']}
    if optimizer_ids != {id(p) for p in adapter_params.values()}:
        raise RuntimeError('Optimizer must update exactly the verified adapter parameters')
    if any(p.requires_grad for p in port.base_parameters()):
        raise RuntimeError('Reference/base parameters must stay frozen')
    before = snapshot(port)
    start = time.perf_counter()
    sampler, reference, masks = [], [], []
    for record in records:
        length = len(record['tokens'])-1
        if length < 1 or any(len(record[k]) != length for k in ('sampling_logprobs','mask')):
            raise ValueError('Shifted next-token metadata alignment mismatch')
        masks.append(torch.tensor(record['mask'],dtype=torch.float32,device=port.device))
        sampler.append(torch.tensor(record['sampling_logprobs'],dtype=torch.float32,device=port.device))
        with torch.no_grad():
            reference.append(port.reference_logprobs(record['tokens']).detach())
    scalar = [float(a) for a in entropic]
    advantages, kl = token_advantages(scalar, sampler, reference, masks)
    port.optimizer.zero_grad(set_to_none=True)
    loss_value = 0.
    # Accumulate all response losses, then take one optimizer step.
    for record, s, a, mask in zip(records,sampler,advantages,masks,strict=True):
        learner = port.logprobs(record['tokens'])
        loss = importance_loss([learner],[s],[a],[mask])
        if not torch.isfinite(loss).item(): raise RuntimeError('Nonfinite RL loss')
        loss.backward()
        port.synchronize()
        loss_value += float(loss.detach().cpu())
    port.optimizer.step()
    port.synchronize()
    after = snapshot(port)
    local_max = max(float((after[k].float()-before[k].float()).abs().max()) for k in before)
    # For a sharded model, the port MUST aggregate this across all TP/EP ranks.
    max_change = float(port.global_max(local_max))
    if not math.isfinite(max_change) or max_change <= 0:
        raise RuntimeError('No finite adapter change; update proof failed')
    from neuronx_distributed.trainer import save_checkpoint, load_checkpoint
    metadata = dict(port.checkpoint_manifest(), model_id=MODEL,
                    parent_checkpoint=port.checkpoint_id, optimizer_steps=1)
    save_checkpoint(str(checkpoint_dir),tag,model=port.model,optimizer=port.optimizer,
                    user_content=metadata,async_save=False)
    port.synchronize()
    # Successful load AND exact local adapter equality, not merely file existence.
    restored = load_checkpoint(str(checkpoint_dir),tag=tag,model=port.model,
                               optimizer=port.optimizer,strict=True)
    port.synchronize()
    reloaded = snapshot(port)
    if any(not torch.equal(after[k],reloaded[k]) for k in after):
        raise RuntimeError('Checkpoint adapter reload mismatch')
    if restored != metadata: raise RuntimeError('Checkpoint manifest reload mismatch')
    port.checkpoint_id = str(checkpoint_dir)+'/'+tag
    return dict(checkpoint_id=port.checkpoint_id,elapsed_s=time.perf_counter()-start,
                local_before_sha256=fingerprint(before),local_after_sha256=fingerprint(after),
                adapter_max_abs_change=max_change,reload_verified=True,
                kl_policy_base=kl,loss=loss_value,optimizer_steps=1,
                reference_checkpoint=port.reference_checkpoint_id)
