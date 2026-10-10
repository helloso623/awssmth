"""TTT-Discover entropic feedback and token objective, not PPO/GRPO.

KL centering follows upstream incorporate_kl_penalty. The loss follows
Tinker's importance_sampling contract: -sum(exp(logp-logp_sampler)*A).
No clipping, extra logp factor, sequence normalization or prompt loss.
"""
import math
import numpy as np
import torch
from upstream import author_advantages

class RewardGroup:
    def __init__(self, rewards): self.rewards = rewards
    def get_total_rewards(self): return self.rewards

def adaptive_beta(rewards, gamma=math.log(2), cap=1e6):
    r = np.asarray(rewards, dtype=np.float64)
    if r.ndim != 1 or len(r) < 2 or not np.isfinite(r).all():
        raise ValueError('Need at least two finite same-parent rewards')
    if gamma < 0 or cap <= 0: raise ValueError('Invalid beta search settings')
    maxima = np.count_nonzero(r == r.max())
    limit = math.log(len(r)/maxima)
    if np.ptp(r) == 0 or gamma == 0:
        return 0., 'equal_rewards' if np.ptp(r) == 0 else 'zero_target'
    def kl(b):
        logits = b*(r-r.max())
        logq = logits - np.logaddexp.reduce(logits)
        q = np.exp(logq)
        return float(np.sum(q*(logq+math.log(len(r)))))
    hi = 1.
    while hi < cap and kl(hi) < gamma: hi = min(2*hi, cap)
    if gamma >= limit or kl(hi) < gamma:
        return hi, 'unattainable_or_cap_limited'
    lo = 0.
    for _ in range(60):
        mid = (lo+hi)/2
        if kl(mid) < gamma: lo = mid
        else: hi = mid
    return hi, 'attained'

def entropic_advantages(rewards):
    beta, status = adaptive_beta(rewards)
    if status == 'attained':
        # Preserve the authors' float32 arithmetic on ordinary groups.
        advantage = author_advantages([RewardGroup(rewards)])[0]
    elif status == 'equal_rewards':
        advantage = torch.zeros(len(rewards), dtype=torch.float32)
    else:
        r = torch.tensor(rewards, dtype=torch.float32)
        z = torch.exp(beta*(r-r.max()))
        advantage = z/((z.sum()-z)/(len(r)-1)+1e-12)-1
    return advantage, dict(beta=beta, beta_status=status)

def token_advantages(entropic, sampler_logp, reference_logp, mask, kl_coef=.1):
    """Lists of aligned next-token vectors; prompts/padding/forced tokens mask=0."""
    if not (len(entropic) == len(mask) == len(sampler_logp) == len(reference_logp)):
        raise ValueError('Batch length mismatch')
    diffs = []
    for s, r, m in zip(sampler_logp, reference_logp, mask, strict=True):
        if s.shape != r.shape or s.shape != m.shape: raise ValueError('Token alignment mismatch')
        if not torch.all((m == 0) | (m == 1)): raise ValueError('Mask must be binary')
        diffs.append((s.detach()-r.detach())*m)
    count = sum(m.sum() for m in mask)
    if count.item() == 0: raise ValueError('No response tokens')
    average = sum(d.sum() for d in diffs)/count
    advantages = [(a*m + kl_coef*m*(average-d)).detach()
        for a, m, d in zip(entropic, mask, diffs, strict=True)]
    return advantages, float(average)

def importance_loss(learner_logp, sampler_logp, advantages, mask):
    # Fixed shapes are required on XLA. Guard before exponentiation rather
    # than boolean indexing (data-dependent output shape on Neuron).
    losses = []
    for p, s, a, m in zip(learner_logp, sampler_logp, advantages, mask, strict=True):
        if p.shape != s.shape or p.shape != a.shape or p.shape != m.shape:
            raise ValueError('Token alignment mismatch')
        active = m.bool()
        diff=torch.where(active,p-s.detach(),torch.zeros_like(p))
        advantage=torch.where(active,a.detach(),torch.zeros_like(a))
        losses.append(-(torch.exp(diff)*advantage).sum())
    return sum(losses)
