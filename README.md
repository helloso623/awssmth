# GPT-OSS-20B Trainium2 TTT-Discover work-in-progress backup

Code-only recovery snapshot. No pretrained weights, trained adapters, credentials,
virtual environments, compiler caches, or compiler binaries are included.

## Verified milestones and current limits

- Fused NKI softmax passes the fixed CPU PyTorch checker on Trainium2.
- Deterministic expert forward/backward parity passes, including routing and LoRA gradients.
- Four-core tensor-parallel expert optimizer updates pass on Trainium.
- A small two-layer model completes a LoRA update, exact adapter reload, and next-token generation.
- The actual pretrained GPT-OSS-20B (20,914,757,184 parameters) loads TP=4;
  its reference forward compiles and executes on all four ranks.
- **At snapshot time, the full pretrained backward graph is still compiling. A full-model
  optimizer update and the complete online RL experiment have not been demonstrated.**
- `neuron_backend.py` is the new direct training/generation implementation. Its integrated
  generation/update path and checkpointing fallback still need device validation.
  `controller.py` and older backend files are retained for recovery; their integration
  with the new backend remains unfinished. Do not treat them as a validated end-to-end run.

## Hardware and coordination

The original allocation is seat-57, logical cores 0–3 under LNC2, 24 GiB HBM per
logical core. TP=4 training and cores 0–1 evaluation run **sequentially**. Read the
shared `/workspace/STATUS-evaluator.md` and `/workspace/STATUS-training.md` before
device use. Use only your confirmed allocation; no server is stopped by these scripts.

## Restore environment and pretrained files

Requires an existing Trainium2 pod with compatible Neuron driver/runtime, `uv`,
and network access. Run from `fused_softmax`:

```bash
bash bootstrap_training.sh
source .venv-neuron/bin/activate
export LD_LIBRARY_PATH="$PWD/.python/cpython-3.12.14-linux-x86_64-gnu/lib:${LD_LIBRARY_PATH:-}"
python download_pretrained.py
```

The lock captures the tested Python 3.12 / Torch 2.9 / Torch-XLA 2.9 / NxD 0.19
stack. Transformers 4.56.2 and islpy 2026.1 are intentional compatibility pins.
Weights are downloaded separately from pinned Hugging Face revision
`6cee5e81ee83917806bbde320786a8fb61efebee`; downloading does not load the model.

## Validation and full-model attempt

Only run device commands after evaluator release:

```bash
python -m unittest test_training_components test_objective
bash run_trainium.sh validate_tp4.py
bash run_trainium.sh train_gptoss.py --small --sequence 8 --output small_model_step
bash run_trainium.sh train_gptoss.py \
  --checkpoint "$(cat pretrained-path.txt)" --sequence 16 --output pretrained_model_step
# Prepared fallback, pending validation at snapshot time:
bash run_trainium.sh validate_checkpointing.py
```

The sequence-16 run is an execution test, not full kernel generation. First-step
elapsed times include compilation. Preserve emitted evidence and adapters locally.
An initial-policy snapshot must remain fixed; disabling adapters was not numerically
equivalent on the tested BF16 graph, so the backend restores initial adapter values
when computing reference log-probabilities, without a second full base model.

## Reproduce two-core softmax measurement

Use the evaluator's working NKI Python environment (independent from the training
environment). It must expose native `nki` 0.6 and the matching Neuron compiler/runtime.
From `fused_softmax`, with all training processes released:

```bash
NEURON_RT_VISIBLE_CORES=0,1 NEURON_LOGICAL_NC_CONFIG=2 python evaluate.py \
  --source baseline.py --output baseline-new.json --iterations 10 --seed 911
NEURON_RT_VISIBLE_CORES=0,1 NEURON_LOGICAL_NC_CONFIG=2 python evaluate.py \
  --source experiments/fused_reuse_smoke.py --output fused-new.json \
  --baseline baseline-new.json --iterations 10 --seed 911
```

Preserved results in `experiments/`: baseline / candidate latencies in microseconds
are 200.1319 / 196.9001 (128×512), 229.3979 / 224.8756 (129×513), and
165.3221 / 161.0296 (257×61). Geometric-mean speedup is approximately 1.021×;
this small smoke-test difference is not a statistically established win.
Timing is synchronous invocation wall time including transfers, allocation, and
runtime overhead; compilation is excluded. Twenty warmups precede each of three
repeated means. Correctness covers 64 checks including hostile inputs.

## Method provenance and departures

Authors' source is vendored without modification from
`test-time-training/discover` commit `6c40e82dab9d5de7416ac873ad5cd3106084aaed`;
its license is included. `upstream.py` extracts the actual entropic estimator.
The archive uses pinned authors' PUCT, including its seed-excluded reward scale
with a 1e-6 floor and first-child parent maximum initialization; these are pinned-code
details differing from a literal reading of Appendix A.2, not new archive adjustments.
The token objective retains unclipped sampler/learner importance correction and
the authors' token-level centered reference KL term; prompt/padding tokens are masked.

Adaptations: GPT-OSS-20B instead of 120B; Trainium NKI softmax; configurable smaller
rollout groups; correctness-gated geometric-mean baseline/candidate speedup rather
than literal inverse-runtime reward. Local rank-32 AWS parallel-linear LoRA plus
a validated packed-expert extension uses alpha 32. Tinker's internal placement and
scaling are opaque, so this is **not claimed to match its undocumented adapter defaults**.
The static expert MVP computes all experts in chunks, rather than an optimized
sparse dispatched kernel. Complete NKI source generation remains the final goal.

`SOURCE-MANIFEST.json` records original snapshot file hashes. Benchmark evidence is
preserved, but weights must be downloaded and adapter training resumed from separately
retained checkpoints; a code-only backup cannot recover unsaved optimizer state.
