"""Read-only local training checks. Never loads a model or probes other cores."""
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path

ROOT = Path(__file__).parent
MODEL = 'openai/gpt-oss-20b'

def inspect():
    packages = {}
    for name in ('torch','torch-xla','torch-neuronx','libtorch-neuronx-lite',
                 'neuronx-distributed','neuronx-distributed-training','transformers'):
        try: packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError: packages[name] = None
    limits = {name:(Path('/sys/fs/cgroup')/name).read_text().strip()
              for name in ('memory.max','memory.current','cpu.max','cpuset.cpus.effective')}
    blockers = []
    for name in ('torch-neuronx', 'neuronx-distributed'):
        if packages[name] is None: blockers.append(f'Missing training dependency: {name}')
    blockers.append('GPT-OSS Trainium training model port and packed-expert LoRA implementation not verified; generic NxD LoRA cannot wrap GptOssExperts')
    blockers.append('Assigned cores 0–1 last launch failed busy (ret=-16); organizer allocation resolution required')
    return dict(model=MODEL, requested_lora_rank=32, packages=packages,
                pod_limits=limits, devices=[str(p) for p in Path('/dev').glob('neuron[0-9]*')],
                requested_cores=[0,1], requested_lnc=2, blockers=blockers,
                training_supported=False, weights_loaded=False,
                adapter_placement='unverified: do not reduce to head-only or attention-only',
                memory_fit='unverified: need BF16 shard sizes, activations, adapters, optimizer, compilation peak; host limit is not HBM capacity')

def require_training_support():
    report = inspect()
    raise RuntimeError('Local GPT-OSS training is blocked:\n'+'\n'.join(report['blockers']))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output')
    args = p.parse_args()
    report = inspect()
    rendered = json.dumps(report,indent=2)+'\n'
    if args.output: Path(args.output).write_text(rendered)
    print(rendered)
