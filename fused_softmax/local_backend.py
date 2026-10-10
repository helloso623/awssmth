"""One-stage local backend process. Exit releases this stage's device resources.

No verified GPT-OSS Neuron model port currently exists in this workspace.
--port names a future verified model implementation, not a generic HF import.
Its constructor initializes supported NxD parallelism, rank32 adapter placement,
Adam(4e-5,betas=(.9,.95),eps=1e-8), tokenizer and checkpoint loading.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import traceback
from local_preflight import inspect

def execute(request, port_module=None):
    report = inspect()
    if not port_module:
        raise RuntimeError('Missing verified GPT-OSS Neuron training model port; '+
                           '; '.join(report['blockers']))
    missing = [n for n in ('torch-neuronx','neuronx-distributed') if report['packages'][n] is None]
    if missing: raise RuntimeError('Missing training packages: '+', '.join(missing))
    if os.environ.get('NEURON_RT_VISIBLE_CORES') != '0,1' or os.environ.get('NEURON_LOGICAL_NC_CONFIG') != '2':
        raise RuntimeError('Local backend must use allocated cores0–1 with LNC2')
    if not request.get('allocation_resolved'):
        raise RuntimeError('Organizer has not resolved cores0–1 allocation conflict')
    module = importlib.import_module(port_module)
    port = module.create_port(request)
    if port.model_id != 'openai/gpt-oss-20b' or port.lora_rank != 32:
        raise RuntimeError('Unsupported policy or LoRA rank')
    if not port.adapter_placement_verified:
        raise RuntimeError('Adapter placement must be documented and validated, including experts')
    if request['action'] == 'generate':
        results = []
        for group in request['groups']:
            for _ in range(request['responses']):
                record = port.sample(group['prompt'],request['generation'])
                required = ('tokens','sampling_logprobs','mask','response','prompt_tokens')
                if any(k not in record for k in required): raise RuntimeError('Incomplete rollout token metadata')
                record.update(parent_id=group['parent_id'],checkpoint_id=port.checkpoint_id,
                              reference_checkpoint=port.reference_checkpoint_id,
                              generation=request['generation'])
                results.append(record)
        return dict(records=results,checkpoint_id=port.checkpoint_id)
    if request['action'] == 'update':
        from train_engine import optimizer_update
        return optimizer_update(port,request['records'],request['advantages'],
                                request['checkpoint_dir'],request['tag'])
    raise ValueError('Unknown backend action')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--request',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--port')
    a = p.parse_args()
    try:
        result = execute(json.loads(Path(a.request).read_text()),a.port)
    except Exception:
        Path(a.output).write_text(json.dumps(dict(ok=False,failure=traceback.format_exc()),indent=2))
        raise
    Path(a.output).write_text(json.dumps(dict(ok=True,**result),indent=2))
