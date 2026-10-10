"""Resumable full-source TTT-Discover control plane.

Device stages are separate subprocesses and run sequentially. The actual
GPT-OSS Neuron model port is a missing dependency, never replaced by CPU or
an inference endpoint. Frozen mode is an explicit matched-budget comparison.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from archive import open_archive, kernel_state
from objective import entropic_advantages
from source_policy import extract_source, prompt

ROOT = Path(__file__).parent

def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)

def stage(command, request, output, timeout):
    """No shell; terminate only the child process group we created on timeout."""
    output = Path(output)
    if output.exists():
        cached = json.loads(output.read_text())
        if cached.get('ok'): return cached
    input_path = output.with_suffix('.request.json')
    atomic_json(input_path, request)
    env = dict(os.environ,NEURON_RT_VISIBLE_CORES='0,1',NEURON_LOGICAL_NC_CONFIG='2')
    with output.with_suffix('.log').open('a') as log:
        proc = subprocess.Popen(command+['--request',str(input_path),'--output',str(output)],
            stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try: proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL)
            proc.wait()
            raise RuntimeError(f'Stage timed out; see {output.with_suffix(".log")}')
    if proc.returncode or not output.exists(): raise RuntimeError(f'Stage failed; see {output.with_suffix(".log")}')
    result = json.loads(output.read_text())
    if not result.get('ok'): raise RuntimeError(result.get('failure','Stage failed'))
    return result

def reward(result, baseline):
    if not result.get('ok'): return 0.
    if result.get('timing_version') != baseline.get('timing_version') or result.get('cores') != [0,1]:
        raise ValueError('Evaluator timing/allocation contract mismatch')
    ratios = []
    for b,c in zip(baseline['timings'],result['timings'],strict=True):
        if b['shape'] != c['shape']: raise ValueError('Benchmark case mismatch')
        bt,ct = float(b['latency_us']),float(c['latency_us'])
        if not (math.isfinite(bt) and math.isfinite(ct) and bt>0 and ct>0):
            raise ValueError('Invalid Trainium timing')
        ratios.append(bt/ct)
    if not ratios or not result.get('checks') or not all(c['ok'] for c in result['checks']):
        raise ValueError('Missing passing correctness evidence')
    return math.exp(sum(map(math.log,ratios))/len(ratios))

def run(config):
    if config['responses'] < 4: raise ValueError('At least four responses per same-parent group')
    if config['model'] != 'openai/gpt-oss-20b': raise ValueError('GPT-OSS-20B required')
    root = Path(config['run_dir']).resolve()
    root.mkdir(parents=True,exist_ok=True)
    manifest = root/'state.json'
    baseline_path = root/'baseline.fixed.json'
    baseline = json.loads(Path(config['baseline']).read_text())
    if not baseline.get('ok') or baseline.get('cores') != [0,1] or baseline.get('timing_version') != 2:
        raise ValueError('Need a verified fresh baseline on assigned cores with current timing method')
    if baseline_path.exists():
        if json.loads(baseline_path.read_text()) != baseline: raise ValueError('Experiment baseline changed')
    else: atomic_json(baseline_path,baseline)
    seed = kernel_state(Path(config['seed_source']).read_text(),1.,json.dumps(baseline))
    saved = json.loads(manifest.read_text()) if manifest.exists() else None
    if saved and saved['config'] != config: raise ValueError('Resume configuration changed')
    round_index = saved['next_round'] if saved else 0
    checkpoint = saved['checkpoint_id'] if saved else config['initial_checkpoint']
    archive = open_archive(root/'archive.json',seed,resume_step=round_index if saved else None)
    for step in range(round_index,config['rounds']):
        folder = root/f'round_{step:04d}'
        folder.mkdir(exist_ok=True)
        selection = folder/'parents.json'
        if selection.exists():
            from upstream import State
            parents = [State.from_dict(s) for s in json.loads(selection.read_text())]
        else:
            parents = archive.sample_states(config['groups'])
            atomic_json(selection,[s.to_dict() for s in parents])
        generation = stage(config['backend_command'],dict(action='generate',model=config['model'],
            checkpoint_id=checkpoint,reference_checkpoint=config['initial_checkpoint'],
            allocation_resolved=config.get('allocation_resolved',False),responses=config['responses'],
            groups=[dict(parent_id=p.id,prompt=prompt(p,baseline)) for p in parents],
            generation=config['generation']),folder/'rollouts.json',config['stage_timeout_s'])
        records = generation['records']
        if any(r['checkpoint_id'] != checkpoint for r in records): raise ValueError('Stale rollout sampler')
        if len(records) != len(parents)*config['responses']: raise ValueError('Incomplete group batch')
        parent_lookup = {p.id:p for p in parents}
        children, child_parents, evaluations = [],[],[]
        for index,record in enumerate(records):
            if record['parent_id'] not in parent_lookup: raise ValueError('Unexpected rollout parent')
            source = folder/f'candidate_{index:04d}.py'
            output = folder/f'evaluation_{index:04d}.json'
            if output.exists(): result = json.loads(output.read_text())
            else:
                try:
                    source.write_text(extract_source(record['response']))
                    # Evaluator owns this command and its immutable checker.
                    result = stage(config['evaluator_command'],dict(source=str(source),
                        baseline=str(baseline_path)),output,config['stage_timeout_s'])
                except Exception as exc:
                    result = dict(ok=False,reward=0.,failure=str(exc))
                    atomic_json(output,result)
            try: r = reward(result,baseline)
            except Exception as exc: r=0.; result.update(ok=False,failure=str(exc))
            result['reward'] = r
            evaluations.append(result)
            child = kernel_state(source.read_text() if source.exists() else '',r,json.dumps(result),step+1)
            child.id = hashlib.sha256(f'{step}:{index}:{record["parent_id"]}'.encode()).hexdigest()
            children.append(child); child_parents.append(parent_lookup[record['parent_id']])
        advantages = [0.]*len(records)
        math_stats = []
        for parent in parents:
            indices = [i for i,r in enumerate(records) if r['parent_id']==parent.id]
            if len(indices) != config['responses']: raise ValueError('Incomplete same-parent group')
            values,stats = entropic_advantages([evaluations[i]['reward'] for i in indices])
            for i,a in zip(indices,values,strict=True): advantages[i]=float(a)
            math_stats.append(dict(parent_id=parent.id,**stats))
        if not config.get('frozen',False):
            update = stage(config['backend_command'],dict(action='update',model=config['model'],
                checkpoint_id=checkpoint,reference_checkpoint=config['initial_checkpoint'],
                allocation_resolved=config.get('allocation_resolved',False),records=records,
                advantages=advantages,checkpoint_dir=str(root/'checkpoints'),tag=f'round_{step:04d}'),
                folder/'update.json',config['stage_timeout_s'])
            if not update.get('reload_verified') or update.get('adapter_max_abs_change',0)<=0:
                raise RuntimeError('Missing real update/reload evidence')
            checkpoint = update['checkpoint_id']
        archive.update_states(children,child_parents,save=False)
        archive.flush(step=step+1)
        # Champion promotion requires an independent fresh worker evaluation.
        champion_path = root/'champion.json'
        champion = json.loads(champion_path.read_text()) if champion_path.exists() else {'reward':1.}
        best = max(range(len(children)),key=lambda i:children[i].value)
        if children[best].value > champion['reward']:
            recheck = stage(config['evaluator_command'],dict(source=str(folder/f'candidate_{best:04d}.py'),
                baseline=str(baseline_path)),folder/'champion_recheck.json',config['stage_timeout_s'])
            confirmed = reward(recheck,baseline)
            if confirmed > champion['reward']:
                atomic_json(champion_path,dict(reward=confirmed,source=children[best].code,evaluation=recheck))
        summary = dict(round=step,attempts=len(records),valid_rate=sum(e['ok'] for e in evaluations)/len(records),
            best_reward=max(c.value for c in children),math=math_stats,checkpoint_id=checkpoint,
            frozen=config.get('frozen',False))
        atomic_json(folder/'summary.json',summary)
        atomic_json(manifest,dict(next_round=step+1,checkpoint_id=checkpoint,config=config))
        print(json.dumps(summary),flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    args = p.parse_args()
    run(json.loads(Path(args.config).read_text()))
