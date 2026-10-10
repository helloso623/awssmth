"""Own evaluator subprocess; enforce core allocation and candidate AST restrictions.

Static validation is a restricted kernel DSL, not an OS security sandbox.
For adversarial untrusted models an organizer-provided container/seccomp
sandbox with read-only checker mounts is still required.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from source_policy import validate_source

ROOT = Path(__file__).parent

def evaluate_candidate(source, output, baseline=None, timeout=900, heldout=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    validate_source(source.read_text())
    env = dict(os.environ, NEURON_RT_VISIBLE_CORES='0,1', NEURON_LOGICAL_NC_CONFIG='2')
    env.pop('NEURON_VISIBLE_DEVICES', None)
    command = [sys.executable, str(ROOT/'evaluate.py'), '--source', str(source), '--output', str(output)]
    if baseline: command += ['--baseline', str(Path(baseline).resolve())]
    if heldout: command += ['--heldout']
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix('.log').open('w') as log:
        process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try: process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Only the process group created by this evaluator, never server PIDs.
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            result = dict(ok=False,reward=0.,failure='Evaluator timeout')
            output.write_text(json.dumps(result))
            return result
    if not output.exists():
        result = dict(ok=False,reward=0.,failure=f'Worker exited {process.returncode} without result')
        output.write_text(json.dumps(result))
        return result
    result = json.loads(output.read_text())
    if process.returncode or not result.get('ok'):
        result.update(ok=False,reward=0.)
    return result
