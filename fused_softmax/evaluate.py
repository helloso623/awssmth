"""Isolated Trainium2 evaluation. Never rewards simulator/CPU timings."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from device import compile_kernel
from source_policy import validate_source

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import kernelbench

BENCHMARK_SHAPES = [(128, 512), (129, 513), (257, 61)]
HELDOUT_SHAPES = [(17, 509), (131, 521)]
TIMING_LABEL = ('Trainium synchronous invocation wall time in microseconds: '
                'median of three means; includes input/output transfers, allocation, '
                'and runtime overhead; excludes compilation and model loading')

def timing_contract(seed, iterations, heldout):
    return dict(timing=TIMING_LABEL, timing_version=2, cores=[0, 1], lnc=2,
                iterations=iterations, warmup=20, repeats=3, seed=seed,
                unseen_shapes=heldout)

def validate_baseline(baseline, contract, shapes):
    if not baseline.get('ok'):
        raise ValueError('Baseline did not pass correctness')
    for key, expected in contract.items():
        if baseline.get(key) != expected:
            raise ValueError(f'Baseline {key} mismatch; regenerate with identical evaluator settings')
    timings = baseline.get('timings', [])
    if [t.get('shape') for t in timings] != [list(s) for s in shapes]:
        raise ValueError('Baseline benchmark shapes mismatch')
    for timing in timings:
        latency = timing.get('latency_us')
        if not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency <= 0:
            raise ValueError('Baseline contains invalid latency')

def supplement(shape, seed):
    rng = np.random.default_rng(seed)
    yield 'fresh_random', rng.standard_normal(shape).astype(np.float32)
    yield 'constant', np.full(shape, -1234., np.float32)
    x = np.full(shape, -1000., np.float32)
    x[np.arange(shape[0]), rng.integers(0, shape[1], shape[0])] = 1000.
    yield 'dominant', x
    x = rng.choice(np.array([-1e30, 0, 1e30], np.float32), shape)
    yield 'extreme_finite', x

def evaluate(source, artifacts, seed=911, iterations=100, heldout=False):
    if os.environ.get('NEURON_RT_VISIBLE_CORES') not in ('0,1', '0-1'):
        raise RuntimeError('Evaluator requires assigned cores 0–1 explicitly')
    if os.environ.get('NEURON_LOGICAL_NC_CONFIG') != '2':
        raise RuntimeError('Evaluator requires LNC2')
    if iterations < 1: raise ValueError('iterations must be positive')
    validate_source(Path(source).read_text())
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location('candidate', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    shapes = HELDOUT_SHAPES if heldout else BENCHMARK_SHAPES
    cases = [] if heldout else [(label, args[0]) for label, args in kernelbench.LEVELS[5]['build']()]
    for shape in (HELDOUT_SHAPES if heldout else sorted(set(x.shape for _, x in cases))):
        cases.extend(supplement(shape, seed + shape[0]*31 + shape[1]))
    compiled = {}
    checks = []
    for label, x in cases:
        shape = x.shape
        if shape not in compiled:
            compiled[shape] = compile_kernel(module.kernel, x, Path(artifacts)/f'{shape[0]}x{shape[1]}')
        actual = next(iter(compiled[shape].run(x=x).outputs.values()))
        # Independent CPU float64 softmax then cast to the task output dtype.
        reference = torch.softmax(torch.from_numpy(x).double(), dim=-1).float().numpy()
        mismatch = kernelbench.describe_mismatch(actual, reference, 1e-4)
        if not heldout:
            required = kernelbench.describe_mismatch(actual, kernelbench._ref_5(x), 1e-4)
            mismatch = mismatch or required
        check = dict(shape=shape, label=label, ok=mismatch is None,
                     max_abs=float(np.max(np.abs(actual-reference))), detail=mismatch)
        checks.append(check)
        print(json.dumps(check), flush=True)
        if mismatch:
            return dict(ok=False, reward=0., checks=checks, failure=mismatch)
    timings = []
    for shape in shapes:
        x = next(supplement(shape, seed))[1]
        if shape not in compiled:
            compiled[shape] = compile_kernel(module.kernel, x, Path(artifacts)/f'{shape[0]}x{shape[1]}')
        measurements = []
        for _ in range(3):
            for _ in range(20): compiled[shape].run(x=x)
            # run() returns NumPy outputs after synchronous execution/readback.
            # Time a complete invocation, not the mean of two LNC trace events.
            samples = []
            for _ in range(iterations):
                start = time.perf_counter_ns()
                result = compiled[shape].run(x=x)
                samples.append((time.perf_counter_ns()-start)/1000.)
            if not all(math.isfinite(t) and t > 0 for t in samples):
                raise RuntimeError('Invalid latency measurements')
            measurements.append(float(np.mean(samples)))
        timings.append(dict(shape=shape, mean_us_repeats=measurements,
                            latency_us=float(np.median(measurements))))
    return dict(ok=True, checks=checks, timings=timings,
                **timing_contract(seed, iterations, heldout))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--baseline')
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--seed', type=int, default=911)
    p.add_argument('--heldout', action='store_true')
    args = p.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        baseline = None
        if args.baseline:
            baseline = json.loads(Path(args.baseline).read_text())
            validate_baseline(baseline, timing_contract(args.seed, args.iterations, args.heldout),
                              HELDOUT_SHAPES if args.heldout else BENCHMARK_SHAPES)
        result = evaluate(args.source, output.with_suffix('.artifacts'), args.seed, args.iterations, args.heldout)
        if baseline is not None and result['ok']:
            ratios = []
            for base, candidate in zip(baseline['timings'], result['timings'], strict=True):
                ratios.append(base['latency_us']/candidate['latency_us'])
            result['speedups'] = ratios
            result['reward'] = math.exp(sum(math.log(r) for r in ratios)/len(ratios))
        elif result['ok']:
            result['reward'] = 1.
    except Exception:
        result = dict(ok=False, reward=0., failure=traceback.format_exc())
    result['source_sha256'] = hashlib.sha256(Path(args.source).read_bytes()).hexdigest()
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'checks'}), flush=True)
    return 0 if result['ok'] else 1

if __name__ == '__main__':
    sys.exit(main())
