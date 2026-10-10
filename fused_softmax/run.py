"""Validate real device execution separately from device timing."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import neuronxcc.nki as nki
from kernel import softmax

def inputs(shape):
    rng = np.random.default_rng(5)
    yield 'normal', rng.standard_normal(shape).astype(np.float32)
    yield 'large', (rng.standard_normal(shape)*1e3+1e4).astype(np.float32)
    yield 'zeros', np.zeros(shape, np.float32)
    yield 'constant', np.full(shape, -1234., np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape', type=int, nargs=2, default=[1024, 4096])
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--output', default='results.json')
    args = parser.parse_args()
    torch.set_num_threads(2)
    records = []
    shapes = [(1, 7), (7, 1), (31, 127), (129, 513), (257, 61), tuple(args.shape)]
    for rows, fused in [(128, False), (32, True), (64, True), (128, True)]:
        execute = nki.baremetal(softmax, additional_compile_opt='--target=trn2')
        worst = 0.
        for shape in shapes:
            for label, x in inputs(shape):
                actual = execute[2](x, rows, fused)
                reference = torch.softmax(torch.from_numpy(x), dim=-1).numpy()
                np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-7,
                                           err_msg=f'{rows=} {fused=} {shape=} {label}')
                worst = max(worst, float(np.max(np.abs(actual-reference))))
        print(f'PASS rows={rows} fused={fused} max_abs={worst}', flush=True)
        bench = nki.benchmark(softmax, warmup=20, iters=args.iters,
                              additional_compile_opt='--target=trn2')
        x = next(inputs(tuple(args.shape)))[1]
        times = []
        for repeat in range(3):
            bench[2](x, rows, fused)
            times.append(bench.benchmark_result.nc_latency.get_latency_percentile(50))
        record = dict(rows=rows, fused=fused, correctness='pass', max_abs=worst,
                      p50_us=times, median_us=float(np.median(times)))
        records.append(record)
        print(json.dumps(record), flush=True)
        Path(args.output).write_text(json.dumps(dict(shape=args.shape, iterations=args.iters,
                                                    records=records), indent=2)+'\n')
    best = min(records, key=lambda r: r['median_us'])
    print('BEST', best, 'SPEEDUP', records[0]['median_us']/best['median_us'])

if __name__ == '__main__':
    main()
