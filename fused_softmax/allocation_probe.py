"""One tiny correctness invocation on assigned cores; no benchmark."""
import os
from pathlib import Path
import numpy as np
from device import compile_kernel
from baseline import kernel

if __name__ == '__main__':
    if os.environ.get('NEURON_RT_VISIBLE_CORES') != '0,1':
        raise RuntimeError('Probe requires cores 0–1')
    if os.environ.get('NEURON_LOGICAL_NC_CONFIG') != '2':
        raise RuntimeError('Probe requires LNC2')
    x = np.zeros((1, 7), np.float32)
    compiled = compile_kernel(kernel, x, Path(__file__).parent/'experiments/allocation_probe.artifacts')
    actual = next(iter(compiled.run(x=x).outputs.values()))
    np.testing.assert_allclose(actual, np.full_like(x, 1/7), rtol=1e-4, atol=1e-7)
    print('PASS: one 1x7 baseline invocation on cores 0–1, LNC2', flush=True)
