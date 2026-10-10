"""The smoke policy emits configuration edits; expand them into standalone code."""
from pathlib import Path
import re
from kernel_v2 import kernel
import inspect

def materialize(rows=128, fused=False, reuse=False):
    if rows not in (8, 16, 32, 64, 128):
        raise ValueError('rows must be 8,16,32,64,128')
    code = Path(inspect.getfile(kernel)).read_text()
    return re.sub(r'def kernel\(x, rows=128, fused=False, reuse=False\):',
                  f'def kernel(x):\n    rows = {rows}\n    fused = {bool(fused)}\n    reuse = {bool(reuse)}', code)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--rows', type=int, default=128)
    p.add_argument('--fused', action='store_true')
    p.add_argument('--reuse', action='store_true')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    Path(args.output).write_text(materialize(args.rows, args.fused, args.reuse))
