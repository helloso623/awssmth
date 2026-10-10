"""Full-source actions. No fixed configuration menu or inference endpoint."""
import ast
import json
import re
from pathlib import Path

ALLOWED_MODULES = {'nki.language': 'nl', 'nki.isa': 'nisa'}
SAFE_CALLS = {'min', 'max', 'range', 'float', 'int', 'len'}

def validate_source(source):
    if len(source.encode()) > 100_000: raise ValueError('Source exceeds 100 KB')
    tree = ast.parse(source)
    functions = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, ast.Import):
            for name in node.names:
                if ALLOWED_MODULES.get(name.name) != name.asname:
                    raise ValueError('Only nki.language as nl and nki.isa as nisa imports permitted')
        elif isinstance(node, ast.FunctionDef):
            functions.append(node)
            if node.name != 'kernel' or node.decorator_list or node.returns or node.args.defaults:
                raise ValueError('Define undecorated kernel(x) only')
            if [a.arg for a in node.args.args] != ['x'] or node.args.kwonlyargs or node.args.vararg or node.args.kwarg or node.args.posonlyargs:
                raise ValueError('Required interface: kernel(x)')
            if node.args.args[0].annotation: raise ValueError('No evaluated annotations')
        else: raise ValueError('No executable module-level candidate statements')
    if len(functions) != 1: raise ValueError('Exactly one kernel function required')
    denied = (ast.ImportFrom, ast.Global, ast.Nonlocal, ast.With, ast.Try,
              ast.ClassDef, ast.AsyncFunctionDef, ast.Lambda, ast.Delete,
              ast.Yield, ast.YieldFrom, ast.Await, ast.NamedExpr)
    for node in ast.walk(functions[0]):
        if isinstance(node, denied) or isinstance(node, ast.Import):
            raise ValueError(f'Unsupported statement: {type(node).__name__}')
        if isinstance(node, ast.Attribute) and node.attr.startswith('_'):
            raise ValueError('Private attributes forbidden')
        if isinstance(node, ast.Name) and node.id.startswith('__'):
            raise ValueError('Private names forbidden')
        if isinstance(node, ast.Call):
            f = node.func
            allowed = isinstance(f, ast.Name) and f.id in SAFE_CALLS
            allowed |= (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                        and f.value.id in ('nl', 'nisa') and not f.attr.startswith('_'))
            if not allowed: raise ValueError('Calls must be supported NKI primitives or static scalar helpers')
    return source

def extract_source(response):
    # Reasoning remains in the saved response and tokens. Only final source runs.
    matches = re.findall(r'<NKI_SOURCE>\s*```python\n(.*?)\n```\s*</NKI_SOURCE>', response, re.S)
    if len(matches) != 1: raise ValueError('Exactly one tagged Python source block required')
    return validate_source(matches[0]+'\n')

def prompt(parent, baseline):
    example = (Path(__file__).parent/'baseline.py').read_text()
    return f'''Generate a complete replacement Trainium2 NKI fused row-softmax kernel.
Contract: input FP32 matrix X; output same shape/dtype, kernel(x).
For each row: m=max(X), E=exp(X-m), S=sum(E), Y=E/S.
Required shapes: 8x16,128x512,129x513,1x7,7x1,128x1,31x127,257x61.
Original checker tolerance: maximum relative error 1e-4; all tests must pass.
Finite hostile, constant and dominant-value rows are tested independently.
Tiles: at most 128 rows and 512 columns; handle exact ragged slices.
One fused invocation; on-chip intermediates where feasible. Assigned cores 0–1, LNC2.
Use installed native nki.language as nl and nki.isa as nisa (no legacy neuronxcc).
Use nl.subtract/nl.multiply rather than overloaded tensor arithmetic; first dimension is partition.
Source imports only these modules. Checker and benchmark are external and immutable.
Example of installed syntax:
```python
{example}
```
Parent source:
```python
{parent.code}
```
Parent feedback: {parent.observation}
Parent reward: {parent.value}
Fixed baseline: {json.dumps(baseline.get('timings'))}
Return full code using exactly <NKI_SOURCE> followed by a ```python block and </NKI_SOURCE>.
The reward is correctness-gated geometric mean speedup on the three declared benchmark shapes.
'''
