"""Load small upstream components without importing the authors' Ray launcher.

The vendored source is unmodified. The entropic estimator is extracted from
the actual function, rather than reimplementing its attainable-target path.
"""
import ast
import math
import sys
import types
from pathlib import Path
import numpy as np
import torch

VENDOR = Path(__file__).parent / 'vendor/discover/ttt_discover'
if 'ttt_discover' not in sys.modules:
    package = types.ModuleType('ttt_discover')
    package.__path__ = [str(VENDOR)]
    sys.modules['ttt_discover'] = package

def __getattr__(name):
    # Training mathematics must not require the Tinker SDK. Only the unchanged
    # authors' archive/types import its annotations on the host controller.
    if name == 'PUCTSampler':
        from ttt_discover.tinker_utils.sampler import PUCTSampler
        return PUCTSampler
    if name == 'State':
        from ttt_discover.tinker_utils.state import State
        return State
    raise AttributeError(name)

def author_advantages(groups):
    tree = ast.parse((VENDOR / 'rl/train.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == 'compute_advantages')
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), fn], type_ignores=[])
    ns = dict(torch=torch, np=np, math=math)
    exec(compile(ast.fix_missing_locations(module), str(VENDOR / 'rl/train.py'), 'exec'), ns)
    return ns['compute_advantages'](groups, 'entropic_adaptive_beta', 1.)
