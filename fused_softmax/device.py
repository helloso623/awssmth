"""Installed NKI compiler bridge. Compiled objects expose device-only benchmark."""
from pathlib import Path
import nki
from nki.framework.compiled import compile_kernel_to_nir
from nki.compiler.frontend import resolve_frontend_cls
from nki.compiler.ncc_driver import CompileOptions, compile_bir_to_neff

def compile_kernel(fn, x, directory, **static):
    directory = str(Path(directory).resolve())
    Path(directory).mkdir(parents=True, exist_ok=True)
    opts = CompileOptions(target='trn2', lnc=2, artifacts_dir=directory,
                          output_path=str(Path(directory)/'kernel.neff'))
    k = nki.jit(fn)[2]
    bir = compile_kernel_to_nir(k, inputs=dict(x=x, **static), compile_opts=opts,
                               frontend=resolve_frontend_cls()(), enable_cache=False)
    return compile_bir_to_neff(opts, bir, input_arrays=[],
                              argument_names=[s.name for s in bir.descriptor.input_specs],
                              output_arg_names=[s.name for s in bir.descriptor.output_specs])

if __name__ == '__main__':
    import numpy as np
    from kernel_v2 import kernel
    x = np.zeros((8, 16), np.float32)
    compiled = compile_kernel(kernel, x, 'probe_v2', rows=128, fused=False)
    print(compiled.run(x=x))
    print(compiled.benchmark(x=x, warmup=10, iterations=100))
