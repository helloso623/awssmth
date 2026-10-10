"""Stable FP32 row softmax; intermediates remain in SBUF."""
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa


def softmax(x, rows=128, fused=False):
    m, n = x.shape
    y = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    p = nl.arange(rows)[:, None]
    f = nl.arange(n)[None, :]
    for t in nl.affine_range((m + rows - 1) // rows):
        r = t * rows + p
        a = nl.load(x[r, f], mask=r < m).astype(nl.float32)
        maximum = nl.max(a, axis=1, keepdims=True)
        if fused:
            total = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
            e = nisa.activation_reduce(op=nl.exp, data=a, bias=-maximum,
                                       reduce_op=nl.add, reduce_res=total)
            result = e * nl.reciprocal(total)
        else:
            e = nl.exp(a - maximum)
            total = nl.sum(e, axis=1, keepdims=True)
            result = e / total
        nl.store(y[r, f], result, mask=r < m)
    return y
