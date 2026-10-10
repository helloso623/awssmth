"""NKI 0.3 implementation; all softmax intermediates live in SBUF."""
import nki.language as nl
import nki.isa as nisa


def kernel(x):
    rows = 128
    fused = True
    reuse = True
    m, n = x.shape
    y = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    for start in nl.static_range(0, m, rows):
        count = min(rows, m - start)
        chunks = (n + 511) // 512
        exponentials = nl.ndarray((count, chunks, 512),
                                  dtype=nl.float32, buffer=nl.sbuf)
        cached = nl.ndarray((count, chunks, 512),
                            dtype=nl.float32, buffer=nl.sbuf)
        maximum = nl.full((count, 1), float('-inf'), dtype=nl.float32, buffer=nl.sbuf)
        for c in nl.static_range(chunks):
            offset = c * 512
            width = min(512, n - offset)
            a = nl.load(x[start:start + count, offset:offset + width])
            if reuse:
                nisa.tensor_copy(dst=cached[:, c, :width], src=a)
            maximum = nl.maximum(maximum, nl.max(a, axis=1, keepdims=True))
        negative = nl.multiply(maximum, -1.0)
        total = nl.zeros((count, 1), dtype=nl.float32, buffer=nl.sbuf)
        for c in nl.static_range(chunks):
            offset = c * 512
            width = min(512, n - offset)
            if reuse:
                a = cached[:, c, :width]
            else:
                a = nl.load(x[start:start + count, offset:offset + width])
            if fused:
                subtotal = nl.ndarray((count, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation_reduce(dst=exponentials[:, c, :width], op=nl.exp,
                                       data=a, bias=negative, reduce_op=nl.add,
                                       reduce_res=subtotal)
            else:
                e = nl.exp(nl.subtract(a, maximum))
                nisa.tensor_copy(dst=exponentials[:, c, :width], src=e)
                subtotal = nl.sum(e, axis=1, keepdims=True)
            total = nl.add(total, subtotal)
        inverse = nl.reciprocal(total)
        for c in nl.static_range(chunks):
            offset = c * 512
            width = min(512, n - offset)
            result = nl.multiply(exponentials[:, c, :width], inverse)
            nl.store(y[start:start + count, offset:offset + width], result)
    return y
