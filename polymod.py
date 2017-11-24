import numpy as np


def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)


def modinv(a, m):
    g, x, y = egcd(a % m, m)
    if g != 1:
        raise Exception('modular inverse does not exist')
    else:
        return x % m


def lagrange(x: np.array, w: np.array, m, n):
    res = np.zeros(n, dtype=int)
    M = len(x)
    p = np.poly1d(np.array([0], dtype=int))
    for j in range(M):
        pt = np.poly1d(np.array(w[j], dtype=int))
        for k in range(M):
            if k == j:
                continue
            fac = x[j]-x[k]
            pt = np.poly1d((pt * np.poly1d(np.array([1, -x[k]], dtype=int)) * modinv(fac, m)).coeffs % m)
        p = np.poly1d(np.array((p + pt).coeffs, dtype=int) % m)
    res[-len(p.coeffs):] = p.coeffs
    return res
