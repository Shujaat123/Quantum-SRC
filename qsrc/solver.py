"""Elastic-net kernel SRC with explicit KKT residuals."""
import numpy as np
from scipy.linalg import eigh


def psd_repair(K):
    K = (K + K.T)/2
    w, V = eigh(K, check_finite=False)
    return (V * np.maximum(w, 0)) @ V.T


def kkt_residual(K, k, alpha, lam, mu):
    g = K @ alpha - k + mu*alpha
    nz = alpha != 0
    r = np.sign(g)*np.maximum(np.abs(g)-lam, 0)
    r[nz] = g[nz] + lam*np.sign(alpha[nz])
    return float(np.linalg.norm(r))


def solve(K, k, lam=0.01, mu=0.1, tol=1e-7, max_iter=5000, initial=None):
    """FISTA; k can be a single vector or columns of query vectors.

    KKT residuals are reported independently of the step stopping criterion.
    """
    if mu <= 0 or lam < 0:
        raise ValueError('Require mu > 0 and lambda >= 0')
    k = np.asarray(k, float)
    a = np.zeros_like(k) if initial is None else np.array(initial, copy=True)
    z, t = a.copy(), 1.0
    L = max(float(eigh(K, eigvals_only=True, check_finite=False)[-1]), 0) + mu
    converged = False
    for iteration in range(max_iter):
        v = z - (K @ z + mu*z - k)/L
        b = np.sign(v)*np.maximum(np.abs(v)-lam/L, 0)
        if np.linalg.norm(b-a) <= tol*(1+np.linalg.norm(a)):
            a = b
            converged = True
            break
        tn = (1+np.sqrt(1+4*t*t))/2
        z = b + ((t-1)/tn)*(b-a)
        a, t = b, tn
    # Vector and batched diagnostics, independent of the stopping heuristic.
    aa = a[:, None] if a.ndim == 1 else a
    kk = k[:, None] if k.ndim == 1 else k
    residual = max(kkt_residual(K, kk[:,j], aa[:,j], lam, mu)
                   for j in range(aa.shape[1]))
    return a, {'iterations': iteration+1, 'step_converged': converged,
               'kkt_residual': residual}


def residuals(K, k, a, labels):
    classes = np.unique(labels)
    one = a.ndim == 1
    aa, kk = (a[:,None], k[:,None]) if one else (a, k)
    out = []
    for c in classes:
        b = aa*(labels == c)[:,None]
        out.append(1 - 2*np.sum(b*kk, axis=0) + np.sum(b*(K@b), axis=0))
    out = np.stack(out, axis=-1)
    return classes, out[0] if one else out


def predict(K, k, labels, lam, mu):
    a, info = solve(K, k, lam, mu)
    classes, r = residuals(K, k, a, labels)
    return classes[np.argmin(r, axis=-1)], a, info
