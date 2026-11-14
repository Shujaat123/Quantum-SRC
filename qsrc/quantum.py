"""Exact pure-state encoding and fidelity kernels; qubit zero is least significant."""
import numpy as np

def ry(state, angle, wire):
    ids = np.arange(len(state))
    lo = ids[(ids & (1 << wire)) == 0]
    hi = lo | (1 << wire)
    a, b = state[lo].copy(), state[hi].copy()
    c, s = np.cos(angle / 2), np.sin(angle / 2)
    state[lo], state[hi] = c*a - s*b, s*a + c*b

def rz(state, angle, wire):
    bit = (np.arange(len(state)) >> wire) & 1
    state *= np.exp(0.5j * angle * (2*bit - 1))

def cx(state, control, target):
    ids = np.arange(len(state))
    lo = ids[((ids >> control) & 1 == 1) & ((ids >> target) & 1 == 0)]
    hi = lo ^ (1 << target)
    state[lo], state[hi] = state[hi].copy(), state[lo].copy()

def encode(x, qubits=4, depth=2, entangle=True, bandwidth=1.0):
    """Data reuploading RY/RZ circuit with alternating nearest-neighbor CNOTs.

    x is already mapped to [0, pi]. No trainable circuit parameters.
    Every layer uses all coordinates by cyclic assignment when d != qubits.
    """
    x = np.asarray(x, dtype=float)
    state = np.zeros(2**qubits, complex)
    state[0] = 1
    for layer in range(depth):
        for j in range(max(len(x), qubits)):
            wire = j % qubits
            angle = bandwidth*x[j % len(x)]
            ry(state, angle, wire)
            rz(state, angle * (0.5 + 0.25*layer), wire)
        if entangle:
            for j in range(qubits - 1):
                c, t = (j, j+1) if layer % 2 == 0 else (j+1, j)
                cx(state, c, t)
    return state

def states(X, **kwargs):
    return np.stack([encode(x, **kwargs) for x in X])

def fidelity(A, B=None):
    B = A if B is None else B
    return np.clip(np.abs(A.conj() @ B.T)**2, 0, 1).real
