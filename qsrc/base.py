"""Validation tuning and quantum matrices, preserving the original numerical path."""
import itertools
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.svm import SVC
from .quantum import states, fidelity
from .solver import predict

def metrics(y, pred):
    return {'accuracy':float(accuracy_score(y,pred)),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0))}

def tune_src(K, kv, labels, yv, lambdas, mus):
    best = None
    for lam, mu in itertools.product(lambdas,mus):
        pred, _, _ = predict(K, kv, labels, lam, mu)
        score = balanced_accuracy_score(yv,pred)
        if best is None or score > best['validation_score']:
            best = {'lam':float(lam),'mu':float(mu),'validation_score':float(score)}
    return best

def tune_svm(K, kv, labels, yv, Cs):
    best = None
    for C in Cs:
        model = SVC(kernel='precomputed',C=C).fit(K,labels)
        score = balanced_accuracy_score(yv,model.predict(kv.T))
        if best is None or score > best['validation_score']:
            best = {'C':C,'validation_score':float(score)}
    return best

def quantum_matrices(data, qubits, depth, bandwidth, entangle=True):
    args = dict(qubits=qubits,depth=depth,bandwidth=bandwidth,entangle=entangle)
    sd,sv,st = [states(data[key],**args) for key in ('D','V','T')]
    return fidelity(sd), fidelity(sd,sv), fidelity(sd,st)

def matrices_from_selected(data, exact):
    p=exact['quantum_params']
    return quantum_matrices(data,p['qubits'],p['depth'],p['bandwidth'])
