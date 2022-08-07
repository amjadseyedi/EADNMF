import torch
import numpy as np
from robust import calcD
from side import np_to_var, var_to_np, np_to_tensor


def rwnmf(J, V, k, iter, eps, cost, para, dtype):
    d, n = V.shape

    W = abs(np.random.normal(0, 1, size=(d, k)))
    W = np_to_var(W)[0].type(dtype)
    H = abs(np.random.normal(0, 1, size=(k, n)))
    H = np_to_var(H)[0].type(dtype)

    J = J.type(dtype)
    JV = J.type(dtype) * V.type(dtype)
    for i in range(iter):

        d0 = J.shape[0]
        d = calcD(JV - J * (W @ H), cost, para, dtype)

        Uu = ((JV * d.repeat(d0,1)) @ H.T)
        Ud = ((J * (W @ H)) * d.repeat(d0, 1)) @ H.T + W
        W = W * (Uu / torch.maximum(Ud, torch.tensor(1e-10)))

        Vu = W.T @ (JV * d.repeat(d0, 1))
        Vd = W.T @ ((J * (W @ H)) * d.repeat(d0, 1)) + H
        H = H * (Vu / torch.maximum(Vd, torch.tensor(1e-10)))

    return W, H
