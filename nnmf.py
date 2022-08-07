import torch
import numpy as np
from side import np_to_var, var_to_np, np_to_tensor


def nnmf(V, k, iter, eps, dtype):
    d, n = V.shape

    W = abs(np.random.normal(0, 1, size=(d, k)))
    W = np_to_var(W)[0].type(dtype)
    H = abs(np.random.normal(0, 1, size=(k, n)))
    H = np_to_var(H)[0].type(dtype)

    for i in range(iter):

        Uu = (V @ H.T)
        Ud = (W @ H) @ H.T + W
        W  = W * (Uu / torch.maximum(Ud, torch.tensor(1e-10)))

        Hu = W.T @ V
        Hd = W.T @ (W @ H) + H
        H = H * (Hu / torch.maximum(Hd, torch.tensor(1e-10)))

    return W, H