import torch
import numpy as np

def robust(M, R, main, reg, l_r, para, type):
    if main == 'F':
        D1 = torch.ones(M.shape[1]).type(type)
    elif main == '21':
        D1 = 1/torch.norm(M, dim=0)
    elif main == 'co':
        D1 = torch.exp(-(torch.norm(M, dim=0)**2)/(2*(para**2)))
    elif main == 'el':
        D1 = (1+para)*(torch.norm(M, dim=0)+2*para) / torch.maximum(2*(torch.norm(M, dim=0)+para)**2, torch.tensor(1e-10))


    if reg == 'F':
        D2 = torch.ones(R.shape[1]).type(type)
    elif reg == '21':
        D2 = 1/torch.norm(R, dim=0)
    C = torch.diag(D1/(l_r * D2 - D1))
    return C


def calcD(M, main, para, type):
    if main == 'F': # D matrix for Square error NMF
        d = torch.ones(M.shape[1]).type(type)
    elif main == '21': # D matrix for NMF-2,1
        d = 1/torch.norm(M, dim=0)
    elif main == 'co': # D matrix for Correntropy NMF
        d = torch.exp(-(torch.norm(M, dim=0)**2)/(2*(para**2)))
    elif main == 'el': # D matrix for Elastic NMF
        d = (1+para)*(torch.norm(M, dim=0)+2*para)/(2*(torch.norm(M, dim=0)+para)**2)
    return d