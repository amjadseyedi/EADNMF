import torch
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from robust import calcD
from sklearn.decomposition import NMF
from rwnmf import rwnmf
from nnmf import nnmf
from side import np_to_var, var_to_np, np_to_tensor


class DNMF(object):
    def __init__(self, J, A, args):
        self.A = A
        self.A0 = A
        self.args = args
        self.p = len(self.args.layers)
        self.J = J

    def setup_z(self, i):
        if i == 0:
            self.Z = self.A
        else:
            self.Z = self.V_s[i - 1]

    def sklearn_pretrain(self, i):
        if i == 0:
            U, V = rwnmf(self.J, self.Z, self.args.layers[i], self.args.pre_iterations, 10 ** -6,
                         self.args.cost, self.args.para, self.args.type)
        else:
            U, V = nnmf(self.Z, self.args.layers[i], self.args.pre_iterations, 10 ** -6, self.args.type)

        return U, V

    def pre_training(self):
        # Pre-training each NMF layer.
        print("\nLayer pre-training started. \n")
        self.U_s = []
        self.V_s = []
        for i in tqdm(range(self.p), desc="Layers trained: ", leave=True):
            self.setup_z(i)
            U, V = self.sklearn_pretrain(i)
            self.U_s.append(U)
            self.V_s.append(V)

    def setup_Q(self):
        # Setting up Q matrices.
        self.Q_s = [None for _ in range(self.p + 1)]
        self.Q_s[self.p] = torch.eye(self.args.layers[self.p - 1]).type(self.args.type)
        for i in range(self.p - 1, -1, -1):
            self.Q_s[i] = self.U_s[i] @ self.Q_s[i + 1]

    def update_U(self, i):
        # Updating left hand factors.
        d = self.A.shape[0]
        if i == 0:
            Ru = (self.J * self.A) * self.d.repeat(d, 1) @ self.V_s[self.p - 1].T @ self.Q_s[1].T
            Rd = ((self.J * (self.U_s[0] @ self.Q_s[1] @ self.V_s[self.p - 1])) * self.d.repeat(d, 1)) @ self.V_s[
                self.p - 1].T @ self.Q_s[1].T + self.U_s[i]
            self.U_s[0] = (self.U_s[0] * (Ru / torch.maximum(Rd, torch.tensor(1e-10))))
        else:
            Ru = self.P.T @ ((self.J * self.A) * self.d.repeat(d, 1)) @ self.V_s[self.p - 1].T @ self.Q_s[i + 1].T
            Rd = self.P.T @ (
                        (self.J * (self.P @ self.U_s[i] @ self.Q_s[i + 1] @ self.V_s[self.p - 1])) * self.d.repeat(d,
                                                                                                                   1)) @ \
                 self.V_s[self.p - 1].T @ self.Q_s[i + 1].T + self.U_s[i]
            self.U_s[i] = (self.U_s[i] * (Ru / torch.maximum(Rd, torch.tensor(1e-10))))

    def update_P(self, i):
        # Setting up P matrices.
        if i == 0:
            self.P = self.U_s[0]
        else:
            self.P = self.P @ self.U_s[i]

    def update_V(self, i):
        # Updating right hand factors.
        d = self.A.shape[0]
        Vu = self.P.T @ ((self.J * self.A) * self.d.repeat(d, 1))
        Vd = self.P.T @ ((self.J * (self.P @ self.V_s[i])) * self.d.repeat(d, 1)) + self.V_s[i]
        self.V_s[i] = self.V_s[i] * (Vu / torch.maximum(Vd, torch.tensor(1e-10)))

    def calculate_cost(self, i):
        reconstruction_loss_1 = torch.sum(torch.norm(self.J * (self.A - self.P @ (self.V_s[-1])), dim=0))

    def training(self):
        # Training process after pre-training.
        self.loss = []
        self.rmse_train = []
        self.rmse_test = []
        self.rmse_train0 = []
        self.tsp = []
        for iteration in range(self.args.iterations):
            self.setup_Q()

            V_ap_old = self.Q_s[0] @ self.V_s[self.p - 1]

            for i in range(self.p):
                self.d = calcD((self.J * (self.A - self.Q_s[0] @ self.V_s[self.p - 1])), self.args.cost, self.args.para, self.args.type)
                self.update_U(i)
                self.update_P(i)
                self.update_V(i)

            V_ap_new = self.P @ self.V_s[self.p - 1]

            stp = (V_ap_old - V_ap_new) / V_ap_old
            stp[stp != stp] = 0
            print(torch.norm(stp, 'fro')**2)
            self.tsp.append(torch.norm(stp, 'fro')**2)
            rmseTr = torch.norm((self.J * self.A) - (self.J * V_ap_new), 'fro') / torch.sqrt(torch.sum(self.J))
            rmseTr0 = torch.norm((self.J * self.A0) - (self.J * V_ap_new), 'fro') / torch.sqrt(torch.sum(self.J))

            Jb = 1 - self.J
            rmseTe = torch.norm((Jb * self.A0) - (Jb * V_ap_new), 'fro') / torch.sqrt(torch.sum(Jb))
            self.rmse_test.append(rmseTe)
            self.rmse_train.append(rmseTr)
            self.rmse_train0.append(rmseTr0)
            if torch.norm(stp, 'fro') < 1e-2:
                print('Reach inner stopping criterion at out %d' % (iteration))
                break
                # if self.args.calculate_loss:
            #     self.calculate_cost(iteration)
        return self.rmse_train, self.rmse_test, self.tsp, self.rmse_train0
