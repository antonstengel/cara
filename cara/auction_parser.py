import numpy as np

from cara.distribution import Distribution
from cara.hyperparameters import DEFAULT_HYPERPARAMETERS

class Parser:
    """Parses the raw string from an input txt."""
    
    def __init__(self, s: str):
        self.lines = s.split('\n')

        if len(self.lines[0].split()) != 6:
            raise ValueError('Incorrect number of parameters in first line.')
        self.n, self.m, self.c, self.r, self.t, self.a = list(map(int, self.lines[0].split()))
        self.p_nc = np.zeros((self.n, self.c))
        self.N_n = np.empty(self.n, dtype=object)
        self.d_mc = np.empty((self.m, self.c), dtype=object)

        self.phi_r = np.empty(self.r, dtype=object)
        self.inv_phi_r = np.empty(self.r, dtype=object)
        self.L_r = np.zeros(self.r)
        self.U_r = np.zeros(self.r)
        self.T_r = np.empty(self.r, dtype=object)
        self.S_r = np.zeros(self.r)
        self.P_r = np.empty(self.r, dtype=object)

        self.h = {}
        
        # reading in p_nc, N_n
        offset = 1
        for ni in range(self.n):
            words = self.lines[ni+offset].split()
            if len(words) != self.c + 1:
                raise ValueError('Incorrect number of parameters in line.')
            for ci in range(self.c):
                self.p_nc[ni, ci] = float(words[ci])
            self.N_n[ni] = words[-1]

        # reading in d_mc
        offset += self.n
        for mi in range(self.m):
            words = self.lines[mi+offset].split()
            if len(words) != self.c:
                raise ValueError('Incorrect number of parameters in line.')
            for ci in range(self.c):
                self.d_mc[mi, ci] = words[ci]

        # reading in phi_r, L_r, U_r, T_r, S_r, P_r
        offset += self.m
        for ri in range(self.r):
            words = self.lines[ri+offset].split()
            self.L_r[ri] = np.float64(words[1])
            self.U_r[ri] = np.float64(words[2])

            phis = words[0].split('|')
            if len(phis) != 1 and len(phis) != 2:
                raise ValueError('Continuous valuation distribution is invalid.')
            elif len(phis) == 1:
                self.inv_phi_r[ri] = None
            elif len(phis) == 2:
                self.inv_phi_r[ri] = phis[1]
                if set(self.inv_phi_r[ri]) > set('123456789+-/*()y'):
                    raise ValueError('PPF contains bad characters.')              
            self.phi_r[ri] = Distribution(phis[0], self.L_r[ri], self.U_r[ri])
                            
            self.T_r[ri] = words[3]
            
            if self.T_r[ri] == 'N/A':
                self.S_r[ri] = None
                self.P_r = None
            else:
                self.S_r[ri] = np.float64(words[4])
                if len(words) > 5:
                    self.P_r[ri] = words[5:]
                else:
                    self.P_r[ri] = None

        # reading in hyperparameters
        if '=' in self.lines[-1]:
            for s in self.lines[-1].split():
                if len(s.split('=')) != 2:
                    raise ValueError('Invalid hyperparameter.')
                k, v = s.split('=')
                if k not in DEFAULT_HYPERPARAMETERS.keys():
                    raise ValueError('Invalid hyperparameter.')
                self.h[k] = v