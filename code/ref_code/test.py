from bs_ec_snsbs import SNS_Euro_Call_BootStrap

import numpy as np
import pandas as pd

Gamma = 10000

n_front_vec = np.arange(50, 101, 5)
n_back_vec = np.arange(50, 101, 5)

d = 20
sigma = 0.1
S_0 = 100
K = [90, 100, 110]
mu = 0.08
r = 0.05
tau = 3/50
T = 1
level_list = [0.8, 0.9, 0.95, 0.99, 0.996]

result_true = pd.read_csv("./data/trueValue_11.csv")

L0 = result_true["VaR_0.9"].values[0]
print(L0)

print(SNS_Euro_Call_BootStrap(Gamma, n_front_vec, n_back_vec, d, S_0, K, mu, sigma, r, tau, T,
                                    L0, I=500, level_list=[0.8, 0.9, 0.95, 0.99, 0.996]))