import numpy as np
import pandas as pd
import method_Broadie
from sklearn.linear_model import LinearRegression
import time
import os
from joblib import Parallel, delayed


def broadie_reg(M, N, S_0, K, H, mu, sigma, r, tau, T, seed_train, seed_test):

    crude = False
    loss_flag = True
    X_train, y_train = method_Broadie.simPredictors(M, N, S_0, K, H, mu, sigma, r, tau, T, seed_train, crude, loss_flag)

    reg = LinearRegression().fit(X_train, y_train)

    X_test, y_test = method_Broadie.simPredictors(M, N, S_0, K, H, mu, sigma, r, tau, T, seed_test, crude, loss_flag)

    loss = reg.predict(X_test)

    return loss


def cal_RRMSE(M, N, S_0, K, H, mu, sigma, r, tau, T, alpha, L0, seed):

    rng = np.random.default_rng(seed)

    loss = broadie_reg(M, N, S_0, K, H, mu, sigma, r, tau, T, rng.integers(0, 2147483647), rng.integers(0, 2147483647))
    indicator = np.mean((loss > L0))
    hockey = np.mean(np.maximum(loss - L0, 0))
    quadratic = np.mean((loss - L0) ** 2)

    loss.sort()
    VaR = loss[int(np.ceil((1 - alpha) * M))]
    CVaR = np.mean(loss[loss >= VaR])

    return indicator, hockey, quadratic, VaR, CVaR


local = True

save_path = "./results/"

if not os.path.exists(save_path):
    os.makedirs(save_path)

if local:
    n_cores = 4
else:
    n_cores = 60


K = [101, 110, 114.5]
H = [91, 100, 104.5]

S_0 = 100
mu = 0.08
sigma = 0.2
r = 0.03
T = 1/12
tau = 1/52

alpha = 0.05

result_true = np.array(pd.read_csv("./trueValue_Broadie.csv")).flatten()[1:]

L0 = result_true[3]

M_list = []
for i in range(2):
    M_list = M_list + list(np.arange(1, 10) * 10 ** (i + 3))
M_list = M_list + [10 ** (i + 4)]
M_vec = np.array(M_list)
n_trials = len(M_list)
N_vec = [1] * n_trials

result_table = np.zeros([n_trials, 5])

start = time.time()

n_rep = 150

seed = np.random.randint(low=0, high=2147483647, size=n_rep)

for j in range(n_trials):

    M = M_vec[j]
    N = N_vec[j]

    res = Parallel(n_jobs=n_cores, verbose=10)(delayed(cal_RRMSE)(M, N,
                                                                  S_0, K, H, mu, sigma, r, tau, T, alpha, L0, seed[n])
                                               for n in range(n_rep))

    Bias = np.nanmean(res, axis=0) - result_true
    Variance = np.nanvar(res, axis=0)

    MSE = Bias ** 2 + Variance
    RRMSE = np.sqrt(MSE) / result_true

    result_table[j, :] = RRMSE
    print("Regression Estimation Done for:", M, "x", N)
    print(RRMSE)

print(result_table)
np.save(save_path + "result_153.npy", result_table)
