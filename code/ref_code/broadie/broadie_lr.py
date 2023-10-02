import numpy as np
import pandas as pd
import method_Broadie
import time
import os
from joblib import Parallel, delayed


def compute_Weight_D(S_0, mu, r, sigma, h, tau, sample_inner_tau):
    return (np.log(sample_inner_tau / S_0) - (mu - (1 / 2) * sigma ** 2) * tau
            - (r - (1 / 2) * sigma ** 2) * h) ** 2 / (2 * sigma ** 2 * (tau + h))


def compute_Weight_U(sample_inner_tau, sample_outer_test, r, sigma, h):

    log_ratio = np.log(sample_inner_tau.reshape(-1, 1) / sample_outer_test.reshape(1, -1))

    return (log_ratio - (r - sigma ** 2 / 2) * h) ** 2 / (2 * sigma ** 2 * h)


def compute_P(M, N, K, H, S_T, S_tau, t, r, sigma):

    prob = 1 - np.exp(- (2 * np.log(H / np.maximum(S_tau, H).reshape(N, -1))
                         * np.log(H / np.maximum(S_T, H).reshape(M, -1))) / (t * sigma ** 2))

    P = np.exp(-r * t) * (np.maximum(K - S_T, 0) * prob)

    return P


def broadie_lr(M, N, S_0, K, H, mu, sigma, r, tau, T, seed_train, seed_test):

    crude = True
    loss_flag = False
    lr_flag = True

    h = 1/156
    n = 10

    X_train, X_T, X_tau = method_Broadie.simPredictors(M, N, S_0, K, H, mu, sigma, r, tau, T, seed_train, crude,
                                                       loss_flag, lr_flag)

    X_test, y_test = method_Broadie.simPredictors(M, N, S_0, K, H, mu, sigma, r, tau, T, seed_test, crude, loss_flag)

    loss_lr = np.zeros(M)

    weight_D = compute_Weight_D(S_0, mu, r, sigma, h, tau, X_tau)

    n_partition = int(M // n)

    for i in range(n_partition):

        weight_U = compute_Weight_U(X_tau, X_test[i * n:(i + 1) * n], r, sigma, h)

        weight = np.sqrt((tau+h) / h) * np.exp(weight_D.reshape(-1, 1) - weight_U)

        t = T - tau

        K_1, K_2, K_3 = K[0], K[1], K[2]
        H_1, H_2, H_3 = H[0], H[1], H[2]

        # Computing P1, P2, P3
        P_1 = compute_P(M, N, K_1, H_1, X_T, X_test[i * n:(i + 1) * n], t, r, sigma)
        P_2 = compute_P(M, N, K_2, H_2, X_T, X_test[i * n:(i + 1) * n], t, r, sigma)
        P_3 = compute_P(M, N, K_3, H_3, X_T, X_test[i * n:(i + 1) * n], t, r, sigma)

        # Compute V1, V2, V3

        V_tau = P_1 + P_2 - P_3

        V_0 = method_Broadie.barrier(S_0, K_1, H_1, r, 0, sigma, tau, T) \
              + method_Broadie.barrier(S_0, K_2, H_2, r, 0, sigma, tau, T) \
              - method_Broadie.barrier(S_0, K_3, H_3, r, 0, sigma, tau, T)

        loss = (V_0 - V_tau) * np.exp(-r * tau)

        loss_lr[i * n:(i + 1) * n] = np.mean(loss * weight, axis=0)

    return loss_lr


def cal_RRMSE(M, N, S_0, K, H, mu, sigma, r, tau, T, alpha, L0, seed):

    rng = np.random.default_rng(seed)

    loss = broadie_lr(M, N, S_0, K, H, mu, sigma, r, tau, T, rng.integers(0, 2147483647), rng.integers(0, 2147483647))
    indicator = np.mean((loss > L0))
    hockey = np.mean(np.maximum(loss - L0, 0))
    quadratic = np.mean((loss - L0) ** 2)

    loss.sort()
    VaR = loss[int(np.ceil((1 - alpha) * M))]
    CVaR = np.mean(loss[loss >= VaR])

    return indicator, hockey, quadratic, VaR, CVaR


local = False

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
    print("LR Estimation Done for:", M, "x", N)
    print(RRMSE)

print(result_table)
np.save(save_path + "result_155.npy", result_table)
