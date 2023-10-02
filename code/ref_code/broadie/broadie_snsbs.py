import numpy as np
import pandas as pd
import method_Broadie
from sklearn.linear_model import LinearRegression
import time
import os
from joblib import Parallel, delayed


def broadie_SNS(M, N, S_0, K, H, mu, sigma, r, tau, T, seed):

    crude = False
    loss_flag = True
    X, loss = method_Broadie.simPredictors(M, N, S_0, K, H, mu, sigma, r, tau, T, seed, crude, loss_flag)

    return loss


def SNS_bootstrap(Gamma, M_vec, N_vec, S_0, K, H, mu, sigma, r, tau, T, alpha, L0, I=300, seed=22):

    K_1, K_2, K_3 = K[0], K[1], K[2]
    H_1, H_2, H_3 = H[0], H[1], H[2]
    t = T - tau

    M_0 = M_vec[-1]
    N_0 = N_vec[-1]

    crude = True
    loss_flag = False
    S_tau, S_T = method_Broadie.simPredictors(M_0, N_0, S_0, K, H, mu, sigma, r, tau, T, seed, crude, loss_flag)

    M_shape = M_vec.shape[0]
    a_mat = np.zeros([M_shape, 5])

    counter = 0
    for N in N_vec:

        res = np.zeros([I, 5])

        for i in range(I):

            index_M = np.random.choice(M_0, size=M_0, replace=True)
            index_N = np.random.choice(N_0, size=N, replace=True)

            sample_M = S_tau[index_M]
            sample = S_T[index_M, :]
            sample_N = sample[:, index_N]

            # Computing P1, P2, P3
            P_1 = method_Broadie.compute_P(M_0, N, K_1, H_1, sample_N, sample_M, t, r, sigma)
            P_2 = method_Broadie.compute_P(M_0, N, K_2, H_2, sample_N, sample_M, t, r, sigma)
            P_3 = method_Broadie.compute_P(M_0, N, K_3, H_3, sample_N, sample_M, t, r, sigma)

            # Compute V1, V2, V3

            V_tau = P_1 + P_2 - P_3

            V_0 = method_Broadie.barrier(S_0, K_1, H_1, r, 0, sigma, tau, T) \
                  + method_Broadie.barrier(S_0, K_2, H_2, r, 0, sigma, tau, T) \
                  - method_Broadie.barrier(S_0, K_3, H_3, r, 0, sigma, tau, T)

            loss_bs = (V_0 - V_tau) * np.exp(-r * tau)

            res[i, 0] = np.nanmean((loss_bs > L0))
            res[i, 1] = np.nanmean(np.maximum(loss_bs - L0, 0))
            res[i, 2] = np.nanmean((loss_bs - L0) ** 2)

            loss_bs.sort()
            res[i, 3] = loss_bs[int(np.ceil((1 - alpha) * M_0))]
            res[i, 4] = np.nanmean(loss_bs[loss_bs >= res[i, 3]])

        a_mat[counter, :] = np.mean(res, axis=0)
        counter = counter + 1

    N_shape = N_vec.shape[0]
    s_mat = np.zeros([N_shape, 5])

    counter = 0
    for M in M_vec:

        res = np.zeros([I, 5])

        for i in range(I):

            index_M = np.random.choice(M_0, size=M, replace=True)
            sample_M = S_tau[index_M]
            sample_N = S_T[index_M, :]

            # Computing P1, P2, P3
            P_1 = method_Broadie.compute_P(M, N_0, K_1, H_1, sample_N, sample_M, t, r, sigma)
            P_2 = method_Broadie.compute_P(M, N_0, K_2, H_2, sample_N, sample_M, t, r, sigma)
            P_3 = method_Broadie.compute_P(M, N_0, K_3, H_3, sample_N, sample_M, t, r, sigma)

            # Compute V1, V2, V3

            V_tau = P_1 + P_2 - P_3

            V_0 = method_Broadie.barrier(S_0, K_1, H_1, r, 0, sigma, tau, T) \
                  + method_Broadie.barrier(S_0, K_2, H_2, r, 0, sigma, tau, T) \
                  - method_Broadie.barrier(S_0, K_3, H_3, r, 0, sigma, tau, T)

            loss_bs = (V_0 - V_tau) * np.exp(-r * tau)

            res[i, 0] = np.nanmean((loss_bs > L0))
            res[i, 1] = np.nanmean(np.maximum(loss_bs - L0, 0))
            res[i, 2] = np.nanmean((loss_bs - L0) ** 2)

            loss_bs.sort()
            res[i, 3] = loss_bs[int(np.ceil((1 - alpha) * M))]
            res[i, 4] = np.nanmean(loss_bs[loss_bs >= res[i, 3]])

        s_mat[counter, :] = np.var(res, axis=0)
        counter = counter + 1

    M_opt = np.zeros(5)
    N_opt = np.zeros(5)

    for i in range(5):
        reg_A = LinearRegression().fit(1 / N_vec.reshape(-1, 1), a_mat[:, i])
        A = reg_A.coef_[0]
        reg_B = LinearRegression().fit(1 / M_vec.reshape(-1, 1), s_mat[:, i])
        B = reg_B.coef_[0]

        M_opt[i] = int((B / (2 * A ** 2)) ** (1 / 3) * Gamma ** (2 / 3))
        N_opt[i] = int(((2 * A ** 2) / B) ** (1 / 3) * Gamma ** (1 / 3))


    return M_opt.astype(int), N_opt.astype(int)


def cal_RRMSE(Gamma, M_vec, N_vec, S_0, K, H, mu, sigma, r, tau, T, alpha, L0, seed):

    rng = np.random.default_rng(seed)

    M, N = SNS_bootstrap(Gamma, M_vec, N_vec, S_0, K, H, mu, sigma, r, tau, T, alpha, L0, 300,
                         rng.integers(0, 2147483647))

    loss = broadie_SNS(M[0], N[0], S_0, K, H, mu, sigma, r, tau, T, rng.integers(0, 2147483647))
    indicator = np.mean((loss > L0))

    loss = broadie_SNS(M[1], N[1], S_0, K, H, mu, sigma, r, tau, T, rng.integers(0, 2147483647))
    hockey = np.mean(np.maximum(loss - L0, 0))

    loss = broadie_SNS(M[2], N[2], S_0, K, H, mu, sigma, r, tau, T, rng.integers(0, 2147483647))
    quadratic = np.mean((loss - L0) ** 2)

    loss = broadie_SNS(M[3], N[3], S_0, K, H, mu, sigma, r, tau, T, rng.integers(0, 2147483647))
    loss.sort()
    VaR = loss[int(np.ceil((1 - alpha) * M[3])) - 1]

    loss = broadie_SNS(M[4], N[4], S_0, K, H, mu, sigma, r, tau, T, rng.integers(0, 2147483647))
    loss.sort()
    Q = loss[int(np.ceil((1 - alpha) * M[4])) - 1]
    CVaR = np.mean(loss[loss >= Q])

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

Gamma_list = []
for i in range(2):
    Gamma_list = Gamma_list + list(np.arange(1, 10) * 10 ** (i + 3))
Gamma_list = Gamma_list + [10 ** (i + 4)]
Gamma_vec = np.array(Gamma_list)
n_trials = len(Gamma_list)

M_vec = [np.arange(50, 101, 5)] * n_trials
N_vec = [np.arange(50, 101, 5)] * n_trials
result_table = np.zeros([n_trials, 5])

start = time.time()

n_rep = 150

seed = np.random.randint(low=0, high=2147483647, size=n_rep)

for j in range(n_trials):

    Gamma = Gamma_vec[j]

    res = Parallel(n_jobs=n_cores, verbose=10)(delayed(cal_RRMSE)(Gamma, M_vec[j], N_vec[j],
                                                                  S_0, K, H, mu, sigma, r, tau, T, alpha, L0, seed[n])
                                               for n in range(n_rep))

    Bias = np.nanmean(res, axis=0) - result_true
    Variance = np.nanvar(res, axis=0)

    MSE = Bias ** 2 + Variance
    RRMSE = np.sqrt(MSE) / result_true

    result_table[j, :] = RRMSE
    print("SNS Estimation Done for:", Gamma)
    print(RRMSE)

print(result_table)
np.save(save_path + "result_152.npy", result_table)
