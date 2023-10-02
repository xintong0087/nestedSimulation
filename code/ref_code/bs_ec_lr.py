import numpy as np
import pandas as pd
import methods
from joblib import Parallel, delayed


def lr_Euro_Call(n_front, n, d, S_0, K, mu, sigma, r, tau, T, h):

    cor_mat = np.ones((d, d)) * 0.3 + np.identity(d) * 0.7
    cov_mat = cor_mat * sigma**2

    sample_outer_train = methods.GBM_front(n_front=n_front, d=d,
                                   S_0=S_0, drift_vec=mu, diffusion_mat=cov_mat, tau=tau)
    sample_inner_tau = methods.GBM_back(n_front=n_front, n_back=1, d=d,
                                S_tau=sample_outer_train, drift=r, diffusion=cov_mat, T=h)[:, :, 0]
    sample_inner_T = methods.GBM_back(n_front=n_front, n_back=1, d=d,
                              S_tau=sample_inner_tau, drift=r, diffusion=cov_mat, T=T-tau-h)
    sample_outer_test = methods.GBM_front(n_front=n_front, d=d,
                                          S_0=S_0, drift_vec=mu, diffusion_mat=cov_mat, tau=tau)

    portfolio_value_0 = 0
    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_CP(S_0, T, sigma, r, K[j], 0, "C", "long")

    payoff = np.zeros([d, n_front])
    for j in range(len(K)):
        price = np.mean(np.maximum(sample_inner_T - K[j], 0), axis=2) * np.exp(-r * (T - tau))
        payoff = payoff + price

    diff = portfolio_value_0 - payoff
    Loss_LR = np.zeros(n_front)

    Weight_D = methods.compute_Weight_D(S_0, mu, r, sigma, h, tau, sample_inner_tau)

    n_partition = int(n_front // n)

    for i in range(n_partition):

        LR_Loss_Matrix = np.zeros([n_front, n])

        for j in range(d):

            Weight_U = methods.compute_Weight_U(sample_inner_tau[j, :], sample_outer_test[j, i * n:(i + 1) * n],
                                                r, sigma, h)

            Weight = np.sqrt((tau + h) / h) * np.exp(Weight_D[j, :].reshape(-1, 1) - Weight_U)

            LR_Loss_Matrix = LR_Loss_Matrix + diff[j, :].reshape(-1, 1) * Weight

        Loss_LR[i * n:(i + 1) * n] = np.mean(LR_Loss_Matrix, axis=0)

    return Loss_LR


def cal_RRMSE(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h, alpha, L0):

    loss = lr_Euro_Call(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h)

    indicator = np.mean((loss > L0))
    hockey = np.mean(np.maximum(loss - L0, 0))
    quadratic = np.mean((loss - L0) ** 2)

    loss.sort()
    VaR = loss[int(np.ceil((1 - alpha) * n_front))]
    CVaR = np.mean(loss[loss >= VaR])

    return indicator, hockey, quadratic, VaR, CVaR


np.random.seed(22)

d_11 = 20
sigma_11 = 0.1
S_0_11 = 100
K_11 = [90, 100, 110]
mu_11 = 0.08
r_11 = 0.05
T_11 = 1
h_11 = 1/50
tau_11 = h_11 * 3
alpha_11 = 0.1

result_true = np.array(pd.read_csv("data/trueValue_11.csv")).flatten()[1:]

L0_11 = result_true[3]
n_front_list = []
for _i in range(2):
    n_front_list = n_front_list + list(np.arange(1, 10) * 10 ** (_i + 3))
n_front_list = n_front_list + [10 ** (_i + 4)]
n_front_vec = np.array(n_front_list)
n_trials = len(n_front_list)
n_vec = [10] * n_trials

result_table = np.zeros([n_trials, 5])

n_rep = 150

for _j in range(n_trials):

    n_front_11 = n_front_vec[_j]
    n_11 = n_vec[_j]

    res = Parallel(n_jobs=20, verbose=10)(delayed(cal_RRMSE)(n_front_11, n_11, d_11, S_0_11, K_11,
                                                         mu_11, sigma_11, r_11, tau_11, T_11, h_11,
                                                         alpha_11, L0_11)
                                          for _n in range(n_rep))
    res = np.array(res)

    Bias = np.mean(res, axis=0) - result_true
    Variance = np.var(res, axis=0)

    MSE = Bias ** 2 + Variance
    RRMSE = np.sqrt(MSE) / result_true

    result_table[_j, :] = RRMSE
    print("LR Estimation Done for:", n_front_11, "x", 1)
    print(RRMSE)

print(result_table)
np.save("data/result_115.npy", result_table)
