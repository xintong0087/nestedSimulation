import numpy as np
import pandas as pd
import methods
from joblib import Parallel, delayed
import time


def lr_Barrier_U(n_front, n, d, S_0, K, mu, sigma, r, tau, T, h, U):

    n_back = 1
    cor_mat = np.ones((d, d)) * 0.3 + np.identity(d) * 0.7
    cov_mat = cor_mat * sigma ** 2

    start_sim = time.time()
    portfolio_value_0 = 0
    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_barrier([S_0], K[j], T, sigma, r, U, 0, 0,
                                                                      option_type=np.array([["O", "U", "C"]]))

    sample_outer_train = methods.GBM_front(n_front=n_front, d=d, S_0=S_0,
                                   drift_vec=mu, diffusion_mat=cov_mat, tau=tau,
                                   step_size=h, path=True)
    n_step_outer = sample_outer_train.shape[2] - 1

    outer_prob_knock_out = np.random.uniform(low=0.0, high=1.0,
                                             size=[d, n_front, n_step_outer])
    sample_outer_max = np.zeros_like(outer_prob_knock_out)

    for n in range(n_step_outer):
        sample_outer_max[:, :, n] = np.exp((np.log(sample_outer_train[:, :, n] * sample_outer_train[:, :, n + 1])
                                            + np.sqrt(
                    np.log(sample_outer_train[:, :, n + 1] / sample_outer_train[:, :, n]) ** 2
                    - 2 * sigma ** 2 * h * np.log(outer_prob_knock_out[:, :, n]))) / 2)

    sample_inner = methods.GBM_back(n_front=n_front, n_back=1, d=d, S_tau=sample_outer_train,
                            drift=r, diffusion=cov_mat, T=T - tau,
                            step_size=h, path=True)

    n_step_inner = sample_inner.shape[3] - 1

    sample_inner_tau = sample_inner[:, :, :, 1].reshape([d, n_front])
    sample_inner_T = sample_inner[:, :, :, -1].reshape([d, n_front])

    inner_prob_knock_out = np.random.uniform(low=0.0, high=1.0,
                                             size=[d, n_front, n_back, n_step_inner])
    sample_inner_max = np.zeros_like(inner_prob_knock_out)

    for i in range(n_back):
        for n in range(n_step_inner):
            sample_inner_max[:, :, i, n] = np.exp((np.log(sample_inner[:, :, i, n] * sample_inner[:, :, i, n + 1])
                                                   + np.sqrt(
                        np.log(sample_inner[:, :, i, n + 1] / sample_inner[:, :, i, n]) ** 2
                        - 2 * sigma ** 2 * h * np.log(inner_prob_knock_out[:, :, i, n]))) / 2)

    max_inner = np.max(sample_inner_max, axis=3)
    time_sim = time.time() - start_sim

    start_pred = time.time()
    sample_outer_test = methods.GBM_front(n_front=n_front, d=d, S_0=S_0,
                                  drift_vec=mu, diffusion_mat=cov_mat, tau=tau,
                                  step_size=h, path=True)
    sample_outer_test_tau = sample_outer_test[:, :, -1]

    n_step_outer = sample_outer_test.shape[2] - 1

    outer_prob_knock_out = np.random.uniform(low=0.0, high=1.0,
                                             size=[d, n_front, n_step_outer])
    sample_outer_max = np.zeros_like(outer_prob_knock_out)

    for n in range(n_step_outer):
        sample_outer_max[:, :, n] = np.exp((np.log(sample_outer_test[:, :, n] * sample_outer_test[:, :, n + 1])
                                            + np.sqrt(
                    np.log(sample_outer_test[:, :, n + 1] / sample_outer_test[:, :, n]) ** 2
                    - 2 * sigma ** 2 * h * np.log(outer_prob_knock_out[:, :, n]))) / 2)

    max_outer_test = np.max(sample_outer_max, axis=2)

    max_inner = max_inner.reshape([d, n_front])

    Weight_D = methods.compute_Weight_D(S_0, mu, r, sigma, h, tau, sample_inner_tau)

    Loss_LR = np.zeros(n_front)
    n_partition = int(n_front // n)
    for i in range(n_partition):

        LR_Loss_Matrix = np.zeros([n_front, n])

        for j in range(d):

            payoff = np.zeros([n_front, n])

            for k in range(len(K)):
                price = (max_outer_test[j, i * n:(i + 1) * n] <= U) \
                        * (np.maximum(sample_inner_T[j, :] - K[k], 0) * (max_inner[j, :] <= U)).reshape(-1, 1) \
                        * np.exp(-r * (T - tau))
                payoff = payoff + price

            diff = portfolio_value_0[0] - payoff

            Weight_U = methods.compute_Weight_U(sample_inner_tau[j, :], sample_outer_test_tau[j, i * n:(i + 1) * n],
                                        r, sigma, h)

            Weight = np.sqrt((tau + h) / h) * np.exp(Weight_D[j, :].reshape(-1, 1) - Weight_U)

            LR_Loss_Matrix = LR_Loss_Matrix + diff * Weight

        Loss_LR[i * n:(i + 1) * n] = np.mean(LR_Loss_Matrix, axis=0)
    time_pred = time.time() - start_pred

    time_sim_pred = 0
    time_fit = 0

    return Loss_LR, time_sim, time_sim_pred, time_fit, time_pred


def cal_RRMSE(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h, U, alpha, L0):

    loss, time_sim, time_sim_pred, time_fit, time_pred = lr_Barrier_U(n_front, n_back,
                                                                      d, S_0, K, mu, sigma, r, tau, T, h, U)

    indicator = np.mean((loss > L0))
    hockey = np.mean(np.maximum(loss - L0, 0))
    quadratic = np.mean((loss - L0) ** 2)

    loss.sort()
    VaR = loss[int(np.ceil((1 - alpha) * n_front))]
    CVaR = np.mean(loss[loss >= VaR])

    return time_sim, time_sim_pred, time_fit, time_pred


np.random.seed(22)

d_13 = 20
sigma_13 = 0.2
S_0_13 = 100
K_13 = [90, 100, 110]
mu_13 = 0.08
r_13 = 0.05
T_13 = 1
h_13 = 1/50
tau_13 = h_13 * 3
alpha_13 = 0.1
U_13 = 120

result_true = np.array(pd.read_csv("data/trueValue_13.csv")).flatten()[1:]

L0_13 = result_true[3]

n_front_list = []
for _i in range(1):
    n_front_list = n_front_list + list(np.arange(1, 10) * 10 ** (_i + 3))
n_front_list = n_front_list + [10 ** (_i + 4)]
n_front_vec = np.array(n_front_list)
n_trials = len(n_front_list)
n_back_vec = [1] * n_trials

result_table = np.zeros([n_trials, 5])

n_rep = 150

for _j in range(n_trials):

    n_front_13 = n_front_vec[_j]
    n_back_13 = n_back_vec[_j]

    time_cv = 0

    res = Parallel(n_jobs=20, verbose=10)(delayed(cal_RRMSE)(n_front_13, n_back_13, d_13, S_0_13, K_13,
                                                         mu_13, sigma_13, r_13, tau_13, T_13,
                                                         h_13, U_13, alpha_13, L0_13)
                                          for _n in range(n_rep))

    res = np.array(res)

    time_vec = np.mean(res, axis=0)

    result_table[_j, :4] = time_vec
    result_table[_j, 4] = time_cv

    print("LR Estimation Done for:", n_front_13, "x", 1)
    print(time_vec, time_cv)

print(result_table)
np.save("data/result_135_time.npy", result_table)
