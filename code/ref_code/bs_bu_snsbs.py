import numpy as np
import pandas as pd
import methods
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed


def SNS_Barrier_U(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h, H):
    cor_mat = np.ones((d, d)) * 0.3 + np.identity(d) * 0.7
    cov_mat = cor_mat * sigma ** 2

    sample_outer = methods.GBM_front(n_front=n_front, d=d, S_0=S_0,
                                     drift_vec=mu, diffusion_mat=cov_mat, tau=tau,
                                     step_size=h, path=True)
    n_step_outer = sample_outer.shape[2] - 1

    outer_prob_knock_out = np.random.uniform(low=0.0, high=1.0,
                                             size=[d, n_front, n_step_outer])
    sample_outer_max = np.zeros_like(outer_prob_knock_out)

    for n in range(n_step_outer):
        sample_outer_max[:, :, n] = np.exp((np.log(sample_outer[:, :, n] * sample_outer[:, :, n + 1])
                                            + np.sqrt(np.log(sample_outer[:, :, n + 1] / sample_outer[:, :, n]) ** 2
                                                      - 2 * sigma ** 2 * h * np.log(
                    outer_prob_knock_out[:, :, n]))) / 2)

    max_outer = np.max(sample_outer_max, axis=2)

    sample_inner = methods.GBM_back(n_front=n_front, n_back=n_back, d=d, S_tau=sample_outer,
                                    drift=r, diffusion=cov_mat, T=T - tau,
                                    step_size=h, path=True)
    n_step_inner = sample_inner.shape[3] - 1

    inner_prob_knock_out = np.random.uniform(low=0.0, high=1.0,
                                             size=[d, n_front, n_back, n_step_inner])
    sample_inner_max = np.zeros_like(inner_prob_knock_out)

    for i in range(n_back):
        for n in range(n_step_inner):
            sample_inner_max[:, :, i, n] = np.exp((np.log(sample_inner[:, :, i, n] * sample_inner[:, :, i, n + 1])
                                                   + np.sqrt(
                        np.log(sample_inner[:, :, i, n + 1] / sample_inner[:, :, i, n]) ** 2
                        - 2 * sigma ** 2 * h * np.log(inner_prob_knock_out[:, :, i, n]))) / 2)

    max_outer = np.repeat(max_outer[:, :, np.newaxis], n_back, axis=2)
    max_inner = np.max(sample_inner_max, axis=3)

    payoff = np.zeros([d, n_front])
    S_T = sample_inner[:, :, :, -1]

    for j in range(len(K)):
        payoff = payoff + np.mean((max_outer < H) * (max_inner < H)
                                  * np.maximum(S_T - K[j], 0), axis=2) * np.exp(-r * (T - tau))

    portfolio_value_0 = 0
    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_barrier([S_0], K[j], T, sigma, r, H, 0, 0,
                                                                      option_type=np.array([["O", "U", "C"]]))

    loss_SNS = d * portfolio_value_0 - np.sum(payoff, axis=0)

    return loss_SNS


def SNS_Barrier_U_BootStrap(Gamma, n_front_vec, n_back_vec, d, S_0, K, mu, sigma, r, tau, T, h, H,
                            L0, I=500, alpha=0.1):
    cor_mat = np.ones((d, d)) * 0.3 + np.identity(d) * 0.7
    cov_mat = cor_mat * sigma ** 2

    n_front_0 = n_front_vec[-1]
    n_back_0 = n_back_vec[-1]

    sample_outer = methods.GBM_front(n_front=n_front_0, d=d, S_0=S_0,
                                     drift_vec=mu, diffusion_mat=cov_mat, tau=tau,
                                     step_size=h, path=True)
    n_step_outer = sample_outer.shape[2] - 1

    outer_prob_knock_out = np.random.uniform(low=0.0, high=1.0,
                                             size=[d, n_front_0, n_step_outer])
    sample_outer_max = np.zeros_like(outer_prob_knock_out)

    for n in range(n_step_outer):
        sample_outer_max[:, :, n] = np.exp((np.log(sample_outer[:, :, n] * sample_outer[:, :, n + 1])
                                            + np.sqrt(np.log(sample_outer[:, :, n + 1] / sample_outer[:, :, n]) ** 2
                                                      - 2 * sigma ** 2 * h * np.log(
                    outer_prob_knock_out[:, :, n]))) / 2)

    max_outer = np.max(sample_outer_max, axis=2)

    sample_inner = methods.GBM_back(n_front=n_front_0, n_back=n_back_0, d=d, S_tau=sample_outer,
                                    drift=r, diffusion=cov_mat, T=T - tau,
                                    step_size=h, path=True)
    n_step_inner = sample_inner.shape[3] - 1

    inner_prob_knock_out = np.random.uniform(low=0.0, high=1.0,
                                             size=[d, n_front_0, n_back_0, n_step_inner])
    sample_inner_max = np.zeros_like(inner_prob_knock_out)

    for i in range(n_back_0):
        for n in range(n_step_inner):
            sample_inner_max[:, :, i, n] = np.exp((np.log(sample_inner[:, :, i, n] * sample_inner[:, :, i, n + 1])
                                                   + np.sqrt(
                        np.log(sample_inner[:, :, i, n + 1] / sample_inner[:, :, i, n]) ** 2
                        - 2 * sigma ** 2 * h * np.log(inner_prob_knock_out[:, :, i, n]))) / 2)

    max_outer = np.repeat(max_outer[:, :, np.newaxis], n_back_0, axis=2)
    max_inner = np.max(sample_inner_max, axis=3)

    S_T = sample_inner[:, :, :, -1]

    portfolio_value_0 = 0
    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_barrier([S_0], K[j], T, sigma, r, H, 0, 0,
                                                                      option_type=np.array([["O", "U", "C"]]))

    outer_shape = n_front_vec.shape[0]
    alpha_mat = np.zeros([outer_shape, 5])

    counter = 0
    for n_back in n_back_vec:

        res = np.zeros([I, 5])

        for i in range(I):

            index_outer = np.random.choice(n_front_0, size=n_front_0, replace=True)
            index_inner = np.random.choice(n_back_0, size=n_back, replace=True)
            temp = max_outer[:, index_outer, :]
            max_outer_bs = temp[:, :, index_inner]
            temp = max_inner[:, index_outer, :]
            max_inner_bs = temp[:, :, index_inner]
            temp = S_T[:, index_outer, :]
            S_T_bs = temp[:, :, index_inner]

            payoff = np.zeros([d, n_front_0])
            for j in range(len(K)):
                payoff = payoff + np.mean((max_outer_bs < H) * (max_inner_bs < H)
                                          * np.maximum(S_T_bs - K[j], 0), axis=2) * np.exp(-r * (T - tau))

            loss_bs = d * portfolio_value_0 - np.sum(payoff, axis=0)

            res[i, 0] = np.mean((loss_bs > L0))
            res[i, 1] = np.mean(np.maximum(loss_bs - L0, 0))
            res[i, 2] = np.mean((loss_bs - L0) ** 2)

            loss_bs.sort()
            res[i, 3] = loss_bs[int(np.ceil((1 - alpha) * n_front_0))]
            res[i, 4] = np.mean(loss_bs[loss_bs >= res[i, 3]])

        alpha_mat[counter, :] = np.mean(res, axis=0)
        counter = counter + 1

    inner_shape = n_back_vec.shape[0]
    s_mat = np.zeros([inner_shape, 5])

    counter = 0
    for n_front in n_front_vec:

        res = np.zeros([I, 5])

        for i in range(I):

            index_outer = np.random.choice(n_front_0, size=n_front, replace=True)
            max_outer_bs = max_outer[:, index_outer, :]
            max_inner_bs = max_inner[:, index_outer, :]
            S_T_bs = S_T[:, index_outer, :]

            payoff = np.zeros([d, n_front])
            for j in range(len(K)):
                payoff = payoff + np.mean((max_outer_bs < H) * (max_inner_bs < H)
                                          * np.maximum(S_T_bs - K[j], 0), axis=2) * np.exp(-r * (T - tau))

            loss_bs = d * portfolio_value_0 - np.sum(payoff, axis=0)

            res[i, 0] = np.nanmean((loss_bs > L0))
            res[i, 1] = np.nanmean(np.maximum(loss_bs - L0, 0))
            res[i, 2] = np.nanmean((loss_bs - L0) ** 2)

            loss_bs.sort()
            res[i, 3] = loss_bs[int(np.ceil((1 - alpha) * n_front))]
            res[i, 4] = np.nanmean(loss_bs[loss_bs >= res[i, 3]])

        s_mat[counter, :] = np.var(res, axis=0)
        counter = counter + 1

    n_front_opt = np.zeros(5)
    n_back_opt = np.zeros(5)

    for i in range(5):
        reg_A = LinearRegression().fit(1 / n_back_vec.reshape(-1, 1), alpha_mat[:, i])
        A = reg_A.coef_[0]
        reg_B = LinearRegression().fit(1 / n_front_vec.reshape(-1, 1), s_mat[:, i])
        B = reg_B.coef_[0]

        n_front_opt[i] = int((B / (2 * A ** 2)) ** (1 / 3) * Gamma ** (2 / 3))
        n_back_opt[i] = int(((2 * A ** 2) / B) ** (1 / 3) * Gamma ** (1 / 3))

    return n_front_opt.astype(int), n_back_opt.astype(int)


def cal_RRMSE(Gamma, n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h, H, alpha, L0):

    n_front_opt, n_back_opt = SNS_Barrier_U_BootStrap(Gamma, n_front, n_back,
                                                      d, S_0, K, mu, sigma, r, tau, T, h, H,
                                                      L0, 500, alpha)

    loss = SNS_Barrier_U(n_front_opt[0], n_back_opt[0], d, S_0, K, mu, sigma, r, tau, T, h, H)
    indicator = np.nanmean((loss > L0))

    loss = SNS_Barrier_U(n_front_opt[1], n_back_opt[1], d, S_0, K, mu, sigma, r, tau, T, h, H)
    hockey = np.nanmean(np.maximum(loss - L0, 0))

    loss = SNS_Barrier_U(n_front_opt[2], n_back_opt[2], d, S_0, K, mu, sigma, r, tau, T, h, H)
    quadratic = np.nanmean((loss - L0) ** 2)

    loss = SNS_Barrier_U(n_front_opt[3], n_back_opt[3], d, S_0, K, mu, sigma, r, tau, T, h, H)
    loss.sort()
    VaR = loss[int(np.ceil((1 - alpha) * n_front_opt[3]))]

    loss = SNS_Barrier_U(n_front_opt[4], n_back_opt[4], d, S_0, K, mu, sigma, r, tau, T, h, H)
    loss.sort()
    Q = loss[int(np.ceil((1 - alpha) * n_front_opt[4]))]
    CVaR = np.nanmean(loss[loss >= Q])

    return indicator, hockey, quadratic, VaR, CVaR


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

Gamma_list = []
for i in range(2):
    Gamma_list = Gamma_list + list(np.arange(1, 10) * 10 ** (i + 3))
Gamma_list = Gamma_list + [10 ** (i + 4)]
Gamma = np.array(Gamma_list)
n_trials = len(Gamma)

n_front_vec = [np.arange(50, 101, 5)] * n_trials
n_back_vec = [np.arange(50, 101, 5)] * n_trials
result_table = np.zeros([n_trials, 5])

n_rep = 150

for _j in range(n_trials):

    Gamma_13 = Gamma[_j]

    res = Parallel(n_jobs=20, verbose=10)(delayed(cal_RRMSE)(Gamma_13, n_front_vec[_j], n_back_vec[_j],
                                                            d_13, S_0_13, K_13,
                                                            mu_13, sigma_13, r_13, tau_13, T_13,
                                                            h_13, U_13, alpha_13, L0_13)
                                          for _n in range(n_rep))
    res = np.array(res)

    Bias = np.mean(res, axis=0) - result_true
    Variance = np.var(res, axis=0)

    MSE = Bias ** 2 + Variance
    RRMSE = np.sqrt(MSE) / result_true

    result_table[_j, :] = RRMSE
    print("SNS Estimation Done for:", Gamma_13)
    print(RRMSE)

print(result_table)
np.save("data/result_132.npy", result_table)