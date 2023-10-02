import numpy as np
import pandas as pd
import methods
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed


def SNS_Euro_Call(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T):

    cor_mat = np.ones((d, d)) * 0.3 + np.identity(d) * 0.7
    cov_mat = cor_mat * sigma**2

    sample_outer = methods.GBM_front(n_front = n_front, d = d, S_0 = S_0,
                                     drift_vec = mu, diffusion_mat = cov_mat, tau = tau)
    sample_inner = methods.GBM_back(n_front = n_front, n_back = n_back, d = d, S_tau = sample_outer,
                                    drift = r, diffusion = cov_mat, T = T - tau)

    payoff = np.zeros([d, n_front])
    for j in range(len(K)):
        price = np.mean(np.maximum(sample_inner - K[j], 0), axis=2) * np.exp(-r * (T - tau))
        payoff = payoff + price

    portfolio_value_0 = 0
    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_CP(S_0, T, sigma, r, K[j], 0, "C", "long")

    loss_SNS = d * portfolio_value_0 - np.sum(payoff, axis=0)

    return loss_SNS


def SNS_Euro_Call_BootStrap(Gamma, n_front_vec, n_back_vec, d, S_0, K, mu, sigma, r, tau, T,
                            L0, I=500, alpha=0.1):
    cor_mat = np.ones((d, d)) * 0.3 + np.identity(d) * 0.7
    cov_mat = cor_mat * sigma ** 2

    portfolio_value_0 = 0
    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_CP(S_0, T, sigma, r, K[j], 0, "C", "long")

    n_front_0 = n_front_vec[-1]
    n_back_0 = n_back_vec[-1]

    sample_outer_0 = methods.GBM_front(n_front=n_front_0, d=d, S_0=S_0,
                                       drift_vec=mu, diffusion_mat=cov_mat, tau=tau)

    sample_inner_0 = methods.GBM_back(n_front=n_front_0, n_back=n_back_0, d=d, S_tau=sample_outer_0,
                                      drift=r, diffusion=cov_mat, T=T - tau)

    outer_shape = n_front_vec.shape[0]
    alpha_mat = np.zeros([outer_shape, 5])

    counter = 0
    for n_back in n_back_vec:

        res = np.zeros([I, 5])

        for i in range(I):

            index_outer = np.random.choice(n_front_0, size=n_front_0, replace=True)
            index_inner = np.random.choice(n_back_0, size=n_back, replace=True)
            sample_outer_bs = sample_inner_0[:, index_outer, :]
            sample_inner_bs = sample_outer_bs[:, :, index_inner]

            payoff = np.zeros([d, n_front_0])
            for j in range(len(K)):
                price = np.mean(np.maximum(sample_inner_bs - K[j], 0), axis=2) * np.exp(-r * (T - tau))
                payoff = payoff + price

            loss_bs = d * portfolio_value_0 - np.sum(payoff, axis=0)

            res[i, 0] = np.nanmean((loss_bs > L0))
            res[i, 1] = np.nanmean(np.maximum(loss_bs - L0, 0))
            res[i, 2] = np.nanmean((loss_bs - L0) ** 2)

            loss_bs.sort()
            res[i, 3] = loss_bs[int(np.ceil((1 - alpha) * n_front_0))]
            res[i, 4] = np.nanmean(loss_bs[loss_bs >= res[i, 3]])

        alpha_mat[counter, :] = np.mean(res, axis=0)
        counter = counter + 1

    inner_shape = n_back_vec.shape[0]
    s_mat = np.zeros([inner_shape, 5])

    counter = 0
    for n_front in n_front_vec:

        res = np.zeros([I, 5])

        for i in range(I):

            sample_inner_bs = sample_inner_0[:, np.random.choice(n_front_0, size=n_front, replace=True), :]

            payoff = np.zeros([d, n_front])
            for j in range(len(K)):
                price = np.mean(np.maximum(sample_inner_bs - K[j], 0), axis=2) * np.exp(-r * (T - tau))
                payoff = payoff + price

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


def cal_RRMSE(Gamma, n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, alpha, L0):

    n_front_opt, n_back_opt = SNS_Euro_Call_BootStrap(Gamma, n_front, n_back,
                                                      d, S_0, K, mu, sigma, r, tau, T,
                                                      L0, 500, alpha)

    loss = SNS_Euro_Call(n_front_opt[0], n_back_opt[0], d, S_0, K, mu, sigma, r, tau, T)
    indicator = np.mean((loss > L0))

    loss = SNS_Euro_Call(n_front_opt[1], n_back_opt[1], d, S_0, K, mu, sigma, r, tau, T)
    hockey = np.mean(np.maximum(loss - L0, 0))

    loss = SNS_Euro_Call(n_front_opt[2], n_back_opt[2], d, S_0, K, mu, sigma, r, tau, T)
    quadratic = np.mean((loss - L0) ** 2)

    loss = SNS_Euro_Call(n_front_opt[3], n_back_opt[3], d, S_0, K, mu, sigma, r, tau, T)
    loss.sort()
    VaR = loss[int(np.ceil((1 - alpha) * n_front_opt[3]))]

    loss = SNS_Euro_Call(n_front_opt[4], n_back_opt[4], d, S_0, K, mu, sigma, r, tau, T)
    loss.sort()
    Q = loss[int(np.ceil((1 - alpha) * n_front_opt[4]))]
    CVaR = np.mean(loss[loss >= Q])

    return indicator, hockey, quadratic, VaR, CVaR


np.random.seed(22)

d_11 = 20
sigma_11 = 0.1
S_0_11 = 100
K_11 = [90, 100, 110]
mu_11 = 0.08
r_11 = 0.05
tau_11 = 3/50
T_11 = 1
alpha_11 = 0.1

result_true = np.array(pd.read_csv("data/trueValue_11.csv")).flatten()[1:]

L0_11 = result_true[3]

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

    Gamma_11 = Gamma[_j]

    res = Parallel(n_jobs=20, verbose=10)(delayed(cal_RRMSE)(Gamma_11, n_front_vec[_j], n_back_vec[_j],
                                                            d_11, S_0_11, K_11,
                                                            mu_11, sigma_11, r_11, tau_11, T_11,
                                                            alpha_11, L0_11)
                                          for _n in range(n_rep))

    Bias = np.nanmean(res, axis=0) - result_true
    Variance = np.nanvar(res, axis=0)

    MSE = Bias ** 2 + Variance
    RRMSE = np.sqrt(MSE) / result_true

    result_table[_j, :] = RRMSE
    print("SNS Estimation Done for:", Gamma_11)
    print(RRMSE)

print(result_table)
np.save("data/result_112.npy", result_table)
