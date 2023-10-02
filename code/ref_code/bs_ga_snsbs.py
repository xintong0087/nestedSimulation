import numpy as np
import pandas as pd
import methods
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed


def SNS_Geo_Asian(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h):

    cor_mat = np.ones((d, d)) * 0.3 + np.identity(d) * 0.7
    cov_mat = cor_mat * sigma ** 2

    sample_outer = methods.GBM_front(n_front=n_front, d=d, S_0=S_0,
                                     drift_vec=mu, diffusion_mat=cov_mat, tau=tau,
                                     step_size=h, path=True)
    n_step_outer = sample_outer.shape[2] - 1
    geometric_sum_outer = np.prod(sample_outer[:, :, 1:], axis=2)

    sample_inner = methods.GBM_back(n_front=n_front, n_back=n_back, d=d, S_tau=sample_outer,
                                    drift=r, diffusion=cov_mat, T=T - tau,
                                    step_size=h, path=True)
    n_step_inner = sample_inner.shape[3] - 1
    geometric_sum_inner = np.prod(sample_inner[:, :, :, 1:], axis=3)

    geometric_average = np.zeros([d, n_front, n_back])
    for i in range(n_back):
        geometric_average[:, :, i] = (geometric_sum_outer * geometric_sum_inner[:, :, i]) ** (
                    1 / (n_step_outer + n_step_inner))

    payoff = np.zeros([d, n_front])
    for j in range(len(K)):
        price = np.mean(np.maximum(geometric_average - K[j], 0), axis=2) * np.exp(-r * (T - tau))
        payoff = payoff + price

    portfolio_value_0 = 0
    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_Asian_G(S=S_0, T=T, sigma=sigma, r=r, K=K[j],
                                                                      continuous=False,
                                                                      args=n_step_outer + n_step_inner)

    loss_SNS = d * portfolio_value_0 - np.sum(payoff, axis=0)

    return loss_SNS


def SNS_Geo_Asian_BootStrap(Gamma, n_front_vec, n_back_vec, d, S_0, K, mu, sigma, r, tau, T, h,
                            L0, I=500, alpha=0.1):
    cor_mat = np.ones((d, d)) * 0.3 + np.identity(d) * 0.7
    cov_mat = cor_mat * sigma ** 2

    n_front_0 = n_front_vec[-1]
    n_back_0 = n_back_vec[-1]

    sample_outer_0 = methods.GBM_front(n_front=n_front_0, d=d, S_0=S_0,
                                       drift_vec=mu, diffusion_mat=cov_mat, tau=tau,
                                       step_size=h, path=True)
    n_step_outer = sample_outer_0.shape[2] - 1
    geometric_sum_outer = np.prod(sample_outer_0[:, :, 1:], axis=2)

    sample_inner_0 = methods.GBM_back(n_front=n_front_0, n_back=n_back_0, d=d, S_tau=sample_outer_0,
                                      drift=r, diffusion=cov_mat, T=T - tau,
                                      step_size=h, path=True)
    n_step_inner = sample_inner_0.shape[3] - 1
    geometric_sum_inner = np.prod(sample_inner_0[:, :, :, 1:], axis=3)

    geometric_average = np.zeros([d, n_front_0, n_back_0])
    for i in range(n_back_0):
        geometric_average[:, :, i] = (geometric_sum_outer * geometric_sum_inner[:, :, i]) ** (
                    1 / (n_step_outer + n_step_inner))

    portfolio_value_0 = 0
    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_Asian_G(S=S_0, T=T, sigma=sigma, r=r, K=K[j],
                                                                      continuous=False,
                                                                      args=n_step_outer + n_step_inner)

    outer_shape = n_front_vec.shape[0]
    alpha_mat = np.zeros([outer_shape, 5])

    counter = 0
    for n_back in n_back_vec:

        res = np.zeros([I, 5])

        for i in range(I):

            index_outer = np.random.choice(n_front_0, size=n_front_0, replace=True)
            index_inner = np.random.choice(n_back_0, size=n_back, replace=True)
            temp = geometric_average[:, index_outer, :]
            geometric_average_bs = temp[:, :, index_inner]

            payoff = np.zeros([d, n_front_0])
            for j in range(len(K)):
                price = np.mean(np.maximum(geometric_average_bs - K[j], 0), axis=2) * np.exp(-r * (T - tau))
                payoff = payoff + price

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

            geometric_average_bs = geometric_average[:, np.random.choice(n_front_0, size=n_front, replace=True), :]

            payoff = np.zeros([d, n_front])
            for j in range(len(K)):
                price = np.mean(np.maximum(geometric_average_bs - K[j], 0), axis=2) * np.exp(-r * (T - tau))
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


def cal_RRMSE(Gamma, n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h, alpha, L0):

    n_front_opt, n_back_opt = SNS_Geo_Asian_BootStrap(Gamma, n_front, n_back,
                                                      d, S_0, K, mu, sigma, r, tau, T, h,
                                                      L0, 500, alpha)

    loss = SNS_Geo_Asian(n_front_opt[0], n_back_opt[0], d, S_0, K, mu, sigma, r, tau, T, h)
    indicator = np.nanmean((loss > L0))

    loss = SNS_Geo_Asian(n_front_opt[1], n_back_opt[1], d, S_0, K, mu, sigma, r, tau, T, h)
    hockey = np.nanmean(np.maximum(loss - L0, 0))

    loss = SNS_Geo_Asian(n_front_opt[2], n_back_opt[2], d, S_0, K, mu, sigma, r, tau, T, h)
    quadratic = np.nanmean((loss - L0) ** 2)

    loss = SNS_Geo_Asian(n_front_opt[3], n_back_opt[3], d, S_0, K, mu, sigma, r, tau, T, h)
    loss.sort()
    VaR = loss[int(np.ceil((1 - alpha) * n_front_opt[3]))]

    loss = SNS_Geo_Asian(n_front_opt[4], n_back_opt[4], d, S_0, K, mu, sigma, r, tau, T, h)
    loss.sort()
    Q = loss[int(np.ceil((1 - alpha) * n_front_opt[4]))]
    CVaR = np.nanmean(loss[loss >= Q])

    return indicator, hockey, quadratic, VaR, CVaR


np.random.seed(22)

d_12 = 20
sigma_12 = 0.3
S_0_12 = 100
K_12 = [90, 100, 110]
mu_12 = 0.08
r_12 = 0.05
T_12 = 1
h_12 = 1/50
tau_12 = h_12 * 3
alpha_12 = 0.1

result_true = np.array(pd.read_csv("data/trueValue_12.csv")).flatten()[1:]

L0_12 = result_true[3]

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

    Gamma_12 = Gamma[_j]

    res = Parallel(n_jobs=20, verbose=10)(delayed(cal_RRMSE)(Gamma_12, n_front_vec[_j], n_back_vec[_j],
                                                            d_12, S_0_12, K_12,
                                                            mu_12, sigma_12, r_12, tau_12, T_12,
                                                            h_12, alpha_12, L0_12)
                                          for _n in range(n_rep))
    res = np.array(res)

    Bias = np.nanmean(res, axis=0) - result_true
    Variance = np.nanvar(res, axis=0)

    MSE = Bias ** 2 + Variance
    RRMSE = np.sqrt(MSE) / result_true

    result_table[_j, :] = RRMSE
    print("SNS Estimation Done for:", Gamma_12)
    print(RRMSE)

print(result_table)
np.save("data/result_122.npy", result_table)
