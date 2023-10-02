import numpy as np
import pandas as pd
import methods
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


def cal_RRMSE(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h, alpha, L0):

    loss = SNS_Geo_Asian(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h)

    indicator = np.mean((loss > L0))
    hockey = np.mean(np.maximum(loss - L0, 0))
    quadratic = np.mean((loss - L0) ** 2)

    loss.sort()
    VaR = loss[int(np.ceil((1 - alpha) * n_front))]
    CVaR = np.mean(loss[loss >= VaR])

    return indicator, hockey, quadratic, VaR, CVaR


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

n_front_vec = [10, 20, 40, 50,
               50, 100, 200, 400,
               200, 400, 1000, 2000]
n_back_vec = [100, 50, 25, 20,
              200, 100, 50, 25,
              500, 250, 100, 50]
n_trials = len(n_front_vec)
result_table = np.zeros([n_trials, 5])

n_rep = 150

for _j in range(n_trials):

    n_front_12 = n_front_vec[_j]
    n_back_12 = n_back_vec[_j]

    res = Parallel(n_jobs=75, verbose=10)(delayed(cal_RRMSE)(n_front_12, n_back_12, d_12, S_0_12, K_12,
                                                        mu_12, sigma_12, r_12, tau_12, T_12,
                                                        h_12, alpha_12, L0_12)
                                          for _n in range(n_rep))
    res = np.array(res)

    Bias = np.mean(res, axis=0) - result_true
    Variance = np.var(res, axis=0)

    MSE = Bias ** 2 + Variance
    RRMSE = np.sqrt(MSE) / result_true

    result_table[_j, :] = RRMSE
    print("SNS Estimation Done for:", n_front_12, "x", n_back_12)
    print(RRMSE)

print(result_table)
np.save("data/result_121.npy", result_table)
