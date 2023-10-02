import numpy as np
import pandas as pd
import methods
from joblib import Parallel, delayed


def SNS_Barrier_D(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h, H):

    cor_mat = np.ones((d, d)) * 0.3 + np.identity(d) * 0.7
    cov_mat = cor_mat * sigma ** 2

    sample_outer = methods.GBM_front(n_front=n_front, d=d, S_0=S_0,
                                     drift_vec=mu, diffusion_mat=cov_mat, tau=tau,
                                     step_size=h, path=True)
    n_step_outer = sample_outer.shape[2] - 1

    outer_prob_knock_out = np.random.uniform(low=0.0, high=1.0,
                                             size=[d, n_front, n_step_outer])
    sample_outer_min = np.zeros_like(outer_prob_knock_out)

    for n in range(n_step_outer):
        sample_outer_min[:, :, n] = np.exp((np.log(sample_outer[:, :, n] * sample_outer[:, :, n + 1])
                                            - np.sqrt(np.log(sample_outer[:, :, n + 1] / sample_outer[:, :, n]) ** 2
                                            - 2 * sigma ** 2 * h * np.log(outer_prob_knock_out[:, :, n]))) / 2)

    min_outer = np.min(sample_outer_min, axis=2)

    sample_inner = methods.GBM_back(n_front=n_front, n_back=n_back, d=d, S_tau=sample_outer,
                                    drift=r, diffusion=cov_mat, T=T - tau,
                                    step_size=h, path=True)
    n_step_inner = sample_inner.shape[3] - 1

    inner_prob_knock_out = np.random.uniform(low=0.0, high=1.0,
                                             size=[d, n_front, n_back, n_step_inner])
    sample_inner_min = np.zeros_like(inner_prob_knock_out)

    for i in range(n_back):
        for n in range(n_step_inner):
            sample_inner_min[:, :, i, n] = np.exp((np.log(sample_inner[:, :, i, n] * sample_inner[:, :, i, n + 1])
                                                   - np.sqrt(
                        np.log(sample_inner[:, :, i, n + 1] / sample_inner[:, :, i, n]) ** 2
                        - 2 * sigma ** 2 * h * np.log(inner_prob_knock_out[:, :, i, n]))) / 2)

    min_outer = np.repeat(min_outer[:, :, np.newaxis], n_back, axis=2)
    min_inner = np.min(sample_inner_min, axis=3)

    payoff = np.zeros([d, n_front])
    S_T = sample_inner[:, :, :, -1]

    for j in range(len(K)):
        payoff = payoff + np.mean((min_outer > H) * (min_inner > H) * np.maximum(S_T - K[j], 0),
                                  axis=2) * np.exp(-r * (T - tau))

    portfolio_value_0 = 0
    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_barrier([S_0], K[j], T, sigma, r, H, 0, 0,
                                                                      option_type=np.array([["O", "D", "C"]]))

    loss_SNS = d * portfolio_value_0 - np.sum(payoff, axis=0)

    return loss_SNS


def cal_RRMSE(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h, D, alpha, L0):

    loss = SNS_Barrier_D(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h, D)

    indicator = np.mean((loss > L0))
    hockey = np.mean(np.maximum(loss - L0, 0))
    quadratic = np.mean((loss - L0) ** 2)

    loss.sort()
    VaR = loss[int(np.ceil((1 - alpha) * n_front))]
    CVaR = np.mean(loss[loss >= VaR])

    return indicator, hockey, quadratic, VaR, CVaR


np.random.seed(22)

d_14 = 20
sigma_14 = 0.2
S_0_14 = 100
K_14 = [90, 100, 110]
mu_14 = 0.08
r_14 = 0.05
T_14 = 1
h_14 = 1/50
tau_14 = h_14 * 3
alpha_14 = 0.1
D_14 = 90

result_true = np.array(pd.read_csv("data/trueValue_14.csv")).flatten()[1:]

L0_14 = result_true[3]

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

    n_front_14 = n_front_vec[_j]
    n_back_14 = n_back_vec[_j]

    res = Parallel(n_jobs=-1, verbose=10)(delayed(cal_RRMSE)(n_front_14, n_back_14, d_14, S_0_14, K_14,
                                                        mu_14, sigma_14, r_14, tau_14, T_14,
                                                        h_14, D_14, alpha_14, L0_14)
                                          for _n in range(n_rep))
    res = np.array(res)

    Bias = np.mean(res, axis=0) - result_true
    Variance = np.var(res, axis=0)

    MSE = Bias ** 2 + Variance
    RRMSE = np.sqrt(MSE) / result_true

    result_table[_j, :] = RRMSE
    print("SNS Estimation Done for:", n_front_14, "x", n_back_14)
    print(RRMSE)

print(result_table)
np.save("data/result_141.npy", result_table)
