import numpy as np
import pandas as pd
import methods
import methods_RS
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed


def SNS_Euro_Call(n_front, n_back, d, S_0, K, mu_1, sigma_1, mu_2, sigma_2, r, tau, T, h, transition_mat):

    rho = 0.3
    cor_mat = methods.generate_cor_mat(d, rho)
    cov_mat_call_1 = cor_mat * sigma_1 ** 2
    cov_mat_call_2 = cor_mat * sigma_2 ** 2

    sample_outer, current_regime = methods_RS.RS_front(n_front=n_front, d=d, S_0=S_0,
                                                       drift_vec_1=mu_1, diffusion_mat_1=cov_mat_call_1,
                                                       drift_vec_2=mu_2, diffusion_mat_2=cov_mat_call_2,
                                                       transition_mat=transition_mat,
                                                       tau=tau, step_size=h)
    sample_inner = methods_RS.RS_back(n_front=n_front, n_back=n_back, d=d, S_tau=sample_outer,
                                      drift_1=r, diffusion_1=cov_mat_call_1,
                                      drift_2=r, diffusion_2=cov_mat_call_2,
                                      transition_mat=transition_mat, current_regime=current_regime,
                                      T=T-tau, step_size=h)

    n_step_outer = sample_outer.shape[2] - 1
    n_step_inner = sample_inner.shape[3] - 1
    n_steps = n_step_outer + n_step_inner

    payoff = np.zeros([d, n_front])
    for j in range(len(K)):
        price = np.mean(np.maximum(sample_inner[:, :, :, -1] - K[j], 0), axis=2) * np.exp(-r * (T - tau))
        payoff = payoff + price

    portfolio_value_0 = 0
    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods_RS.price_CP(S_0, T, sigma_1, sigma_2, r, K[j],
                                                                    transition_mat, 0, n_steps, "C", "long")

    loss_SNS = d * portfolio_value_0 - np.sum(payoff, axis=0)

    return loss_SNS


def SNS_Euro_Call_BootStrap(Gamma, n_front_vec, n_back_vec, d, S_0, K, mu_1, sigma_1, mu_2, sigma_2,
                            transition_mat, r, tau, T, h, L0, I=500, alpha=0.1):

    rho = 0.3
    cor_mat = methods.generate_cor_mat(d, rho)
    cov_mat_call_1 = cor_mat * sigma_1 ** 2
    cov_mat_call_2 = cor_mat * sigma_2 ** 2

    n_steps = int(T // h)
    portfolio_value_0 = 0
    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods_RS.price_CP(S_0, T, sigma_1, sigma_2, r, K[j],
                                                                    transition_mat, 0, n_steps, "C", "long")

    n_front_0 = n_front_vec[-1]
    n_back_0 = n_back_vec[-1]

    sample_outer_0, current_regime = methods_RS.RS_front(n_front=n_front_0, d=d, S_0=S_0,
                                                         drift_vec_1=mu_1, diffusion_mat_1=cov_mat_call_1,
                                                         drift_vec_2=mu_2, diffusion_mat_2=cov_mat_call_2,
                                                         transition_mat=transition_mat,
                                                         tau=tau, step_size=h)

    sample_inner_0 = methods_RS.RS_back(n_front=n_front_0, n_back=n_back_0, d=d, S_tau=sample_outer_0,
                                       drift_1=r, diffusion_1=cov_mat_call_1,
                                       drift_2=r, diffusion_2=cov_mat_call_2,
                                       transition_mat=transition_mat, current_regime=current_regime,
                                       T=T-tau, step_size=h)

    outer_shape = n_front_vec.shape[0]
    alpha_mat = np.zeros([outer_shape, 5])

    counter = 0
    for n_back in n_back_vec:

        result = np.zeros([I, 5])

        for i in range(I):

            index_outer = np.random.choice(n_front_0, size=n_front_0, replace=True)
            index_inner = np.random.choice(n_back_0, size=n_back, replace=True)
            sample_outer_bs = sample_inner_0[:, index_outer, :, -1]
            sample_inner_bs = sample_outer_bs[:, :, index_inner]

            payoff = np.zeros([d, n_front_0])
            for j in range(len(K)):
                price = np.mean(np.maximum(sample_inner_bs - K[j], 0), axis=2) * np.exp(-r * (T - tau))
                payoff = payoff + price

            loss_bs = d * portfolio_value_0 - np.sum(payoff, axis=0)

            result[i, 0] = np.nanmean((loss_bs > L0))
            result[i, 1] = np.nanmean(np.maximum(loss_bs - L0, 0))
            result[i, 2] = np.nanmean((loss_bs - L0) ** 2)

            loss_bs.sort()
            result[i, 3] = loss_bs[int(np.ceil((1 - alpha) * n_front_0))]
            result[i, 4] = np.nanmean(loss_bs[loss_bs >= result[i, 3]])

        alpha_mat[counter, :] = np.mean(result, axis=0)
        counter = counter + 1

    inner_shape = n_back_vec.shape[0]
    s_mat = np.zeros([inner_shape, 5])

    counter = 0
    for n_front in n_front_vec:

        result = np.zeros([I, 5])

        for i in range(I):

            sample_inner_bs = sample_inner_0[:, np.random.choice(n_front_0, size=n_front, replace=True), :, -1]

            payoff = np.zeros([d, n_front])
            for j in range(len(K)):
                price = np.mean(np.maximum(sample_inner_bs - K[j], 0), axis=2) * np.exp(-r * (T - tau))
                payoff = payoff + price

            loss_bs = d * portfolio_value_0 - np.sum(payoff, axis=0)

            result[i, 0] = np.nanmean((loss_bs > L0))
            result[i, 1] = np.nanmean(np.maximum(loss_bs - L0, 0))
            result[i, 2] = np.nanmean((loss_bs - L0) ** 2)

            loss_bs.sort()
            result[i, 3] = loss_bs[int(np.ceil((1 - alpha) * n_front))]
            result[i, 4] = np.nanmean(loss_bs[loss_bs >= result[i, 3]])

        s_mat[counter, :] = np.var(result, axis=0)
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


def cal_RRMSE(Gamma, n_front, n_back, d, S_0, K, mu_1, sigma_1, mu_2, sigma_2, transition_mat,
              r, tau, T, h, alpha, L0):

    n_front_opt, n_back_opt = SNS_Euro_Call_BootStrap(Gamma, n_front, n_back,
                                                      d, S_0, K, mu_1, sigma_1, mu_2, sigma_2,
                                                      transition_mat, r, tau, T, h, L0, 500, alpha)

    loss = SNS_Euro_Call(n_front_opt[0], n_back_opt[0], d, S_0, K, mu_1, sigma_1, mu_2, sigma_2, r, tau, T, h,
                         transition_mat)
    indicator = np.mean((loss > L0))

    loss = SNS_Euro_Call(n_front_opt[1], n_back_opt[1], d, S_0, K, mu_1, sigma_1, mu_2, sigma_2, r, tau, T, h,
                         transition_mat)
    hockey = np.mean(np.maximum(loss - L0, 0))

    loss = SNS_Euro_Call(n_front_opt[2], n_back_opt[2], d, S_0, K, mu_1, sigma_1, mu_2, sigma_2, r, tau, T, h,
                         transition_mat)
    quadratic = np.mean((loss - L0) ** 2)

    loss = SNS_Euro_Call(n_front_opt[3], n_back_opt[3], d, S_0, K, mu_1, sigma_1, mu_2, sigma_2, r, tau, T, h,
                         transition_mat)
    loss.sort()
    VaR = loss[int(np.ceil((1 - alpha) * n_front_opt[3]))]

    loss = SNS_Euro_Call(n_front_opt[4], n_back_opt[4], d, S_0, K, mu_1, sigma_1, mu_2, sigma_2, r, tau, T, h,
                         transition_mat)
    loss.sort()
    Q = loss[int(np.ceil((1 - alpha) * n_front_opt[4]))]
    CVaR = np.mean(loss[loss >= Q])

    return indicator, hockey, quadratic, VaR, CVaR


np.random.seed(22)

d_21 = 20
sigma_1_21 = 0.1
sigma_2_21 = 0.2
S_0_21 = 100
K_21 = [90, 100, 110]
mu_1_21 = 0.08
mu_2_21 = 0.05
r_21 = 0.05
h_21 = 1/50
tau_21 = 3/50
T_21 = 1
alpha_21 = 0.1
transition_mat_21 = np.array([[1-0.0398, 0.0398], [0.3798, 1-0.3798]])

result_true = np.array(pd.read_csv("data/trueValue_21.csv")).flatten()[1:]

L0_21 = result_true[3]

Gamma_list = []
for _i in range(2):
    Gamma_list = Gamma_list + list(np.arange(1, 10) * 10 ** (_i + 3))
Gamma_list = Gamma_list + [10 ** (_i + 4)]
_Gamma = np.array(Gamma_list)
n_trials = len(_Gamma)

n_front_21 = [np.arange(50, 101, 5)] * n_trials
n_back_21 = [np.arange(50, 101, 5)] * n_trials
result_table = np.zeros([n_trials, 5])

n_rep = 150

for _j in range(n_trials):

    Gamma_21 = _Gamma[_j]

    res = Parallel(n_jobs=20, verbose=10)(delayed(cal_RRMSE)(Gamma_21, n_front_21[_j], n_back_21[_j],
                                                             d_21, S_0_21, K_21,
                                                             mu_1_21, sigma_1_21, mu_2_21, sigma_2_21,
                                                             transition_mat_21, r_21, tau_21, T_21, h_21,
                                                             alpha_21, L0_21)
                                          for _n in range(n_rep))

    Bias = np.nanmean(res, axis=0) - result_true
    Variance = np.nanvar(res, axis=0)

    MSE = Bias ** 2 + Variance
    RRMSE = np.sqrt(MSE) / result_true

    result_table[_j, :] = RRMSE
    print("SNS Estimation Done for:", Gamma_21)
    print(RRMSE)

print(result_table)
np.save("data/result_212.npy", result_table)
