import numpy as np
import pandas as pd
import methods
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.kernels import Matern
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from skopt import BayesSearchCV
from skopt.space import Real


def krr_Barrier_U(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h, H, alpha_opt, l_opt, nu_opt):

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
        sample_outer_max[:, :, n] = np.exp((np.log(sample_outer[:, :, n] * sample_outer[:, :, n+1])
                  + np.sqrt(np.log(sample_outer[:, :, n+1] / sample_outer[:, :, n])**2
                  - 2 * sigma**2 * h * np.log(outer_prob_knock_out[:, :, n]))) / 2)

    max_outer = np.max(sample_outer_max, axis = 2)

    sample_inner = methods.GBM_back(n_front=n_front, n_back=n_back, d=d, S_tau=sample_outer,
                                    drift=r, diffusion=cov_mat, T=T - tau,
                                    step_size=h, path=True)
    n_step_inner = sample_inner.shape[3] - 1

    inner_prob_knock_out = np.random.uniform(low=0.0, high=1.0,
                                             size=[d, n_front, n_back, n_step_inner])
    sample_inner_max = np.zeros_like(inner_prob_knock_out)

    for i in range(n_back):
        for n in range(n_step_inner):
            sample_inner_max[:, :, i, n] = np.exp((np.log(sample_inner[:, :, i, n] * sample_inner[:, :, i, n+1])
                  + np.sqrt(np.log(sample_inner[:, :, i, n+1] / sample_inner[:, :, i, n])**2
                  - 2 * sigma**2 * h * np.log(inner_prob_knock_out[:, :, i, n]))) / 2)

    max_outer = np.repeat(max_outer[:, :, np.newaxis], n_back, axis = 2)
    max_inner = np.max(sample_inner_max, axis = 3)

    payoff = np.zeros([d, n_front])
    S_T = sample_inner[:, :, :, -1]

    for j in range(len(K)):
        payoff = payoff + np.mean((max_outer < H) * (max_inner < H)
                                  * np.maximum(S_T - K[j], 0), axis = 2) * np.exp(-r * (T - tau))

    X_train = methods.generate_basis(sample_outer[:, :, -1], option_type="Vanilla", basis_function="None")
    y_train = np.sum(payoff, axis=0)

    kernel = 1.0 * Matern(length_scale=l_opt, nu=nu_opt)
    krr = KernelRidge(alpha=alpha_opt, kernel=kernel).fit(X_train, y_train)

    sample_outer_test = methods.GBM_front(n_front=n_front, d=d, S_0=S_0,
                                          drift_vec=mu, diffusion_mat=cov_mat, tau=tau,
                                          step_size=h, path=True)

    X_test = methods.generate_basis(sample_outer_test[:, :, -1], option_type="Vanilla", basis_function="None")

    y_test = krr.predict(X_test)

    portfolio_value_0 = 0
    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_barrier([S_0], K[j], T, sigma, r, H, 0, 0,
                                                                      option_type=np.array([["O", "U", "C"]]))

    loss_KRR = d * portfolio_value_0 - y_test

    return loss_KRR


def cv_krr(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h, H):
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
                                            - 2 * sigma ** 2 * h * np.log(outer_prob_knock_out[:, :, n]))) / 2)

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

    X_train = methods.generate_basis(sample_outer[:, :, -1], option_type="Vanilla", basis_function="None")
    y_train = np.sum(payoff, axis=0)

    param_distributions = {
        "alpha": Real(1e-5, 1e-1, "log-uniform"),
        "kernel__length_scale": Real(1e-3, 1e3, "log-uniform"),
        "kernel__nu": Real(5e-1, 5e0, "log-uniform"),
    }

    cv_kf = KFold(n_splits=5)

    bayesian_search = BayesSearchCV(estimator=KernelRidge(kernel=Matern()),
                                    search_spaces=param_distributions, n_jobs=20, cv=cv_kf)

    bayesian_search.fit(X_train, y_train)

    alpha = bayesian_search.best_params_["alpha"]
    l = bayesian_search.best_params_["kernel__length_scale"]
    nu = bayesian_search.best_params_["kernel__nu"]

    print("End of CV, optimal hyperparameter =", alpha, l, nu)

    return alpha, l, nu


def cal_RRMSE(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h, U, alpha, L0, alpha_opt, l_opt, nu_opt):

    loss = krr_Barrier_U(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h, U, alpha_opt, l_opt, nu_opt)

    indicator = np.mean((loss > L0))
    hockey = np.mean(np.maximum(loss - L0, 0))
    quadratic = np.mean((loss - L0) ** 2)

    loss.sort()
    VaR = loss[int(np.ceil((1 - alpha) * n_front))]
    CVaR = np.mean(loss[loss >= VaR])

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

# n_front_list = []
# for _i in range(2):
#     n_front_list = n_front_list + list(np.arange(1, 10) * 10 ** (_i + 2))
# n_front_list = n_front_list + [10 ** (_i + 3)]
# n_front_vec = np.array(n_front_list)
n_front_list = [10000]
n_front_vec = np.array(n_front_list)
n_trials = len(n_front_list)
n_back_vec = [10] * n_trials

result_table = np.zeros([n_trials, 5])

n_rep = 150

for _j in range(n_trials):

    n_front_13 = n_front_vec[_j]
    n_back_13 = n_back_vec[_j]

    # alpha_opt_13, l_opt_13, nu_opt_13 = cv_krr(n_front_13, n_back_13,
    #                                            d_13, S_0_13, K_13, mu_13, sigma_13, r_13, tau_13, T_13, h_13, U_13)

    alpha_opt_13, l_opt_13, nu_opt_13 = (0.004258387814150004, 990.6825876120273, 1.6051494706067806)

    res = Parallel(n_jobs=2, verbose=10)(delayed(cal_RRMSE)(n_front_13, n_back_13, d_13, S_0_13, K_13,
                                                         mu_13, sigma_13, r_13, tau_13, T_13,
                                                         h_13, U_13, alpha_13, L0_13, alpha_opt_13, l_opt_13, nu_opt_13)
                                          for _n in range(n_rep))
    res = np.array(res)

    Bias = np.mean(res, axis=0) - result_true
    Variance = np.var(res, axis=0)

    MSE = Bias ** 2 + Variance
    RRMSE = np.sqrt(MSE) / result_true

    result_table[_j, :] = RRMSE
    print("kNN Estimation Done for:", n_front_13, "x", n_back_13)
    print(RRMSE)

print(result_table)
# np.save("data/result_136.npy", result_table)
