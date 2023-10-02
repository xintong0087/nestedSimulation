import numpy as np
import pandas as pd
import methods
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from joblib import Parallel, delayed


def knn_Euro_Call(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, k_opt=-1):

    cor_mat = np.ones((d, d)) * 0.3 + np.identity(d) * 0.7
    cov_mat = cor_mat * sigma**2

    sample_outer_train = methods.GBM_front(n_front = n_front, d = d, S_0 = S_0,
                                           drift_vec = mu, diffusion_mat = cov_mat, tau = tau)
    sample_inner = methods.GBM_back(n_front = n_front, n_back = n_back, d = d, S_tau = sample_outer_train,
                                    drift = r, diffusion = cov_mat, T = T - tau)

    payoff = np.zeros([d, n_front])
    for j in range(len(K)):
        price = np.mean(np.maximum(sample_inner - K[j], 0), axis=2) * np.exp(-r * (T - tau))
        payoff = payoff + price

    portfolio_value_0 = 0
    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_CP(S_0, T, sigma, r, K[j], 0, "C", "long")

    X_train = methods.generate_basis(sample_outer_train, option_type = "Vanilla", basis_function = "None")
    y_train = np.sum(payoff, axis=0)

    if k_opt <= 0:
        cv_kf = KFold(n_splits=3)
        k_range = np.arange(100, 301, 50)
        n_k = k_range.shape[0]
        cv_score = np.zeros(n_k)

        for k in range(n_k):

            for train_ind, val_ind in cv_kf.split(X_train, y_train):

                X = X_train[train_ind]
                X_val = X_train[val_ind]
                y = y_train[train_ind]
                y_val = y_train[val_ind]

                y_hat = KNeighborsRegressor(n_neighbors = k_range[k]).fit(X, y).predict(X_val)
                cv_score[k] = cv_score[k] + np.sum((y_hat - y_val) ** 2)

        k_opt = k_range[np.argmin(cv_score)]

        print("End of CV, optimal #neighbor =", k_opt)

    knn = KNeighborsRegressor(n_neighbors=k_opt, weights="uniform").fit(X_train, y_train)

    sample_outer_test = methods.GBM_front(n_front=n_front, d=d, S_0=S_0,
                                          drift_vec=mu, diffusion_mat=cov_mat, tau=tau)

    X_test = methods.generate_basis(sample_outer_test, option_type = "Vanilla", basis_function = "None")
    y_test = knn.predict(X_test)

    loss_knn = d * portfolio_value_0 - y_test

    return loss_knn


def cv_kNN(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T):

    cor_mat = np.ones((d, d)) * 0.3 + np.identity(d) * 0.7
    cov_mat = cor_mat * sigma ** 2

    sample_outer_train = methods.GBM_front(n_front=n_front, d=d, S_0=S_0,
                                           drift_vec=mu, diffusion_mat=cov_mat, tau=tau)
    sample_inner = methods.GBM_back(n_front=n_front, n_back=n_back, d=d, S_tau=sample_outer_train,
                                    drift=r, diffusion=cov_mat, T=T - tau)

    payoff = np.zeros([d, n_front])
    for j in range(len(K)):
        price = np.mean(np.maximum(sample_inner - K[j], 0), axis=2) * np.exp(-r * (T - tau))
        payoff = payoff + price

    portfolio_value_0 = 0
    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_CP(S_0, T, sigma, r, K[j], 0, "C", "long")

    X_train = methods.generate_basis(sample_outer_train, option_type="Vanilla", basis_function="None")
    y_train = np.sum(payoff, axis=0)

    cv_kf = KFold(n_splits=3)
    k_range = np.arange(100, 301, 50)
    n_k = k_range.shape[0]
    cv_score = np.zeros(n_k)

    for k in range(n_k):

        for train_ind, val_ind in cv_kf.split(X_train, y_train):
            X = X_train[train_ind]
            X_val = X_train[val_ind]
            y = y_train[train_ind]
            y_val = y_train[val_ind]

            y_hat = KNeighborsRegressor(n_neighbors=k_range[k]).fit(X, y).predict(X_val)
            cv_score[k] = cv_score[k] + np.sum((y_hat - y_val) ** 2)

    k_opt = k_range[np.argmin(cv_score)]

    print("End of CV, optimal #neighbor =", k_opt)

    return k_opt


def cal_RRMSE(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, alpha, L0, k_opt):

    loss = knn_Euro_Call(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, k_opt)

    indicator = np.mean((loss > L0))
    hockey = np.mean(np.maximum(loss - L0, 0))
    quadratic = np.mean((loss - L0) ** 2)

    loss.sort()
    VaR = loss[int(np.ceil((1 - alpha) * n_front))]
    CVaR = np.mean(loss[loss >= VaR])

    return indicator, hockey, quadratic, VaR, CVaR


# np.random.seed(22)
#
# d_11 = 20
# sigma_11 = 0.1
# S_0_11 = 100
# K_11 = [90, 100, 110]
# mu_11 = 0.08
# r_11 = 0.05
# tau_11 = 3/50
# T_11 = 1
# alpha_11 = 0.1
#
# result_true = np.array(pd.read_csv("data/trueValue_11.csv")).flatten()[1:]
#
# L0_11 = result_true[3]
# n_front_list = []
# for i in range(2):
#     n_front_list = n_front_list + list(np.arange(1, 10) * 10 ** (i + 3))
# n_front_list = n_front_list + [10 ** (i + 4)]
# n_front_vec = np.array(n_front_list)
# n_trials = len(n_front_list)
# n_back_vec = [1] * n_trials
#
# result_table = np.zeros([n_trials, 5])
#
# n_rep = 150
#
# for _j in range(n_trials):
#
#     n_front_11 = n_front_vec[_j]
#     n_back_11 = n_back_vec[_j]
#
#     k_cv = cv_kNN(n_front_11, n_back_11, d_11, S_0_11, K_11, mu_11, sigma_11, r_11, tau_11, T_11)
#
#     res = Parallel(n_jobs=75, verbose=10)(delayed(cal_RRMSE)(n_front_11, n_back_11, d_11, S_0_11, K_11,
#                                                          mu_11, sigma_11, r_11, tau_11, T_11,
#                                                          alpha_11, L0_11, k_cv)
#                                           for _n in range(n_rep))
#     res = np.array(res)
#
#     Bias = np.mean(res, axis=0) - result_true
#     Variance = np.var(res, axis=0)
#
#     MSE = Bias ** 2 + Variance
#     RRMSE = np.sqrt(MSE) / result_true
#
#     result_table[_j, :] = RRMSE
#     print("kNN Estimation Done for:", n_front_11, "x", n_back_11)
#     print(RRMSE)
#
# print(result_table)
# np.save("data/result_114.npy", result_table)
