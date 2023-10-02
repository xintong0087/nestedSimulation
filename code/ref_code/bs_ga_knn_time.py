import numpy as np
import pandas as pd
import methods
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import time


def knn_Geo_Asian(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h, k_opt = -1):

    cor_mat = np.ones((d, d)) * 0.3 + np.identity(d) * 0.7
    cov_mat = cor_mat * sigma ** 2

    start_sim = time.time()
    sample_outer_train = methods.GBM_front(n_front=n_front, d=d, S_0=S_0,
                                           drift_vec=mu, diffusion_mat=cov_mat, tau=tau,
                                           step_size=h, path=True)
    n_step_outer = sample_outer_train.shape[2] - 1
    geometric_sum_outer = np.prod(sample_outer_train[:, :, 1:], axis=2)

    sample_inner = methods.GBM_back(n_front=n_front, n_back=n_back, d=d, S_tau=sample_outer_train,
                                    drift=r, diffusion=cov_mat, T=T - tau,
                                    step_size=h, path=True)
    n_step_inner = sample_inner.shape[3] - 1
    geometric_sum_inner = np.prod(sample_inner[:, :, :, 1:], axis=3)

    geometric_average = np.zeros([d, n_front, n_back])
    for i in range(n_back):
        geometric_average[:, :, i] = (geometric_sum_outer * geometric_sum_inner[:, :, i]) \
                                     ** (1 / (n_step_outer + n_step_inner))

    payoff = np.zeros([d, n_front])
    for j in range(len(K)):
        price = np.mean(np.maximum(geometric_average - K[j], 0), axis=2) * np.exp(-r * (T - tau))
        payoff = payoff + price
    time_sim = time.time() - start_sim

    start_fit = time.time()
    X_train = methods.generate_basis(sample_outer_train, option_type = "Asian", basis_function = "None")
    y_train = np.sum(payoff, axis=0)

    knn = KNeighborsRegressor(n_neighbors=k_opt, weights="uniform").fit(X_train, y_train)
    time_fit = time.time() - start_fit

    start_sim_pred = time.time()
    sample_outer_test = methods.GBM_front(n_front=n_front, d=d, S_0=S_0,
                                          drift_vec=mu, diffusion_mat=cov_mat, tau=tau,
                                          step_size=h, path=True)
    time_sim_pred = time.time() - start_sim_pred

    start_pred = time.time()
    X_test = methods.generate_basis(sample_outer_test, option_type="Asian", basis_function = "None")
    y_test = knn.predict(X_test)

    portfolio_value_0 = 0
    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_Asian_G(S=S_0, T=T, sigma=sigma, r=r, K=K[j],
                                                              continuous=False, args=n_step_outer + n_step_inner)

    loss_knn = d * portfolio_value_0 - y_test
    time_pred = time.time() - start_pred

    return loss_knn, time_sim, time_sim_pred, time_fit, time_pred


def cv_kNN(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h):

    cor_mat = np.ones((d, d)) * 0.3 + np.identity(d) * 0.7
    cov_mat = cor_mat * sigma ** 2

    sample_outer_train = methods.GBM_front(n_front=n_front, d=d, S_0=S_0,
                                           drift_vec=mu, diffusion_mat=cov_mat, tau=tau,
                                           step_size=h, path=True)
    n_step_outer = sample_outer_train.shape[2] - 1
    geometric_sum_outer = np.prod(sample_outer_train[:, :, 1:], axis=2)

    sample_inner = methods.GBM_back(n_front=n_front, n_back=n_back, d=d, S_tau=sample_outer_train,
                                    drift=r, diffusion=cov_mat, T=T - tau,
                                    step_size=h, path=True)
    n_step_inner = sample_inner.shape[3] - 1
    geometric_sum_inner = np.prod(sample_inner[:, :, :, 1:], axis=3)

    geometric_average = np.zeros([d, n_front, n_back])
    for i in range(n_back):
        geometric_average[:, :, i] = (geometric_sum_outer * geometric_sum_inner[:, :, i]) \
                                     ** (1 / (n_step_outer + n_step_inner))

    payoff = np.zeros([d, n_front])
    for j in range(len(K)):
        price = np.mean(np.maximum(geometric_average - K[j], 0), axis=2) * np.exp(-r * (T - tau))
        payoff = payoff + price

    X_train = methods.generate_basis(sample_outer_train, option_type = "Asian", basis_function = "None")
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


def cal_RRMSE(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h, alpha, L0, k_opt):

    loss, time_sim, time_sim_pred, time_fit, time_pred = knn_Geo_Asian(n_front, n_back,
                                                                       d, S_0, K, mu, sigma, r, tau, T, h, k_opt)

    indicator = np.mean((loss > L0))
    hockey = np.mean(np.maximum(loss - L0, 0))
    quadratic = np.mean((loss - L0) ** 2)

    loss.sort()
    VaR = loss[int(np.ceil((1 - alpha) * n_front))]
    CVaR = np.mean(loss[loss >= VaR])

    return time_sim, time_sim_pred, time_fit, time_pred


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

    n_front_12 = n_front_vec[_j]
    n_back_12 = n_back_vec[_j]

    start_cv = time.time()
    k_cv = cv_kNN(n_front_12, n_back_12, d_12, S_0_12, K_12, mu_12, sigma_12, r_12, tau_12, T_12, h_12)
    time_cv = time.time() - start_cv

    res = Parallel(n_jobs=-20, verbose=10)(delayed(cal_RRMSE)(n_front_12, n_back_12, d_12, S_0_12, K_12,
                                                        mu_12, sigma_12, r_12, tau_12, T_12,
                                                        h_12, alpha_12, L0_12, k_cv)
                                          for _n in range(n_rep))
    res = np.array(res)

    time_vec = np.mean(res, axis=0)

    result_table[_j, :4] = time_vec
    result_table[_j, 4] = time_cv

    print("kNN Estimation Done for:", n_front_12, "x", n_back_12)
    print(time_vec, time_cv)

print(result_table)
np.save("data/result_124_time.npy", result_table)
