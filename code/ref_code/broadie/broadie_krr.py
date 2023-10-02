import numpy as np
import pandas as pd
import method_Broadie
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import KFold
import time
import os
from joblib import Parallel, delayed
from skopt import BayesSearchCV
from skopt.space import Real

def broadie_krr(M, N, S_0, K, H, mu, sigma, r, tau, T, seed_train, seed_test, l_opt, nu_opt, alpha_opt):

    crude = True
    loss_flag = True

    X_train, y_train = method_Broadie.simPredictors(M, N, S_0, K, H, mu, sigma, r, tau, T, seed_train, crude, loss_flag)

    X_train = X_train.reshape(-1, 1)

    kernel = 1.0 * Matern(length_scale=l_opt, nu=nu_opt)
    krr = KernelRidge(alpha=alpha_opt, kernel=kernel).fit(X_train, y_train)

    X_test, y_test = method_Broadie.simPredictors(M, N, S_0, K, H, mu, sigma, r, tau, T, seed_test, crude)

    X_test = X_test.reshape(-1, 1)

    loss = krr.predict(X_test)

    return loss


def cv_krr(M, N, S_0, K, H, mu, sigma, r, tau, T, seed):

    crude = True
    loss_flag = True

    X, y = method_Broadie.simPredictors(M, N, S_0, K, H, mu, sigma, r, tau, T, seed, crude, loss_flag)
    X = X.reshape(-1, 1)

    param_distributions = {
        "alpha": Real(1e-5, 1e-1, "log-uniform"),
        "kernel__length_scale": Real(1e-3, 1e3, "log-uniform"),
        "kernel__nu": Real(5e-1, 5e0, "log-uniform"),
    }

    cv_kf = KFold(n_splits=5)

    bayesian_search = BayesSearchCV(estimator=KernelRidge(kernel=Matern()),
                                    search_spaces=param_distributions, n_jobs=n_cores, cv=cv_kf)

    bayesian_search.fit(X, y)

    alpha = bayesian_search.best_params_["alpha"]
    l = bayesian_search.best_params_["kernel__length_scale"]
    nu = bayesian_search.best_params_["kernel__nu"]

    print("End of CV, optimal hyperparameter =", alpha, l, nu)

    return l, nu, alpha


def cal_RRMSE(M, N, S_0, K, H, mu, sigma, r, tau, T, alpha, L0, seed, l_opt, nu_opt, alpha_opt):

    rng = np.random.default_rng(seed)

    loss = broadie_krr(M, N, S_0, K, H, mu, sigma, r, tau, T, rng.integers(0, 2147483647), rng.integers(0, 2147483647), l_opt, nu_opt, alpha_opt)
    indicator = np.mean((loss > L0))
    hockey = np.mean(np.maximum(loss - L0, 0))
    quadratic = np.mean((loss - L0) ** 2)

    loss.sort()
    VaR = loss[int(np.ceil((1 - alpha) * M))]
    CVaR = np.mean(loss[loss >= VaR])

    return indicator, hockey, quadratic, VaR, CVaR


local = False

save_path = "./results/"

if not os.path.exists(save_path):
    os.makedirs(save_path)

if local:
    n_cores = 4
else:
    n_cores = 60


K = [101, 110, 114.5]
H = [91, 100, 104.5]

S_0 = 100
mu = 0.08
sigma = 0.2
r = 0.03
T = 1/12
tau = 1/52

alpha = 0.05

result_true = np.array(pd.read_csv("./trueValue_Broadie.csv")).flatten()[1:]

L0 = result_true[3]

M_list = []
for i in range(2):
    M_list = M_list + list(np.arange(1, 10) * 10 ** (i + 2))
M_list = M_list + [10 ** (i + 3)]
M_vec = np.array(M_list)
n_trials = len(M_list)
N_vec = [10] * n_trials

result_table = np.zeros([n_trials, 5])

start = time.time()

n_rep = 150

seed = np.random.randint(low=0, high=2147483647, size=n_rep)
seed_cv = np.random.randint(low=0, high=2147483647)

for j in range(n_trials):

    M = M_vec[j]
    N = N_vec[j]

    l_opt, nu_opt, alpha_opt = cv_krr(M, N, S_0, K, H, mu, sigma, r, tau, T, seed_cv)

    res = Parallel(n_jobs=n_cores, verbose=10)(delayed(cal_RRMSE)(M, N,
                                                                  S_0, K, H, mu, sigma, r, tau, T, alpha, L0, seed[n],
                                                                  l_opt, nu_opt, alpha_opt)
                                               for n in range(n_rep))

    Bias = np.nanmean(res, axis=0) - result_true
    Variance = np.nanvar(res, axis=0)

    MSE = Bias ** 2 + Variance
    RRMSE = np.sqrt(MSE) / result_true

    result_table[j, :] = RRMSE
    print("KRR Estimation Done for:", M, "x", N)
    print(RRMSE)

print(result_table)
np.save(save_path + "result_156.npy", result_table)
