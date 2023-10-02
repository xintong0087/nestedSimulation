import numpy as np
import bs_ec_knn
import bs_ga_knn
import bs_bu_knn
import bs_bd_knn
from joblib import Parallel, delayed

np.random.seed(22)

true_value = np.load("data/trueValue_kernel.npy")
true_value = np.insert(true_value[:, :, 1:], 3, true_value[:, :, 0], axis = 2)
n_cores = 30

d_vec = [1, 5, 10]
sigma_vec = [0.1, 0.3, 0.2, 0.2]
n_cmc = 10**7
S_0 = 100
K = [90, 100, 110]
mu = 0.08
r = 0.05
tau = 3/50
T = 1
U = 120
D = 90
h = 1/50
alpha = 0.1

n_front_list = []
for i in range(2):
    n_front_list = n_front_list + list(np.arange(1, 10) * 10 ** (i + 3))
n_front_list = n_front_list + [10 ** (i + 4)]
n_front_vec = np.array(n_front_list)
n_trials = len(n_front_list)
n_back_vec = [1] * n_trials

MSE_table = np.zeros([3, 4, 5, n_trials])
n_rep = 150

for i in range(3):

    print("Starting Estimation for:", d_vec[i])
    d = d_vec[i]

    for j in range(n_trials):

        n_front = n_front_vec[j]
        n_back = n_back_vec[j]

        # European Call
        print("Starting Estimation for European Call...")
        L0 = true_value[i, 0, 3]
        k_cv = bs_ec_knn.cv_kNN(n_front, n_back, d, S_0, K, mu, sigma_vec[0], r, tau, T)
        res = Parallel(n_jobs=n_cores, verbose=10)(delayed(bs_ec_knn.cal_RRMSE)(n_front, n_back, d, S_0, K,
                                                                           mu, sigma_vec[0], r, tau, T,
                                                                           alpha, L0, k_cv)
                                              for n in range(n_rep))

        result_true = true_value[i, 0, :]
        Bias = np.mean(res, axis=0) - result_true
        Variance = np.var(res, axis=0)
        MSE = Bias ** 2 + Variance
        RRMSE = np.sqrt(MSE) / result_true

        MSE_table[i, 0, :, j] = RRMSE

        # Geometric Asian
        print("Starting Estimation for Geometric Asian...")
        L0 = true_value[i, 1, 3]
        k_cv = bs_ga_knn.cv_kNN(n_front, n_back, d, S_0, K, mu, sigma_vec[1], r, tau, T, h)
        res = Parallel(n_jobs=n_cores, verbose=10)(delayed(bs_ga_knn.cal_RRMSE)(n_front, n_back, d, S_0, K,
                                                                           mu, sigma_vec[1], r, tau, T, h,
                                                                           alpha, L0, k_cv)
                                              for n in range(n_rep))

        result_true = true_value[i, 1, :]
        Bias = np.mean(res, axis=0) - result_true
        Variance = np.var(res, axis=0)
        MSE = Bias ** 2 + Variance
        RRMSE = np.sqrt(MSE) / result_true

        MSE_table[i, 1, :, j] = RRMSE

        # Up and Out Barrier
        print("Starting Estimation for Up and Out Barrier...")
        L0 = true_value[i, 2, 3]
        k_cv = bs_bu_knn.cv_kNN(n_front, n_back, d, S_0, K, mu, sigma_vec[2], r, tau, T, U, h)
        res = Parallel(n_jobs=n_cores, verbose=10)(delayed(bs_bu_knn.cal_RRMSE)(n_front, n_back, d, S_0, K,
                                                                           mu, sigma_vec[2], r, tau, T, h, U,
                                                                           alpha, L0, k_cv)
                                              for n in range(n_rep))

        result_true = true_value[i, 2, :]
        Bias = np.mean(res, axis=0) - result_true
        Variance = np.var(res, axis=0)
        MSE = Bias ** 2 + Variance
        RRMSE = np.sqrt(MSE) / result_true

        MSE_table[i, 2, :, j] = RRMSE

        # Down and Out Barrier
        print("Starting Estimation for Down and Out Barrier...")
        L0 = true_value[i, 3, 3]
        k_cv = bs_bd_knn.cv_kNN(n_front, n_back, d, S_0, K, mu, sigma_vec[3], r, tau, T, D, h)
        res = Parallel(n_jobs=n_cores, verbose=10)(delayed(bs_bd_knn.cal_RRMSE)(n_front, n_back, d, S_0, K,
                                                                           mu, sigma_vec[3], r, tau, T, h, D,
                                                                           alpha, L0, k_cv)
                                              for n in range(n_rep))

        result_true = true_value[i, 3, :]
        Bias = np.mean(res, axis=0) - result_true
        Variance = np.var(res, axis=0)
        MSE = Bias ** 2 + Variance
        RRMSE = np.sqrt(MSE) / result_true

        MSE_table[i, 3, :, j] = RRMSE

        print("Estimation done for:", n_front, "x", n_back)
        print(MSE_table[i, :, :, j])

print(MSE_table)
np.save("data/result_kernel.npy", MSE_table)
