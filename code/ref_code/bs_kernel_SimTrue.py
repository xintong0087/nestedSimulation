import bs_ec_simTrue
import bs_ga_simTrue
import bs_bu_simTrue
import bs_bd_simTrue
import numpy as np

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

true_value = np.zeros([3, 4, 5])
for i in range(3):
    print(d_vec[i])
    true_value[i, 0, :] = bs_ec_simTrue.ComputeTrueLoss(n_cmc, d_vec[i], S_0, K, mu,
                                                        sigma_vec[0], r, tau, T, 0.1)
    true_value[i, 1, :] = bs_ga_simTrue.ComputeTrueLoss(n_cmc, d_vec[i], S_0, K, mu,
                                                        sigma_vec[1], r, tau, T, h, 0.1)
    true_value[i, 2, :] = bs_bu_simTrue.ComputeTrueLoss(n_cmc, d_vec[i], S_0, K, mu,
                                                        sigma_vec[2], r, tau, T, h, U, 0.1)
    true_value[i, 3, :] = bs_bd_simTrue.ComputeTrueLoss(n_cmc, d_vec[i], S_0, K, mu,
                                                        sigma_vec[3], r, tau, T, h, D, 0.1)

np.save("data/trueValue_kernel.csv", true_value)

# Dimension 0: d
# Dimension 1: option type
# Dimension 2: risk measures
