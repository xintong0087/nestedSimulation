import numpy as np
import methods
import pandas as pd
from joblib import Parallel, delayed


def ComputeTrueLoss(n_front, d, S_0, K, mu, sigma, r, tau, T, h_barrier, H, alpha=0.1):

    rho = 0.3
    cor_mat = methods.generate_cor_mat(d, rho)
    cov_mat_barrier = cor_mat * sigma ** 2

    print("Barrier Options: Simulating Front Paths...")
    sample_outer_barrier = methods.GBM_front(n_front=n_front, d=d, S_0=S_0,
                                             drift_vec=mu, diffusion_mat=cov_mat_barrier, tau=tau,
                                             step_size=h_barrier, path=True)
    n_step_outer = sample_outer_barrier.shape[2] - 1

    print("Simulating Path Maximums and Minimums...")
    outer_prob_knock_out = np.random.uniform(low=0.0, high=1.0,
                                             size=[d, n_front, n_step_outer])
    sample_outer_max = np.zeros_like(outer_prob_knock_out)

    for n in range(n_step_outer):
        sample_outer_max[:, :, n] = np.exp((np.log(sample_outer_barrier[:, :, n] * sample_outer_barrier[:, :, n + 1])
                                            + np.sqrt(
                    np.log(sample_outer_barrier[:, :, n + 1] / sample_outer_barrier[:, :, n]) ** 2
                    - 2 * sigma ** 2 * h_barrier * np.log(outer_prob_knock_out[:, :, n]))) / 2)

    max_outer = np.max(sample_outer_max, axis=2)

    print("Calculating Value...")
    barrier_tau = np.zeros(n_front)
    for j in range(len(K)):
        barrier_tau_vec = Parallel(n_jobs=d, verbose=10)(
            delayed(methods.price_barrier_up)(n_front, max_outer[k, :],
                                              sample_outer_barrier[k, :, -1], K[j],
                                              T - tau, sigma, r, H)
            for k in range(d))

        barrier_tau = barrier_tau + np.sum(barrier_tau_vec, axis=0)

    print("End of Barrier.")
    portfolio_value_0 = 0

    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_barrier([S_0], K[j], T, sigma, r, H, 0, 0,
                                                                      option_type=np.full([1, 3], ["O", "U", "C"]))

    loss_true = d * portfolio_value_0 - barrier_tau

    loss_true.sort()

    L0 = loss_true[int(np.ceil((1 - alpha) * n_front))]

    indicator_true = alpha
    hockey_true = np.mean(np.maximum(loss_true - L0, 0))
    quadratic_true = np.mean((loss_true - L0) ** 2)
    CVaR = np.mean(loss_true[loss_true > L0])

    return L0, indicator_true, hockey_true, quadratic_true, CVaR


d_13 = 20
sigma_13 = 0.2
n_cmc = 10**7
S_0_13 = 100
K_13 = [90, 100, 110]
mu_13 = 0.08
r_13 = 0.05
T_13 = 1
h_13 = T_13 / 50
tau_13 = 3 * h_13
U = 120

L0_13, indicator_true_13, hockey_true_13, quadratic_true_13, CVaR_13 = ComputeTrueLoss(n_front=n_cmc,
                                                                                       d=d_13,
                                                                                       S_0=S_0_13,
                                                                                       K=K_13,
                                                                                       mu=mu_13,
                                                                                       sigma=sigma_13,
                                                                                       r=r_13,
                                                                                       tau=tau_13,
                                                                                       T=T_13,
                                                                                       h_barrier=h_13,
                                                                                       H = U,
                                                                                       alpha=0.1)

df = pd.DataFrame([indicator_true_13, hockey_true_13, quadratic_true_13, L0_13, CVaR_13],
                  index=["Indicator",
                         "Hockey",
                         "Quadratic",
                         "VaR",
                         "CVaR"],
                  columns=[n_cmc]).T
print(df)
df.to_csv("data/trueValue_13.csv")
