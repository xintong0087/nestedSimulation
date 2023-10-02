import numpy as np
import methods
import pandas as pd
from joblib import Parallel, delayed


def ComputeTrueLoss(n_front, d, S_0, K, mu, sigma, r, tau, T, h_asian, alpha=0.1):

    rho = 0.3
    cor_mat = methods.generate_cor_mat(d, rho)
    cov_mat_asian = cor_mat * sigma ** 2

    print("Geometric Asian: Simulating Front Paths...")
    sample_outer_asian = methods.GBM_front(n_front=n_front, d=d, S_0=S_0,
                                           drift_vec=mu, diffusion_mat=cov_mat_asian, tau=tau,
                                           step_size=h_asian, path=True)
    n_step_outer = sample_outer_asian.shape[2] - 1
    n_step_inner = int(T // h_asian) - n_step_outer
    S_tau = sample_outer_asian[:, :, 1:]

    print("Calculating Value...")
    asian_tau = np.zeros(n_front)
    for j in range(len(K)):
        asian_tau_vec = Parallel(n_jobs=d, verbose=10)(
            delayed(methods.price_Asian_G_tau)(S_tau[k, :, :], T - tau, sigma, r, K[j],
                                               False, args=(n_step_outer, n_step_inner))
            for k in range(d))
        asian_tau = asian_tau + np.sum(asian_tau_vec, axis=0)

    print("End of Geometric Asian.")

    portfolio_value_0 = 0

    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_Asian_G(S=S_0, T=T, sigma=sigma, r=r, K=K[j],
                                                                      continuous=False,
                                                                      args=n_step_outer + n_step_inner)

    loss_true = d * portfolio_value_0 - asian_tau

    loss_true.sort()

    L0 = loss_true[int(np.ceil((1 - alpha) * n_front))]

    indicator_true = alpha
    hockey_true = np.mean(np.maximum(loss_true - L0, 0))
    quadratic_true = np.mean((loss_true - L0) ** 2)
    CVaR = np.mean(loss_true[loss_true > L0])

    return L0, indicator_true, hockey_true, quadratic_true, CVaR


d_12 = 20
sigma_12 = 0.3
n_cmc = 10**7
S_0_12 = 100
K_12 = [90, 100, 110]
mu_12 = 0.08
r_12 = 0.05
T_12 = 1
h_12 = T_12 / 50
tau_12 = 3 * h_12

L0_12, indicator_true_12, hockey_true_12, quadratic_true_12, CVaR_12 = ComputeTrueLoss(n_front=n_cmc,
                                                                                       d=d_12,
                                                                                       S_0=S_0_12,
                                                                                       K=K_12,
                                                                                       mu=mu_12,
                                                                                       sigma=sigma_12,
                                                                                       r=r_12,
                                                                                       tau=tau_12,
                                                                                       T=T_12,
                                                                                       h_asian=h_12,
                                                                                       alpha=0.1)

df = pd.DataFrame([indicator_true_12, hockey_true_12, quadratic_true_12, L0_12, CVaR_12],
                  index=["Indicator",
                         "Hockey",
                         "Quadratic",
                         "VaR",
                         "CVaR"],
                  columns=[n_cmc]).T
print(df)
df.to_csv("data/trueValue_12.csv")
