import numpy as np
import methods
import pandas as pd
from joblib import Parallel, delayed


def ComputeTrueLoss(n_front, d, S_0, K, mu, sigma, r, tau, T, alpha=0.1):

    rho = 0.3
    cor_mat = methods.generate_cor_mat(d, rho)
    cov_mat_call = cor_mat * sigma ** 2

    print("European Call: Simulating Front Paths...")
    sample_outer_call = methods.GBM_front(n_front=n_front, d=d, S_0=S_0,
                                          drift_vec=mu, diffusion_mat=cov_mat_call, tau=tau)

    print("Calculating Value...")
    call_tau = np.zeros(n_front)
    for j in range(len(K)):
        call_tau_vec = Parallel(n_jobs=d, verbose=10)(
            delayed(methods.price_CP)(sample_outer_call[k, :], T - tau, sigma, r, K[j], 0, "C", "long")
            for k in range(d))
        call_tau = call_tau + np.sum(call_tau_vec, axis=0)
    print("End of European Call.")

    print(call_tau)
    portfolio_value_0 = 0

    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_CP(S_0, T, sigma, r, K[j], 0, "C", "long")

    print(d * portfolio_value_0)
    loss_true = d * portfolio_value_0 - call_tau

    loss_true.sort()

    L0 = loss_true[int(np.ceil((1 - alpha) * n_front))]

    indicator_true = alpha
    hockey_true = np.mean(np.maximum(loss_true - L0, 0))
    quadratic_true = np.mean((loss_true - L0) ** 2)
    CVaR = np.mean(loss_true[loss_true > L0])

    return L0, indicator_true, hockey_true, quadratic_true, CVaR


d_11 = 20
sigma_11 = 0.1
n_cmc = 10**7
S_0_11 = 100
K_11 = [90, 100, 110]
mu_11 = 0.08
r_11 = 0.05
tau_11 = 3/50
T_11 = 1


L0_11, indicator_true_11, hockey_true_11, quadratic_true_11, CVaR_11 = ComputeTrueLoss(n_front=n_cmc,
                                                                                       d=d_11,
                                                                                       S_0=S_0_11,
                                                                                       K=K_11,
                                                                                       mu=mu_11,
                                                                                       sigma=sigma_11,
                                                                                       r=r_11,
                                                                                       tau=tau_11,
                                                                                       T=T_11,
                                                                                       alpha=0.1)

df = pd.DataFrame([indicator_true_11, hockey_true_11, quadratic_true_11, L0_11, CVaR_11],
                  index=["Indicator",
                         "Hockey",
                         "Quadratic",
                         "VaR",
                         "CVaR"],
                  columns=[n_cmc]).T
print(df)
df.to_csv("data/trueValue_11.csv")
