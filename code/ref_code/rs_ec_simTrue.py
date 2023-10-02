import numpy as np
import methods_RS
import methods
import pandas as pd
from joblib import Parallel, delayed


def ComputeTrueLoss(n_front, d, S_0, K, mu_1, sigma_1, mu_2, sigma_2, r, tau, T, h, transition_mat, alpha=0.1):

    rho = 0.3
    cor_mat = methods.generate_cor_mat(d, rho)
    cov_mat_call_1 = cor_mat * sigma_1 ** 2
    cov_mat_call_2 = cor_mat * sigma_2 ** 2

    print("European Call: Simulating Front Paths...")
    sample_outer_call, current_regime = methods_RS.RS_front(n_front=n_front, d=d, S_0=S_0,
                                                            drift_vec_1=mu_1, diffusion_mat_1=cov_mat_call_1,
                                                            drift_vec_2=mu_2, diffusion_mat_2=cov_mat_call_2,
                                                            transition_mat=transition_mat,
                                                            tau=tau, step_size=h)
    n_step_outer = sample_outer_call.shape[2] - 1
    n_step_inner = int(T // h) - n_step_outer

    print("Calculating Value...")
    call_tau = np.zeros(n_front)
    for j in range(len(K)):
        for n in range(n_front):
            call_price = Parallel(n_jobs=d, verbose=10)(
                delayed(methods_RS.price_CP)(sample_outer_call[k, n, -1], T - tau, sigma_1, sigma_2, r, K[j],
                                             transition_mat, current_regime[n], n_step_inner, "C", "long")
                for k in range(d))
            call_tau[n] = call_tau[n] + np.sum(call_price)
    print("End of European Call.")

    portfolio_value_0 = 0

    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods_RS.price_CP(S_0, T, sigma_1, sigma_2, r, K[j],
                                                                    transition_mat, 0, n_step_outer + n_step_inner,
                                                                    "C", "long")

    loss_true = d * portfolio_value_0 - call_tau

    loss_true.sort()

    L0 = loss_true[int(np.ceil((1 - alpha) * n_front))]

    indicator_true = alpha
    hockey_true = np.mean(np.maximum(loss_true - L0, 0))
    quadratic_true = np.mean((loss_true - L0) ** 2)
    CVaR = np.mean(loss_true[loss_true > L0])

    return L0, indicator_true, hockey_true, quadratic_true, CVaR


d_21 = 20
sigma_1_21 = 0.1
sigma_2_21 = 0.2
n_cmc = 10**7
S_0_21 = 100
K_21 = [90, 100, 110]
mu_1_21 = 0.08
mu_2_21 = 0.05
r_21 = 0.05
h_21 = 1/50
tau_21 = 3/50
T_21 = 1
transition_mat_21 = np.array([[1-0.0398, 0.0398], [0.3798, 1-0.3798]])

L0_21, indicator_true_21, hockey_true_21, quadratic_true_21, CVaR_21 = ComputeTrueLoss(n_front=n_cmc,
                                                                                       d=d_21,
                                                                                       S_0=S_0_21,
                                                                                       K=K_21,
                                                                                       mu_1=mu_1_21,
                                                                                       sigma_1=sigma_1_21,
                                                                                       mu_2=mu_2_21,
                                                                                       sigma_2=sigma_2_21,
                                                                                       r=r_21,
                                                                                       tau=tau_21,
                                                                                       T=T_21,
                                                                                       h=h_21,
                                                                                       transition_mat=transition_mat_21,
                                                                                       alpha=0.1)

df = pd.DataFrame([indicator_true_21, hockey_true_21, quadratic_true_21, L0_21, CVaR_21],
                  index=["Indicator",
                         "Hockey",
                         "Quadratic",
                         "VaR",
                         "CVaR"],
                  columns=[n_cmc]).T
print(df)
df.to_csv("data/trueValue_21.csv")
