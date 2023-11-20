import numpy as np
import methods
import pandas as pd
from joblib import Parallel, delayed


def ComputeTrueLoss(n_front, d, S_0, K, mu, sigma, r, tau, T, level_list=[0.8, 0.9, 0.95, 0.99, 0.996]):

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

    L0 = loss_true[int(np.ceil(0.9 * n_front))]

    indicator_true = 0.9
    hockey_true = np.mean(np.maximum(loss_true - L0, 0))
    quadratic_true = np.mean((loss_true - L0) ** 2)

    VaR = {}
    CVaR = {}
    for level in level_list:
        VaR[level] = loss_true[int(np.ceil(level * n_front)) - 1] 
        CVaR[level] = np.mean(loss_true[loss_true >= VaR[level]])

    return indicator_true, hockey_true, quadratic_true, VaR, CVaR


d = 20
sigma = 0.1
n_cmc = 10**7
S_0 = 100
K = [90, 100, 110]
mu = 0.08
r = 0.05
tau = 3/50
T = 1


indicator_true, hockey_true, quadratic_true, VaR, CVaR = ComputeTrueLoss(n_front=n_cmc,
                                                                         d=d,
                                                                         S_0=S_0,
                                                                         K=K,
                                                                         mu=mu,
                                                                         sigma=sigma,
                                                                         r=r,
                                                                         tau=tau,
                                                                         T=T,
                                                                         level_list=[0.8, 0.9, 0.95, 0.99, 0.996])

# flatten VaR and CVaR to a list append to indicator, hockey and quadratic

result_true = [indicator_true, hockey_true, quadratic_true]
label_true = ["indicator", "hockeyStick", "quadratic"]
for level in [0.8, 0.9, 0.95, 0.99, 0.996]:
    result_true.append(VaR[level])
    result_true.append(CVaR[level])
    label_true.append("VaR_" + str(level))
    label_true.append("CVaR_" + str(level))

df = pd.DataFrame(result_true,
                  index=label_true,
                  columns=["True Value"]).T

print(df)
df.to_csv("./data/trueValue_11.csv")
