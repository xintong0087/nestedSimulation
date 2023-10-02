import numpy as np
from scipy.stats import multivariate_normal as multinorm
from scipy.stats import norm


def compute_abcd(upper, lower, r, q, t, sigma):
    res = ((np.log(upper / lower) + (r - q) * t) / (sigma * np.sqrt(t)) - (sigma * np.sqrt(t) / 2))

    return res, res + sigma * np.sqrt(t)


def barrier(S, K, B, r, q, sigma, t, T):
    # at, apt = compute_abcd(S, K, r, q, t, sigma)
    aT, apT = compute_abcd(S, K, r, q, T, sigma)
    bt, bpt = compute_abcd(S, B, r, q, t, sigma)
    bT, bpT = compute_abcd(S, B, r, q, T, sigma)
    ct, cpt = compute_abcd(B, S, r, q, t, sigma)
    cT, cpT = compute_abcd(B, S, r, q, T, sigma)
    # dt, dpt = compute_abcd(B ** 2, S * K, r, q, t, sigma)
    dT, dpT = compute_abcd(B ** 2, S * K, r, q, T, sigma)

    rho = np.sqrt(t / T)
    rho_mat = [[1, rho], [rho, 1]]
    nrho_mat = [[1, -rho], [-rho, 1]]
    k = 2 * (r - q) / (sigma ** 2)

    P = (K * np.exp(-r * T) * (multinorm.cdf([bt, -aT], mean=[0, 0], cov=nrho_mat)
                               - multinorm.cdf([bt, -bT], mean=[0, 0], cov=nrho_mat))
         - S * np.exp(-q * T) * (multinorm.cdf([bpt, -apT], mean=[0, 0], cov=nrho_mat)
                                 - multinorm.cdf([bpt, -bpT], mean=[0, 0], cov=nrho_mat))
         - (B / S) ** (k - 1) * K * np.exp(-r * T) * (multinorm.cdf([-ct, -dT], mean=[0, 0], cov=rho_mat)
                                                      - multinorm.cdf([-ct, -cT], mean=[0, 0], cov=rho_mat))
         + (B / S) ** (k + 1) * S * np.exp(-q * T) * (multinorm.cdf([-cpt, -dpT], mean=[0, 0], cov=rho_mat)
                                                      - multinorm.cdf([-cpt, -cpT], mean=[0, 0], cov=rho_mat)))

    return P


def barrier_vector(S, K, H, r, d, sigma, t):

    alpha = (1 / 2) * (1 - (r - d) / ((1 / 2) * (sigma ** 2)))

    P = put(S, K, r, d, sigma, t) \
        - put(S, H, r, d, sigma, t) \
        - (K - H) * put_digit(S, H, r, d, sigma, t)\
        - (S / H) ** (2 * alpha) * (put(H ** 2 / S, K, r, d, sigma, t) - put(H ** 2 / S, H, r, d, sigma, t)
                                    - (K - H) * put_digit(H ** 2 / S, H, r, d, sigma, t))

    P[S < H] = 0

    return P


def put_digit(S, K, r, d, sigma, t):

    d2 = 1 / (sigma * np.sqrt(t)) * (np.log(S / K) + (r - d - (1 / 2) * (sigma ** 2)) * t)

    return np.exp(-r * t) * norm.cdf(-d2)


def put(S, K, r, d, sigma, t):

    d1 = 1 / (sigma * np.sqrt(t)) * (np.log(S / K) + (r - d + (1 / 2) * (sigma ** 2)) * t)
    d2 = 1 / (sigma * np.sqrt(t)) * (np.log(S / K) + (r - d - (1 / 2) * (sigma ** 2)) * t)

    return K * np.exp(-r * t) * norm.cdf(-d2) - S * np.exp(-d * t) * norm.cdf(-d1)


def compute_trueLoss(S_tau, S_0, sigma, r, K, H, tau, T):

    K_1, K_2, K_3 = K[0], K[1], K[2]
    H_1, H_2, H_3 = H[0], H[1], H[2]

    P_d_o10LR = barrier(S_0, K_1, H_1, r, 0, sigma, tau, T)
    P_d_o20LR = barrier(S_0, K_2, H_2, r, 0, sigma, tau, T)
    P_d_o30LR = barrier(S_0, K_3, H_3, r, 0, sigma, tau, T)

    P_d_o1LR = barrier_vector(S_tau, K_1, H_1, r, 0, sigma, T - tau)
    P_d_o2LR = barrier_vector(S_tau, K_2, H_2, r, 0, sigma, T - tau)
    P_d_o3LR = barrier_vector(S_tau, K_3, H_3, r, 0, sigma, T - tau)

    Phi_0 = P_d_o10LR + P_d_o20LR - P_d_o30LR
    Phi_tau = P_d_o1LR + P_d_o2LR - P_d_o3LR

    loss = (Phi_0 - Phi_tau) * np.exp(-r * tau)

    return loss


def compute_P(M, N, K, H, S_T, S_tau, t, r, sigma):

    prob = 1 - np.exp(- (2 * np.log(np.repeat(H / np.maximum(S_tau, H), repeats=N).reshape(M, N))
                         * np.log(H / np.maximum(S_T, H))) / (t * sigma ** 2))

    P = np.exp(-r * t) * (np.maximum(K - S_T, 0) * prob)

    return np.mean(P, axis=1)


def compute_predictors(S_tau, H_1, H_2, H_3):

    x1 = S_tau
    x2 = S_tau ** 2
    x3 = np.maximum(S_tau - H_1, 0)
    x4 = np.maximum(S_tau - H_2, 0)
    x5 = np.maximum(S_tau - H_3, 0)
    x6 = np.maximum(S_tau - H_1, 0) ** 2
    x7 = np.maximum(S_tau - H_2, 0) ** 2
    x8 = np.maximum(S_tau - H_3, 0) ** 2

    return np.vstack((x1, x2, x3, x4, x5, x6, x7, x8)).T


def simPredictors(M, N, S_0, K, H, mu, sigma, r, tau, T, seed=22, crude=True, loss_flag=True, lr_flag=False):

    np.random.seed(seed)

    Z = np.random.normal(0, 1, M)
    drift = (mu - (1/2) * sigma**2) * tau
    diffusion = np.sqrt(tau) * sigma * Z

    S_tau = S_0 * np.exp(drift + diffusion)

    if lr_flag:

        h = 1/156

        t = h
        exponent = (r - (1 / 2) * sigma ** 2) * t + np.sqrt(t) * sigma * np.random.normal(size=(M, N))
        S_h = (np.matmul(S_tau.reshape(-1, 1), np.ones([1, N]))) * np.exp(exponent)

        t = T - tau - h
        exponent = (r - (1 / 2) * sigma ** 2) * t + np.sqrt(t) * sigma * np.random.normal(size=(M, N))
        S_T = S_h * np.exp(exponent)

    else:

        t = T - tau
        exponent = (r - (1 / 2) * sigma ** 2) * t + np.sqrt(t) * sigma * np.random.normal(size=(M, N))

        S_T = (np.matmul(S_tau.reshape(-1, 1), np.ones([1, N]))) * np.exp(exponent)

    K_1, K_2, K_3 = K[0], K[1], K[2]
    H_1, H_2, H_3 = H[0], H[1], H[2]
    
    # Computing P1, P2, P3
    P_1 = compute_P(M, N, K_1, H_1, S_T, S_tau, t, r, sigma)
    P_2 = compute_P(M, N, K_2, H_2, S_T, S_tau, t, r, sigma)
    P_3 = compute_P(M, N, K_3, H_3, S_T, S_tau, t, r, sigma)

    # Compute V1, V2, V3

    V_tau = P_1 + P_2 - P_3

    V_0 = barrier(S_0, K_1, H_1, r, 0, sigma, tau, T) \
          + barrier(S_0, K_2, H_2, r, 0, sigma, tau, T) \
          - barrier(S_0, K_3, H_3, r, 0, sigma, tau, T)

    loss = (V_0 - V_tau) * np.exp(-r * tau)

    if crude:
        X = S_tau
    else:
        X = compute_predictors(S_tau, H_1, H_2, H_3)

    if loss_flag:
        y = loss
    else:
        y = S_T

    # loss.sort()
    #
    # L0 = loss[int(np.ceil((1 - alpha) * M))]
    #
    # indicator = alpha
    # hockey = np.mean(np.maximum(loss - L0, 0))
    # quadratic = np.mean((loss - L0) ** 2)
    # VaR = L0
    # CVaR = np.mean(loss[loss > L0])

    # return indicator, hockey, quadratic, VaR, CVaR
    if lr_flag:
        return X, y, S_h
    else:
        return X, y


def simTrue(M, S_0, K, H, mu, sigma, r, tau, T, alpha=0.1, seed=22):

    np.random.seed(seed)

    Z = np.random.normal(0, 1, M)
    drift = (mu - (1/2) * sigma**2) * tau
    diffusion = np.sqrt(tau) * sigma * Z

    S_tau = S_0 * np.exp(drift + diffusion)

    K_1, K_2, K_3 = K[0], K[1], K[2]
    H_1, H_2, H_3 = H[0], H[1], H[2]

    P_d_o10LR = barrier(S_0, K_1, H_1, r, 0, sigma, tau, T)
    P_d_o20LR = barrier(S_0, K_2, H_2, r, 0, sigma, tau, T)
    P_d_o30LR = barrier(S_0, K_3, H_3, r, 0, sigma, tau, T)

    P_d_o1LR = barrier_vector(S_tau, K_1, H_1, r, 0, sigma, T - tau)
    P_d_o2LR = barrier_vector(S_tau, K_2, H_2, r, 0, sigma, T - tau)
    P_d_o3LR = barrier_vector(S_tau, K_3, H_3, r, 0, sigma, T - tau)

    Phi_0 = P_d_o10LR + P_d_o20LR - P_d_o30LR
    Phi_tau = P_d_o1LR + P_d_o2LR - P_d_o3LR

    loss = (Phi_0 - Phi_tau) * np.exp(-r * tau)

    order = loss.argsort()
    loss_sorted = loss[order]

    L0 = loss_sorted[int(np.ceil((1 - alpha) * M))]

    indicator = alpha
    hockey = np.mean(np.maximum(loss - L0, 0))
    quadratic = np.mean((loss - L0) ** 2)
    VaR = L0
    CVaR = np.mean(loss[loss > L0])

    return indicator, hockey, quadratic, VaR, CVaR, loss, S_tau
