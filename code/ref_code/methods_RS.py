import numpy as np
from scipy.stats.distributions import norm


def RS_front(n_front, d, S_0,
             drift_vec_1, diffusion_mat_1, drift_vec_2, diffusion_mat_2,
             transition_mat,
             tau, step_size):

    drift_vec_1 = np.full(d, drift_vec_1)
    diffusion_mat_1 = np.array(diffusion_mat_1)
    A_1 = np.linalg.cholesky(diffusion_mat_1)

    drift_vec_2 = np.full(d, drift_vec_2)
    diffusion_mat_2 = np.array(diffusion_mat_2)
    A_2 = np.linalg.cholesky(diffusion_mat_2)

    n_step = int(tau // step_size) + 1
    S_front = np.zeros([d, n_front, n_step + 1])
    S_front[:, :, 0] = np.array(S_0)
    current_regime = np.zeros(n_front)

    for n in range(n_front):

        for k in range(1, n_step + 1):

            transition_prob = (1 - current_regime[n]) * transition_mat[0, 1] + current_regime[n] * transition_mat[1, 0]
            rand = np.random.uniform()
            if rand > transition_prob:
                current_regime[n] = 1 - current_regime[n]

            current_drift_vec = (1 - current_regime[n]) * drift_vec_1 + current_regime[n] * drift_vec_2
            current_diffusion_mat = (1 - current_regime[n]) * diffusion_mat_1 + current_regime[n] * diffusion_mat_2
            current_A = (1 - current_regime[n]) * A_1 + current_regime[n] * A_2

            Z = np.random.normal(0, 1, d)
            drift = (current_drift_vec - (1 / 2) * np.diagonal(current_diffusion_mat)) * step_size
            diffusion = np.sqrt(step_size) * np.matmul(current_A, Z)
            S_front[:, n, k] = S_front[:, n, k - 1] * np.exp(drift + diffusion)

    return S_front, current_regime


def RS_back(n_front, n_back, d, S_tau,
            drift_1, diffusion_1, drift_2, diffusion_2,
            transition_mat, current_regime,
            T, step_size):

    drift_vec_1 = np.full(d, drift_1)
    diffusion_mat_1 = np.array(diffusion_1)
    A_1 = np.linalg.cholesky(diffusion_mat_1)

    drift_vec_2 = np.full(d, drift_2)
    diffusion_mat_2 = np.array(diffusion_2)
    A_2 = np.linalg.cholesky(diffusion_mat_2)

    n_step = int(T // step_size) + 1
    S_back = np.zeros([d, n_front, n_back, n_step + 1])
    for j in range(n_back):
        S_back[:, :, j, 0] = S_tau[:, :, -1]

    for i in range(n_front):

        current_regime_back = np.full(n_back, current_regime[i])
        for n in range(n_back):

            current_drift_vec = (1 - current_regime_back[n]) * drift_vec_1 + current_regime_back[n] * drift_vec_2
            current_diffusion_mat = (1 - current_regime_back[n]) * diffusion_mat_1 + \
                                    current_regime_back[n] * diffusion_mat_2
            current_A = (1 - current_regime_back[n]) * A_1 + current_regime_back[n] * A_2

            for k in range(1, n_step + 1):

                Z = np.random.normal(0, 1, d)
                drift = (current_drift_vec - (1 / 2) * np.diagonal(current_diffusion_mat)) * step_size
                drift = drift.reshape(-1, 1)
                diffusion = np.sqrt(step_size) * np.matmul(current_A, Z)
                S_back[:, i, n, k] = S_back[:, i, n, k - 1] * np.exp(drift + diffusion)

                transition_prob = (1 - current_regime_back[n]) * transition_mat[0, 1] \
                                  + current_regime_back[n] * transition_mat[1, 0]
                rand = np.random.uniform()
                if rand > transition_prob:
                    current_regime_back[n] = 1 - current_regime_back[n]

    return S_back


def find_R_dist(transition_mat, n_steps, current_regime=-1):

    pi_1 = transition_mat[1, 0] / (transition_mat[1, 0] + transition_mat[0, 1])
    pi_2 = 1 - pi_1

    R_vec = np.zeros([n_steps, n_steps + 1, 2])
    R_vec[n_steps - 1, 0, 0] = transition_mat[0, 1]
    R_vec[n_steps - 1, 1, 0] = transition_mat[0, 0]
    R_vec[n_steps - 1, 0, 1] = transition_mat[1, 1]
    R_vec[n_steps - 1, 1, 1] = transition_mat[1, 0]

    for t in range(n_steps - 2, -1, -1):
        for r in range(n_steps, 0, -1):
            R_vec[t, r, 0] = transition_mat[0, 0] * R_vec[t + 1, r - 1, 0] \
                             + transition_mat[0, 1] * R_vec[t + 1, r, 1]
            R_vec[t, r, 1] = transition_mat[1, 0] * R_vec[t + 1, r - 1, 0] \
                             + transition_mat[1, 1] * R_vec[t + 1, r, 1]

        R_vec[t, 0, 0] = transition_mat[0, 1] * R_vec[t + 1, 0, 1]
        R_vec[t, 0, 1] = transition_mat[1, 1] * R_vec[t + 1, 0, 1]

    if current_regime >= 0:
        R = R_vec[0, :, int(current_regime)]
    else:
        R = np.zeros(n_steps + 1)
        for r in range(n_steps + 1):
            R[r] = pi_1 * R_vec[0, r, 0] + pi_2 * R_vec[0, r, 1]

    return R


def price_CP(S, T, sigma_1, sigma_2, r, K, transition_mat, current_regime, n_steps, option_type, position):

    R = find_R_dist(transition_mat, n_steps, current_regime)

    price = np.zeros(n_steps + 1)

    if option_type == 'C':
        for i in range(n_steps + 1):
            t = (i / n_steps) * T
            d1_r = (1 / np.sqrt(t * sigma_1**2 + (T - t) * sigma_2**2)) \
                   * (np.log(S / K) + T * r + t * sigma_1**2 / 2 + (T - t) * sigma_2**2 / 2)
            d2_r = d1_r - np.sqrt(t * sigma_1**2 + (T - t) * sigma_2**2)
            price[i] = norm.cdf(d1_r) * S - norm.cdf(d2_r) * K * np.exp(-r * T)
    else:
        for i in range(n_steps + 1):
            t = (i / n_steps) * T
            d1_r = (1 / np.sqrt(t * sigma_1**2 + (T - t) * sigma_2**2)) \
                   * (np.log(S / K) + T * r + t * sigma_1**2 / 2 + (T - t) * sigma_2**2 / 2)
            d2_r = d1_r - np.sqrt(t * sigma_1**2 + (T - t) * sigma_2**2)
            price[i] = norm.cdf(-d2_r) * K * np.exp(-r * T) - norm.cdf(-d1_r) * S

    if position == "short":
        price = - price

    return np.sum(R * price)
