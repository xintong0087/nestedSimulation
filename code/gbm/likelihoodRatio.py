import numpy as np


def compute_Weight_D(S_0, mu, r, sigma, h, tau, sample_inner_tau):
    return (np.log(sample_inner_tau / S_0) - (mu - (1 / 2) * sigma ** 2) * tau - (
                r - (1 / 2) * sigma ** 2) * h) ** 2 / (2 * sigma ** 2 * (tau + h))


def compute_Weight_U(sample_inner_tau, sample_outer_test, r, sigma, h):

    log_ratio = np.log(sample_inner_tau.reshape(-1, 1) / sample_outer_test.reshape(1, -1))

    return (log_ratio - (r - sigma ** 2 / 2) * h) ** 2 / (2 * sigma ** 2 * h)


def likelihoodRatio(M, N, d, S_0, K, mu, sigma, rho, r, tau, T, option_name, option_type, position):

    """
    Implements the nested simulation procedure with likelihood ratio

    return:
        loss: likelihood-ratio-weighted scenario losses
    """

    return None