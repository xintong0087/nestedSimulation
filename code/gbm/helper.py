import numpy as np
from scipy.stats import multivariate_normal as multinorm


def generate_cor_mat(d, rho):
    
    """
    Generate the correlation matrix.

    :param d: dimension
    :param rho: correlation coefficient

    :return: correlation matrix
    """

    return np.ones((d, d)) * rho + np.identity(d) * (1 - rho)


def biNormCDF(x, y, rho):

    """
    Calculate the bivariate normal CDF.

    :param x: x value
    :param y: y value
    :param rho: correlation coefficient

    :return: bivariate normal CDF
    """
    
    rho_mat = np.array([[1, rho], [rho, 1]])

    return multinorm.cdf([x, y], mean=[0, 0], cov=rho_mat)


def calculatePayoff(outerScenarios, innerPaths, K, r, tau, T, 
                    option_name=["European"], option_type=["C"], position=["long"],
                    barrier_info=None):

    """
    Calculate the payoff of the option given sample paths of the underlying asset.

    :param outerScenarios: outer scenarios
    :param innerPaths: inner paths
    :param K: strike price, a list of strike prices for different options
    :param option_type: option type, a list of types for different options

    :return: payoff of the option for each scenario, averaged over the inner paths, 
                                                     summed over the assets.
    """

    if len(option_name) != len(K):
        option_name = option_name * len(K)
        
    if len(option_type) != len(K):
        option_type = option_type * len(K)

    if len(position) != len(K):
        position = position * len(K)
    
    d = outerScenarios.shape[0]
    M = outerScenarios.shape[1]
    N = innerPaths.shape[2]

    payoff = np.zeros([d, M, N])
    
    if option_name == ["European"]:

        # outerScenarios.shape = [d, M]
        # innerPaths.shape = [d, M, N]

        for k, n, t, p in zip(K, option_name, option_type, position):

            
            multiplier_position = 1 if p == "long" else -1
            multiplier_CP = 1 if t == "C" else -1

            payoff += multiplier_position * np.maximum(multiplier_CP * (innerPaths - k), 0) * np.exp(-r * (T - tau))

    elif option_name == "Asian":

        # outerScenarios.shape = [d, M, n_step(0->tau) + 1]
        # innerPaths.shape = [d, M, N, n_step(tau->T) + 1]

        multiplier_position = 1 if p == "long" else -1
        multiplier_CP = 1 if t == "C" else -1

        n_step = outerScenarios.shape[2] + innerPaths.shape[3] - 2

        geometric_sum_outer = np.expand_dims(np.prod(outerScenarios[:, :, 1:], axis=2), axis=2)
        geometric_sum_inner = np.prod(innerPaths[:, :, :, 1:], axis=3)

        for k, n, t, p in zip(K, option_name, option_type, position):
            payoff += np.maximum(multiplier_CP * ((geometric_sum_outer * geometric_sum_inner) ** (1 / n_step) - k), 0) 

    elif option_name == "Barrier":

        # outerScenarios.shape = [d, M, n_step(0->tau) + 1]
        # innerPaths.shape = [d, M, N, n_step(tau->T) + 1]

        sigma = barrier_info[0]
        H = barrier_info[1]
        step_size = barrier_info[2]

        d = outerScenarios.shape[0]
        M = outerScenarios.shape[1]
        N = innerPaths.shape[2]

        n_outer = outerScenarios.shape[2] - 1
        n_inner = innerPaths.shape[3] - 1

        # Simulate the maximum and minimum of the outer scenarios
        U = np.random.uniform(low=0.0, high=1.0, size=[d, M, n_outer])
        sample_outer_max = np.zeros_like(U)
        sample_outer_min = np.zeros_like(U)
        for n in range(n_outer):
            sample_outer_max[:, :, n] = np.exp((np.log(outerScenarios[:, :, n] * outerScenarios[:, :, n+1])
                    + np.sqrt(np.log(outerScenarios[:, :, n+1] / outerScenarios[:, :, n])**2
                    - 2 * sigma**2 * step_size * np.log(U[:, :, n]))) / 2)
            sample_outer_min[:, :, n] = np.exp((np.log(outerScenarios[:, :, n] * outerScenarios[:, :, n+1])
                    - np.sqrt(np.log(outerScenarios[:, :, n+1] / outerScenarios[:, :, n])**2
                    - 2 * sigma**2 * step_size * np.log(U[:, :, n]))) / 2)
        
        outer_max = np.expand_dims(np.max(sample_outer_max, axis=2), axis=2)
        outer_min = np.expand_dims(np.min(sample_outer_min, axis=2), axis=2)

        # Simulate the maximum and minimum of the inner paths
        U = np.random.uniform(low=0.0, high=1.0, size=[d, M, N, n_inner])
        sample_inner_max = np.zeros_like(U)
        sample_inner_min = np.zeros_like(U)

        for i in range(N):
            for n in range(n_inner):
                sample_inner_max[:, :, i, n] = np.exp((np.log(innerPaths[:, :, i, n] * innerPaths[:, :, i, n+1])
                    + np.sqrt(np.log(innerPaths[:, :, i, n+1] / innerPaths[:, :, i, n])**2
                    - 2 * sigma**2 * step_size * np.log(U[:, :, i, n]))) / 2)
                sample_inner_min[:, :, i, n] = np.exp((np.log(innerPaths[:, :, i, n] * innerPaths[:, :, i, n+1])
                    - np.sqrt(np.log(innerPaths[:, :, i, n+1] / innerPaths[:, :, i, n])**2
                    - 2 * sigma**2 * step_size * np.log(U[:, :, i, n]))) / 2)
                
        inner_max = np.max(sample_inner_max, axis=3)
        inner_min = np.min(sample_inner_min, axis=3)

        # Calculate the payoff
        
        S_T = innerPaths[:, :, :, -1]
        for k, h, t, p in zip(K, H, option_type, position):

            multiplier_position = 1 if p == "long" else -1
            multiplier_CP = 1 if "C" in t else -1

            if "DO" in t:
                payoff += multiplier_position * (outer_min > h) * (inner_min > h) * np.maximum(multiplier_CP * (S_T - k), 0) 
            elif "DI" in t:
                payoff += multiplier_position * (outer_min < h) * (inner_min < h) * np.maximum(multiplier_CP * (S_T - k), 0)
            elif "UO" in t:
                payoff += multiplier_position * (outer_max < h) * (inner_max < h) * np.maximum(multiplier_CP * (S_T - k), 0)
            elif "UI" in t:
                payoff += multiplier_position * (outer_max > h) * (inner_max > h) * np.maximum(multiplier_CP * (S_T - k), 0)
                 
    return np.sum(np.exp(-r * (T - tau)) * np.mean(payoff, axis=2), axis=0)


def calculateRMSE(threshold, y_pred, alpha):

    """
    Calculate the RMSE between the true value and the predicted value.

    :param threshold: threshold for calculating indicator, hockey stick, quadratic
    :param y_pred: predicted value
    :param alpha: alpha for calculating VaR, CVaR

    :return: indicator, hockey stick, quadratic, VaR, CVaR
    """

    indicator = np.mean((y_pred > threshold))
    hockey = np.mean(np.maximum(y_pred - threshold, 0))
    quadratic = np.mean((y_pred - threshold) ** 2)

    y_pred.sort()
    VaR = y_pred[int(np.ceil((1 - alpha) * y_pred.shape[0]))]
    CVaR = np.mean(y_pred[y_pred >= VaR])

    return indicator, hockey, quadratic, VaR, CVaR
