import sns
import optionPricing
import helper

import numpy as np
from joblib import Parallel, delayed


def simulateTrueValues(M, d, rho, S_0, K, mu, sigma, r, tau, T, 
                       levels=[0.8, 0.9, 0.95, 0.99, 0.996], benchmark=0.9,
                       option_name=["European"], option_type=["C"], position=["long"],
                       n_jobs=4, verbose=10):
    
    """
    Simulate the true values of the risk measures.

    :param M: number of outer scenarios
    :param d: dimension of the underlying asset
    :param rho: correlation coefficient of the underlying assets
    :param S_0: initial value of the asset
    :param K: strike price, a list of strike prices for different options
    :param mu: drift of the asset
    :param sigma: volatility of the asset
    :param r: risk-free interest rate
    :param tau: time to maturity
    :param T: time horizon
    :param levels: levels of the risk measures
    :param benchmark: benchmark level for quadratic, hockey stick and indicator
                can be a number between 0 and 1 to represent a quantile
                or a number greater than 1 to represent the actual value
    :param option_name: name of the option
    :param option_type: type of the option
    :param position: position of the option
    :paparam n_jobs: number of jobs
    :param verbose: verbosity level

    :return: true values of the risk measures, in a dictionary
    """

    if len(option_name) != len(K):
        option_name = option_name * len(K)
        
    if len(option_type) != len(K):
        option_type = option_type * len(K)

    if len(position) != len(K):
        position = position * len(K)

    cov_mat = helper.generate_cor_mat(d, rho) * sigma ** 2
    path = True

    if ("Asian" in option_name) or ("Barrier" in option_name):
        path = True
    else:
        path = False

    outerScenarios = sns.simOuter(M, d, S_0, mu, cov_mat, tau, path=path)

    value_tau = np.zeros(M)
    for k, n, t, p in zip(K, option_name, option_type, position):
        if n == "European":
            if path:
                value_parallel = Parallel(n_jobs=n_jobs, verbose=verbose)(
                    delayed(optionPricing.priceVanilla)(outerScenarios[i, :, -1], T - tau, sigma, r, k, 0, t, p)
                    for i in range(d))
            else:
                value_parallel = Parallel(n_jobs=n_jobs, verbose=verbose)(
                    delayed(optionPricing.priceVanilla)(outerScenarios[i, :], T - tau, sigma, r, k, 0, t, p)
                    for i in range(d))
        elif n == "Asian":
            value_parallel = Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(optionPricing.priceDiscreteGeoAsian_tau)(outerScenarios[i, :, :], T - tau, sigma, r, k, 0, t, p)
                for i in range(d))
        elif n == "Barrier":
            value_parallel = Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(optionPricing.priceBarrier_tau)(outerScenarios[i, :, :], T - tau, sigma, r, k, 0, t, p)
                for i in range(d))
        else:
            raise ValueError("Option name not recognized.")
        
        value_tau += np.sum(np.array(value_parallel), axis=0)
    
    value_0 = 0

    for k, n, t, p in zip(K, option_name, option_type, position):
        if n == "European":
            value_0 += optionPricing.priceVanilla(S_0, T, sigma, r, k, 0, t, p)
        elif n == "Asian":
            value_0 += optionPricing.priceDiscreteGeoAsian_0(S_0, T, sigma, r, k, 0, t, p)
        elif n == "Barrier":
            value_0 += optionPricing.priceBarrier_0(S_0, T, sigma, r, k, 0, t, p)
        else:
            raise ValueError("Option name not recognized.")
        
    loss = d * value_0 - value_tau

    print(value_0)

    loss.sort()

    if benchmark < 1:
        indicator = 1 - benchmark
        benchmark = loss[int(np.ceil(benchmark * M))]
    else:
        indicator = np.mean(loss > benchmark)
    
    hockey = np.mean(np.maximum(loss - benchmark, 0))
    quadratic = np.mean((loss - benchmark) ** 2)

    VaR = {}
    CVaR = {}

    for l in levels:
        VaR[l] = loss[int(np.ceil(l * M))]
        CVaR[l] = np.mean(loss[loss >= VaR[l]])
    
    res_dict = {"Indicator": indicator,
                "Hockey": hockey,
                "Quadratic": quadratic,
                "VaR": VaR,
                "CVaR": CVaR}
    
    return res_dict





