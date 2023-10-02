import numpy as np
from scipy.stats.distributions import norm
from helper import biNormCDF


def priceVanilla(S, T, sigma, r, K, q, option_type, position):

    """
    Calculate the closed-form Black-Scholes price of a vanilla option (call or put)

    :param S: asset price at current time, can be a vector or a matrix
    :param T: time to maturity
    :param sigma: volatility
    :param r: risk-free rate
    :param K: strike price
    :param q: dividend yield
    :param option_type: "C" or "P"
    :param position: "long" or "short"

    :return: price of the option, can be a vector
    """

    d1 = (1 / (sigma * np.sqrt(T))) * (np.log(S / K) + (r - q + sigma ** 2 / 2) * T)
    d2 = d1 - (sigma * np.sqrt(T))

    if option_type != "P":
        price = S * np.exp(-q * T) * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)
    else:
        price = np.exp(-r * T) * (K * norm.cdf(-d2)) - S * np.exp(-q * T) * norm.cdf(-d1)

    if position == "short":
        price = -price

    return price



def priceDiscreteGeoAsian_0(S, T, sigma, r, K, q, n_step, option_type, position):

    """
    Calculate the closed-form Black-Scholes price of a geometric Asian option (discretely monitored)
    INCLUDE REFERENCE, CHECK VALIDITY

    :param S: asset price at time 0, can be a vector
    :param T: time to maturity
    :param sigma: volatility
    :param r: risk-free rate
    :param K: strike price
    :param q: dividend yield
    :param n_step: number of time steps
    :param option_type: "C" or "P"
    :param position: "long" or "short"

    :return: price of the option, can be a vector
    """

    price = 0

    sigmaG = sigma * np.sqrt((n_step + 1) * (2 * n_step + 1) / (6 * n_step ** 2))
    muG = 0.5 * ((2 * r * (n_step - 1) + sigma**2 * (n_step + 1)) / (2 * n_step) + sigmaG ** 2)

    d1 = (np.log(S / K) + (r - muG + sigmaG ** 2) * T) / (sigmaG * np.sqrt(T))
    d2 = d1 - sigmaG * np.sqrt(T)

    if option_type != "P":
        price = np.exp(-muG * T) * (S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    else:
        price = np.exp(-muG * T) * (K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1))

    if position == "short":
        price = -price

    return price


def priceDiscreteGeoAsian_tau(S, T, sigma, r, K, q, n_inner, option_type, position):

    """
    Calculate the closed-form Black-Scholes price of a geometric Asian option (discretely monitored)
    INCLUDE REFERENCE, CHECK VALIDITY

    :param S: asset path from time 0 to tau, must be 2-dimensional (reshape if necessary)
    :param T: time to maturity
    :param sigma: volatility
    :param r: risk-free rate
    :param K: strike price
    :param q: dividend yield
    :param n_inner: number of time steps for inner paths
    :param option_type: "C" or "P"
    :param position: "long" or "short"
    :param tau: the time to the first observation

    :return: price of the option, can be a vector
    """

    price = 0
    S_tau = S[:, -1]
    S_prod = np.prod(S, axis=1)

    n_outer = S.shape[1] - 1
    n = n_outer + n_inner

    sigmaG = (n_inner/n) * sigma * np.sqrt(1/n) * np.sqrt((n_inner + 1) * (2 * n_inner + 1) / (6 * n_inner))
    muG = r - (r - (sigma ** 2) / 2) * (1/n)**2 * (n_inner*(n_inner + 1) / 2) \
        - 0.5 * (n_inner/n)**2 * sigma**2 * (1/n) * ((n_inner + 1) * (2 * n_inner + 1) / (6 * n_inner))
    
    d1 = (np.log(S_tau**(n_inner/n) * S_prod**(1/n) / K) + (r - muG + 0.5 * sigmaG**2)) / sigmaG
    d2 = d1 - sigmaG

    if option_type != "P":
        price = np.exp(n_outer/n * r * T) * S_prod**(1/n) * (np.exp(-muG * T) * S_tau**(n_inner/n) * norm.cdf(d1) 
                                                             - K * np.exp(-r * T) * S_prod**(1/n) * norm.cdf(d2))
    else:
        price = np.exp(n_outer/n * r * T) * S_prod**(1/n) * (K * np.exp(-r * T) * S_prod**(1/n) * norm.cdf(-d2) 
                                                             - np.exp(-muG * T) * S_tau**(n_inner/n) * norm.cdf(-d1))
    
    if position == "short":
        price = -price

    return price


def priceBarrier_0(S, K, T, sigma, r, H, q, R, option_type, position):

    """
    Calculate the closed-form Black-Scholes price of a barrier option (call or put) at time 0
    Reference: Haug, 2006

    :param S: asset price at current time, can be a vector
    :param K: strike price
    :param T: time to maturity
    :param sigma: volatility
    :param r: risk-free rate
    :param H: barrier level
    :param q: dividend yield
    :param R: rebate
    :param option_type: a length-3 list, for example ["U", "I", "C"] means an up-and-in call option
    :param position: "long" or "short"

    :return: price of the option, scalar
    """

    S = np.array(S)
    length_S = S.shape[0]
    phi = np.zeros(length_S)
    nu = np.zeros(length_S)

    phi[(option_type[:, 0] == "U") & (option_type[:, 2] == "C")] = 1
    nu[(option_type[:, 0] == "U") & (option_type[:, 2] == "C")] = -1

    phi[(option_type[:, 0] == "U") & (option_type[:, 2] == "P")] = -1
    nu[(option_type[:, 0] == "U") & (option_type[:, 2] == "P")] = -1

    phi[(option_type[:, 0] == "D") & (option_type[:, 2] == "C")] = 1
    nu[(option_type[:, 0] == "D") & (option_type[:, 2] == "C")] = 1

    phi[(option_type[:, 0] == "D") & (option_type[:, 2] == "P")] = -1
    nu[(option_type[:, 0] == "D") & (option_type[:, 2] == "P")] = 1

    sigma_sq = sigma ** 2
    sigma_T = sigma * np.sqrt(T)

    mu = (r - q - sigma_sq / 2) / sigma_sq
    _lambda = np.sqrt(mu ** 2 + 2 * r / sigma_sq)
    z = np.log(H / S) / sigma_T + _lambda * sigma_T

    mu_sigma_T = (1 + mu) * sigma_T

    x1 = np.log(S / K) / sigma_T + mu_sigma_T
    x2 = np.log(S / H) / sigma_T + mu_sigma_T
    y1 = np.log(H ** 2 / (S * K)) / sigma_T + mu_sigma_T
    y2 = np.log(H / S) / sigma_T + mu_sigma_T

    A = phi * S * np.exp(-q * T) * norm.cdf(phi * x1, 0, 1) \
        - phi * K * np.exp(-r * T) * norm.cdf(phi * (x1 - sigma_T), 0, 1)
    B = phi * S * np.exp(-q * T) * norm.cdf(phi * x2, 0, 1) \
        - phi * K * np.exp(-r * T) * norm.cdf(phi * (x2 - sigma_T), 0, 1)
    C = phi * S * np.exp(-q * T) * (H / S) ** (2 * (mu + 1)) * norm.cdf(nu * y1, 0, 1) \
        - phi * K * np.exp(-r * T) * (H / S) ** (2 * mu) * norm.cdf(nu * (y1 - sigma_T), 0, 1)
    D = phi * S * np.exp(-q * T) * (H / S) ** (2 * (mu + 1)) * norm.cdf(nu * y2, 0, 1) \
        - phi * K * np.exp(-r * T) * (H / S) ** (2 * mu) * norm.cdf(nu * (y2 - sigma_T), 0, 1)
    E = R * np.exp(-r * T) * (norm.cdf(nu * (x2 - sigma_T))
                              - (H / S) ** (2 * mu) * norm.cdf(nu * (y2 - sigma_T)))
    F = R * ((H / S) ** (mu + _lambda) * (norm.cdf(nu * z))
             - (H / S) ** (mu - _lambda) * norm.cdf(nu * (z - 2 * _lambda * sigma_T)))

    price = np.zeros(length_S)

    # Up and In Options
    IUC_aboveBarrier_flag = (option_type[:, 1] == "I") \
                            & (option_type[:, 0] == "U") \
                            & (option_type[:, 2] == "C") \
                            & (S >= H)
    price[IUC_aboveBarrier_flag] = A[IUC_aboveBarrier_flag] + E[IUC_aboveBarrier_flag]

    IUC_belowBarrier_flag = (option_type[:, 1] == "I") \
                            & (option_type[:, 0] == "U") \
                            & (option_type[:, 2] == "C") \
                            & (S < H)
    price[IUC_belowBarrier_flag] = B[IUC_belowBarrier_flag] - C[IUC_belowBarrier_flag] + D[IUC_belowBarrier_flag] \
                                   + E[IUC_belowBarrier_flag]

    IUP_aboveBarrier_flag = (option_type[:, 1] == "I") \
                            & (option_type[:, 0] == "U") \
                            & (option_type[:, 2] == "P") \
                            & (S >= H)
    price[IUP_aboveBarrier_flag] = A[IUP_aboveBarrier_flag] - B[IUP_aboveBarrier_flag] + D[IUP_aboveBarrier_flag] \
                                   + E[IUP_aboveBarrier_flag]

    IUP_belowBarrier_flag = (option_type[:, 1] == "I") \
                            & (option_type[:, 0] == "U") \
                            & (option_type[:, 2] == "P") \
                            & (S < H)
    price[IUP_belowBarrier_flag] = C[IUP_belowBarrier_flag] + E[IUP_belowBarrier_flag]

    # Down and In Options
    IDC_aboveBarrier_flag = (option_type[:, 1] == "I") \
                            & (option_type[:, 0] == "D") \
                            & (option_type[:, 2] == "C") \
                            & (S >= H)
    price[IDC_aboveBarrier_flag] = C[IDC_aboveBarrier_flag] + E[IDC_aboveBarrier_flag]

    IDC_belowBarrier_flag = (option_type[:, 1] == "I") \
                            & (option_type[:, 0] == "D") \
                            & (option_type[:, 2] == "C") \
                            & (S < H)
    price[IDC_belowBarrier_flag] = A[IDC_belowBarrier_flag] - B[IDC_belowBarrier_flag] + D[IDC_belowBarrier_flag] \
                                   + E[IDC_belowBarrier_flag]

    IDP_aboveBarrier_flag = (option_type[:, 1] == "I") \
                            & (option_type[:, 0] == "D") \
                            & (option_type[:, 2] == "P") \
                            & (S >= H)
    price[IDP_aboveBarrier_flag] = B[IDP_aboveBarrier_flag] - C[IDP_aboveBarrier_flag] + D[IDP_aboveBarrier_flag] \
                                   + E[IDP_aboveBarrier_flag]

    IDP_belowBarrier_flag = (option_type[:, 1] == "I") \
                            & (option_type[:, 0] == "D") \
                            & (option_type[:, 2] == "P") \
                            & (S < H)
    price[IDP_belowBarrier_flag] = A[IDP_belowBarrier_flag] + E[IDP_belowBarrier_flag]

    # Up and Out Options
    OUC_aboveBarrier_flag = (option_type[:, 1] == "O") \
                            & (option_type[:, 0] == "U") \
                            & (option_type[:, 2] == "C") \
                            & (S >= H)
    price[OUC_aboveBarrier_flag] = F[OUC_aboveBarrier_flag]

    OUC_belowBarrier_flag = (option_type[:, 1] == "O") \
                            & (option_type[:, 0] == "U") \
                            & (option_type[:, 2] == "C") \
                            & (S < H)
    price[OUC_belowBarrier_flag] = A[OUC_belowBarrier_flag] - B[OUC_belowBarrier_flag] + C[OUC_belowBarrier_flag] \
                                   - D[OUC_belowBarrier_flag] + F[OUC_belowBarrier_flag]

    OUP_aboveBarrier_flag = (option_type[:, 1] == "O") \
                            & (option_type[:, 0] == "U") \
                            & (option_type[:, 2] == "P") \
                            & (S >= H)
    price[OUP_aboveBarrier_flag] = B[OUP_aboveBarrier_flag] - D[OUP_aboveBarrier_flag] + F[OUP_aboveBarrier_flag]

    OUP_belowBarrier_flag = (option_type[:, 1] == "O") \
                            & (option_type[:, 0] == "U") \
                            & (option_type[:, 2] == "P") \
                            & (S < H)
    price[OUP_belowBarrier_flag] = A[OUP_belowBarrier_flag] - C[OUP_belowBarrier_flag] + F[OUP_belowBarrier_flag]

    # Down and Out Options
    ODC_aboveBarrier_flag = (option_type[:, 1] == "O") \
                            & (option_type[:, 0] == "D") \
                            & (option_type[:, 2] == "C") \
                            & (S >= H)
    price[ODC_aboveBarrier_flag] = A[ODC_aboveBarrier_flag] - C[ODC_aboveBarrier_flag] + F[ODC_aboveBarrier_flag]

    ODC_belowBarrier_flag = (option_type[:, 1] == "O") \
                            & (option_type[:, 0] == "D") \
                            & (option_type[:, 2] == "C") \
                            & (S < H)
    price[ODC_belowBarrier_flag] = B[ODC_belowBarrier_flag] - D[ODC_belowBarrier_flag] + F[ODC_belowBarrier_flag]

    ODP_aboveBarrier_flag = (option_type[:, 1] == "O") \
                            & (option_type[:, 0] == "D") \
                            & (option_type[:, 2] == "P") \
                            & (S >= H)
    price[ODP_aboveBarrier_flag] = A[ODP_aboveBarrier_flag] - B[ODP_aboveBarrier_flag] + C[ODP_aboveBarrier_flag] \
                                   - D[ODP_aboveBarrier_flag] + F[ODP_aboveBarrier_flag]

    ODP_belowBarrier_flag = (option_type[:, 1] == "O") \
                            & (option_type[:, 0] == "D") \
                            & (option_type[:, 2] == "P") \
                            & (S < H)
    price[ODP_belowBarrier_flag] = F[ODP_belowBarrier_flag]

    if position == "short":
        price = - price

    return price


def priceBarrier_tau(S, K, T, step_size, sigma, r, H, q, R, option_type, position):

    """
    Calculate the closed-form Black-Scholes price of a barrier option (call or put) at time tau

    :param S: asset path from time 0 to tau, must be 2-dimensional (reshape if necessary)
        dimension 0: M, number of scenarios
        dimension 1: n_step + 1, number of time steps + 1
    :param K: strike price
    :param T: time to maturity
    :param step_size: simulation time interval between time steps
    :param sigma: volatility
    :param r: risk-free rate
    :param H: barrier level
    :param q: dividend yield
    :param R: rebate
    :param option_type: a length-3 list, for example ["U", "I", "C"] means an up-and-in call option
    :param position: "long" or "short"
    :param tau: the time to the first observation

    :return: price of the option, can be a vector
    """


    if (option_type[0] == "U") & (option_type[1] == "O"):
        payoutIndicator = (np.max(S, axis=1) < H)
    elif (option_type[0] == "D") & (option_type[1] == "O"):
        payoutIndicator = (np.min(S, axis=1) > H)
    elif (option_type[0] == "U") & (option_type[1] == "I"):
        payoutIndicator = (np.max(S, axis=1) >= H)
    else:
        payoutIndicator = (np.min(S, axis=1) <= H)

    # Get the asset value at time tau
    S = S[:, -1]                 

    # Reshape option_type to match the shape of S
    option_type_mat = np.tile(np.array(option_type), (S.shape[0], 1))

    if option_type[1] == "O":
        # if the option is an out option, only the remaining part of the option is priced
        price = payoutIndicator * priceBarrier_0(S, K, T, sigma, r, H, q, R, option_type_mat, position)
    else:
        # if the option is an in option, the "in" part is priced as a vanilla option
        # and the "not-yet-in" part is priced as a barrier option
        price = payoutIndicator * priceVanilla(S, T, sigma, r, K, q, option_type[2], position) \
            + (1 - payoutIndicator) * priceBarrier_0(S, K, T, sigma, r, H, q, R, option_type_mat, position)

    return price


def pricePTBarrier(S, K, tau, T, sigma, r, H, q, option_type, position):

    """
    Calculate the closed-form Black-Scholes price of a partial time barrier option
    Reference: Haug, 2006

    :param S: asset path from time 0 to tau, 1d only
    :param K: strike price
    :param tau: time after which the barrier becomes active
    :param T: time to maturity
    :param sigma: volatility
    :param r: risk-free rate
    :param H: barrier level
    :param q: dividend yield
    :param option_type: a list of length 4,
        for example ["D", "O", "C", "B2"] means an down-and-out call option of type B2
    :param position: "long" or "short"
    :param tau: the time to the first observation

    :return: price of the option, scalar
    """

    price = 0

    d1 = (np.log(S/K) + (r - q + sigma**2/2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    f1 = (np.log(S/K) + 2 * np.log(H/S) + (r - q + sigma**2/2) * T) / (sigma * np.sqrt(T))
    f2 = f1 - sigma * np.sqrt(T)
    
    e1 = (np.log(S/H) + (r - q + sigma**2/2) * tau) / (sigma * np.sqrt(tau))
    e2 = e1 - sigma * np.sqrt(tau)
    e3 = e1 + 2 * np.log(H/S) / (sigma * np.sqrt(tau))
    e4 = e3 - sigma * np.sqrt(tau)
    
    mu = (r - q - sigma**2/2) / sigma**2 
    rho = np.sqrt(tau/T)
    
    g1 = (np.log(S/H) + (r - q + sigma**2/2) * T) / (sigma * np.sqrt(T))
    g2 = g1 - sigma * np.sqrt(T)
    g3 = g1 + 2 * np.log(H/S) / (sigma * np.sqrt(T))
    g4 = g3 - sigma * np.sqrt(T)
    
    z1 = norm.cdf(e2) - (H/S) ** (2 * mu) * norm.cdf(e4)
    z2 = norm.cdf(-e2) - (H/S) ** (2 * mu) * norm.cdf(-e4)
    z3 = biNormCDF(g2, e2, rho) - (H/S) ** (2 * mu) * biNormCDF(g4, -e4, -rho)
    z4 = biNormCDF(-g2, -e2, rho) - (H/S) ** (2 * mu) * biNormCDF(-g4, e2, -rho)
    z5 = norm.cdf(e1) - (H/S) ** (2 * (mu + 1)) * norm.cdf(e3)
    z6 = norm.cdf(-e1) - (H/S) ** (2 * (mu + 1)) * norm.cdf(-e3)
    z7 = biNormCDF(g1, e1, rho) - (H/S) ** (2 * (mu + 1)) * biNormCDF(g3, -e3, -rho)
    z8 = biNormCDF(-g1, -e1, rho) - (H/S) ** (2 * (mu + 1)) * biNormCDF(-g3, e3, -rho)

    if option_type == ["D", "O", "C", "A"]:
        price = S * np.exp(-q*T) * biNormCDF(d1, e1, rho) \
            - (H/S) ** (2 * (mu + 1)) * biNormCDF(f1, e3, rho) \
            - K * np.exp(-r*T) * biNormCDF(d2, e2, rho) \
            - (H/S) ** (2 * mu) * biNormCDF(f2, e4, rho) 
    
    elif option_type == ["U", "O", "C", "A"]:
        price = S * np.exp(-q*T) * biNormCDF(d1, -e1, -rho) \
            - (H/S) ** (2 * (mu + 1)) * biNormCDF(f1, -e3, -rho) \
            - K * np.exp(-r*T) * biNormCDF(d2, -e2, -rho) \
            - (H/S) ** (2 * mu) * biNormCDF(f2, -e4, -rho)     

    elif option_type == ["D", "O", "C", "B2"]:    
        if K < H:
            price = S * np.exp(-q*T) * biNormCDF(g1, e1, rho) \
            - (H/S) ** (2 * (mu + 1)) * biNormCDF(g3, -e3, -rho) \
            - K * np.exp(-r*T) * biNormCDF(g2, e2, rho) \
            - (H/S) ** (2 * mu) * biNormCDF(g4, -e4, -rho) 
        else:
            price = pricePTBarrier(S, K, tau, T, sigma, r, H, q, [None, "O", "C" "B1"], position)

    elif option_type == ["U", "O", "C", "B2"]:
        if K < H:
            price = S * np.exp(-q*T) * biNormCDF(-g1, -e1, rho) \
            - (H/S) ** (2 * (mu + 1)) * biNormCDF(-g3, e3, -rho) \
            - K * np.exp(-r*T) * biNormCDF(-g2, -e2, rho) \
            - (H/S) ** (2 * mu) * biNormCDF(-g4, e4, -rho) \
            - S * np.exp(-q*T) * biNormCDF(-d1, -e1, rho) \
            - (H/S) ** (2 * (mu + 1)) * biNormCDF(-f1, e3, -rho) \
            + K * np.exp(-r*T) * biNormCDF(-d2, -e2, rho) \
            - (H/S) ** (2 * mu) * biNormCDF(-f2, e4, -rho)
        else:
            raise ValueError("Invalid option type. Type cuoB2 is not a valid option type for K >= H.")

    elif option_type == [None, "O", "C", "B1"]:
        if K < H:
            price = S * np.exp(-q*T) * biNormCDF(-g1, -e1, rho) \
            - (H/S) ** (2 * (mu + 1)) * biNormCDF(-g3, e3, -rho) \
            - K * np.exp(-r*T) * biNormCDF(-g2, -e2, rho) \
            - (H/S) ** (2 * mu) * biNormCDF(-g4, e4, -rho) \
            - S * np.exp(-q*T) * biNormCDF(-d1, -e1, rho) \
            - (H/S) ** (2 * (mu + 1)) * biNormCDF(-f1, e3, -rho) \
            + K * np.exp(-r*T) * biNormCDF(-d2, -e2, rho) \
            - (H/S) ** (2 * mu) * biNormCDF(-f2, e4, -rho) \
            + S * np.exp(-q*T) * biNormCDF(g1, e1, rho) \
            - (H/S) ** (2 * (mu + 1)) * biNormCDF(g3, -e3, -rho) \
            - K * np.exp(-r*T) * biNormCDF(g2, e2, rho) \
            - (H/S) ** (2 * mu) * biNormCDF(g4, -e4, -rho)
        else:
            price = S * np.exp(-q*T) * biNormCDF(d1, e1, rho) \
            - (H/S) ** (2 * (mu + 1)) * biNormCDF(f1, -e3, -rho) \
            - K * np.exp(-r*T) * biNormCDF(d2, e2, rho) \
            - (H/S) ** (2 * mu) * biNormCDF(f2, -e4, -rho)

    elif option_type == ["D", "O", "P", "A"]:
        price = pricePTBarrier(S, K, tau, T, sigma, r, H, q, ["D", "O", "C", "A"], position) \
        - S * np.exp(-q*T) * z5 + K * np.exp(-r*T) * z1 

    elif option_type == ["U", "O", "P", "A"]:
        price = pricePTBarrier(S, K, tau, T, sigma, r, H, q, ["U", "O", "C", "A"], position) \
        - S * np.exp(-q*T) * z6 + K * np.exp(-r*T) * z2

    elif option_type == [None, "O", "P", "B1"]:
        price = pricePTBarrier(S, K, tau, T, sigma, r, H, q, [None, "O", "C", "B1"], position) \
        - S * np.exp(-q*T) * z8 + K * np.exp(-r*T) * z4 \
        - S * np.exp(-q*T) * z7 + K * np.exp(-r*T) * z3
    
    elif option_type == ["D", "O", "P", "B2"]:
        price = pricePTBarrier(S, K, tau, T, sigma, r, H, q, ["D", "O", "C", "B2"], position) \
        - S * np.exp(-q*T) * z7 + K * np.exp(-r*T) * z3 
    
    elif option_type == ["U", "O", "P", "B2"]:
        price = pricePTBarrier(S, K, tau, T, sigma, r, H, q, ["U", "O", "C", "B2"], position) \
        - S * np.exp(-q*T) * z8 + K * np.exp(-r*T) * z4

    else:
        raise ValueError(f"Invalid option type: {option_type}")

    if position == "short":
        price = - price

    return price