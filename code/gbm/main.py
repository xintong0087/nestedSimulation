import simTrue
import supervisedLearning


# Common parameters for underlying asset
M = 10 ** 5
N = 1
d = 20 
rho = 0.3
S_0 = 100
tau = 3/50
T = 1 
r = 0.05
mu = 0.08
levels=[0.8, 0.9, 0.95, 0.99, 0.996]
benchmark=0.9

# Parallelization parameters
n_jobs=4
verbose=10

# Portfolio to simulate
portfolio = 1

if portfolio == 1:
    K = [90, 100, 110]
    sigma = 0.1
    option_name=["European"]
    option_type=["C"]
    position=["long"]
elif portfolio == 2:
    K = [90, 100, 110]
    sigma = 0.3
    option_name=["Asian"]
    option_type=["C"]
    position=["long"]
elif portfolio == 3:
    K = [90, 100, 110]
    sigma = 0.2
    option_name=["Barrier"]

# res = simTrue.simulateTrueValues(M, d, rho, S_0, K, mu, sigma, r, tau, T,
#                                  levels=levels, benchmark=benchmark,
#                                  option_name=option_name, option_type=option_type,
#                                  position=position, n_jobs=n_jobs, verbose=verbose)

# print(res)


res_reg = supervisedLearning.regression(M, N, d, S_0, K, mu, sigma, rho, r, tau, T,
                                            option_name=option_name, option_type=option_type,
                                            position=position, test=False)