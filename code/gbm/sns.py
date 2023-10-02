import numpy as np
from scipy.stats.distributions import norm


def simOuter(M, d, S0, drift, diffusion, tau, step_size=1/253, path=False):

    """
    Simulate outer scenarios of the underlying asset.

    :param M: number of scenarios
    :param d: dimension of the model
    :param S0: initial value of the asset
    :param drift: drift of the asset
    :param diffusion: diffusion matrix of the asset
    :param tau: time to maturity
    :param step_size: step size of the discretization
    :param path: whether to return the whole path or not

    :return: outer scenarios
        if path:
            dimension 0: d, number of assets
            dimension 1: M, number of scenarios
            dimension 2: n_step + 1, number of time steps + 1
        else:
            dimension 0: d, number of assets
            dimension 1: M, number of scenarios
    """

    drift = np.full(d, drift)
    diffusion = np.array(diffusion)
    A = np.linalg.cholesky(diffusion)

    if path:
        n_step = int(tau // step_size) + 1
        outerScenarios = np.zeros([d, M, n_step + 1])
        outerScenarios[:, :, 0] = np.array(S0)

        drift = (drift - 0.5 * np.diagonal(diffusion)) * step_size
        drift = drift.reshape(-1, 1)

        for k in range(1, n_step + 1):
            Z = np.random.normal(0, 1, [d, M])
            diffusion = np.sqrt(step_size) * np.matmul(A, Z)
            outerScenarios[:, :, k] = outerScenarios[:, :, k - 1] * np.exp(drift + diffusion)

    else:
        outerScenarios = np.zeros([d, M])

        drift = (drift - 0.5 * np.diagonal(diffusion)) * tau

        for i in range(M):
            Z = np.random.normal(0., 1., d)
            diffusion = np.matmul(A, Z) * np.sqrt(tau)
            outerScenarios[:, i] = S0 * np.exp(drift + diffusion)


    return outerScenarios


def simInner(M, N, d, outerScenarios, drift, diffusion, T, step_size=1/253, path=False):
    
    """
    Simulate inner paths of the underlying asset given a set of outer scenarios.

    :param M: number of scenarios
    :param N: number of inner paths per scenario
    :param d: dimension of the model
    :param outerScenarios: outer scenarios
    :param drift: drift of the asset
    :param diffusion: diffusion matrix of the asset
    :param T: time to maturity
    :param step_size: step size of the discretization
    :param path: whether to return the whole path or not

    :return: inner paths
        if path:
            dimension 0: d, number of assets
            dimension 1: M, number of scenarios
            dimension 2: N, number of inner paths per scenario
            dimension 3: n_step + 1, number of time steps + 1
        else:
            dimension 0: d, number of assets
            dimension 1: M, number of scenarios
            dimension 2: N, number of inner paths per scenario
    """

    if outerScenarios.shape[1] != M:
        raise ValueError("Number of scenarios in outer scenarios does not match M.")
    
    drift_vec = np.full(d, drift)
    diffusion_mat = np.array(diffusion)
    A = np.linalg.cholesky(diffusion_mat)

    if path:
        n_step = int(T // step_size) + 1
        innerPaths = np.zeros([d, M, N, n_step + 1])
        for j in range(N):
            innerPaths[:, :, j, 0] = outerScenarios[:, :, -1]

        for i in range(M):
            for k in range(1, n_step + 1):
                Z = np.random.normal(0, 1, [d, N])
                drift = (drift_vec - 0.5 * np.diagonal(diffusion_mat)) * step_size
                drift = drift.reshape(-1, 1)
                diffusion = np.sqrt(step_size) * np.matmul(A, Z)
                innerPaths[:, i, :, k] = innerPaths[:, i, :, k - 1] * np.exp(drift + diffusion)

    else:
        innerPaths = np.zeros([d, M, N])

        for i in range(M):
            for j in range(N):
                Z = np.random.normal(0., 1., d)
                drift = (drift_vec - 0.5 * np.diagonal(diffusion_mat)) * T
                diffusion = np.matmul(A, Z) * np.sqrt(T)
                innerPaths[:, i, j] = outerScenarios[:, i] * np.exp(drift + diffusion)

    return innerPaths


