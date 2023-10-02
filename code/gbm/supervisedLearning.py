import numpy as np

# Import all supervised learning models
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.kernels import Matern

# Import cross validation methods
from sklearn.model_selection import KFold
from skopt import BayesSearchCV
from skopt.space import Real

import sns


def fitSL(outerScenarios, payoff, model_name, hyperparameters=None):

    """
    Fit a supervised learning model to the payoff of the option.

    :param outerScenarios: outer scenarios
    :param payoff: payoff of the option
    :param model_name: name of the model
    :param hyperparameters: hyperparameters of the model

    :return: fitted model
    """
        
    y = np.sum(payoff, axis=0)

    if model_name == "Regression":
        X = generate_basis(outerScenarios)
        model = LinearRegression().fit(X, y)

    elif model_name == "KNN":
        k_opt = hyperparameters
        model = KNeighborsRegressor(n_neighbors=k_opt, weights="uniform").fit(X, y)

    elif model_name == "KernelRidge":
        alpha_opt = hyperparameters[0]
        l_opt = hyperparameters[1]
        nu_opt = hyperparameters[2]
        model = KernelRidge(alpha=alpha_opt, 
                            kernel=Matern(length_scale=l_opt, nu=nu_opt)).fit(X, y)
    else:
        raise ValueError("Model name not recognized.")
    
    return model


def predictSL(outerScenarios, model_name, model):

    """
    Generate the prediction of the supervised learning model.

    :param outerScenarios: outer scenarios
    :param model_name: name of the model
    :param model: fitted model

    :return: prediction of the model
    """

    if model_name == "Regression":
        X = generate_basis(outerScenarios)
        y_pred = model.predict(X)
    elif model_name == "KNN":
        y_pred = model.predict(X)
    elif model_name == "KernelRidge":
        y_pred = model.predict(X)
    else:
        raise ValueError("Model name not recognized.")

    return y_pred


def crossValSL(outerScenarios, payoff, model_name, n_splits=5):

    """
    Perform cross validation to find the optimal hyperparameters.

    :param outerScenarios: outer scenarios
    :param payoff: payoff of the option
    :param model_name: name of the model
    :param n_splits: number of splits for cross validation

    :return: optimal hyperparameters
    """
    
    X = generate_basis(outerScenarios)
    y = np.sum(payoff, axis=0)
    
    cv = KFold(n_splits=n_splits)
    
    if model_name == "KNN":
        k_range = np.arange(100, 301, 50)
        n_k = k_range.shape[0]
        cv_score = np.zeros(n_k)    

        for k in range(n_k):
            for train_ind, val_ind in cv.split(X, y):
                X_train = X[train_ind]
                X_val = X[val_ind]
                y_train = y[train_ind]
                y_val = y[val_ind]

                y_hat = KNeighborsRegressor(n_neighbors=k_range[k]).fit(X_train, y_train).predict(X_val)
                cv_score[k] = cv_score[k] + np.sum((y_hat - y_val) ** 2)

        k_opt = k_range[np.argmin(cv_score)]

        res = k_opt

    elif model_name == "KernelRidge":

        param_distributions = {"alpha": Real(1e-5, 1e-1, "log-uniform"),
                               "kernel__length_scale": Real(1e-3, 1e3, "log-uniform"),
                               "kernel__nu": Real(5e-1, 5e0, "log-uniform")}
        
        bayesian_search = BayesSearchCV(estimator=KernelRidge(kernel=Matern()),
                                        search_spaces=param_distributions, n_jobs=20, cv=cv)
        
        bayesian_search.fit(X, y)

        alpha = bayesian_search.best_params_["alpha"]
        l = bayesian_search.best_params_["kernel__length_scale"]
        nu = bayesian_search.best_params_["kernel__nu"]

        res = [alpha, l, nu]

    else:
        raise ValueError("Model name not recognized.")

    return res


def Laguerre_polynomial(X, degree=3):

    X_norm = (X - X.mean(axis=1).reshape(-1, 1)) / X.std(axis=1).reshape(-1, 1)

    L_0 = np.exp(-X_norm / 2)
    L_1 = np.exp(-X_norm / 2) * (1 - X_norm)
    L_2 = np.exp(-X_norm / 2) * (1 - 2 * X_norm + (1 / 2) * (X_norm ** 2))
    L_3 = np.exp(-X_norm / 2) * (1 - 3 * X_norm + (3 / 2) * (X_norm ** 2) - (1 / 6) * (X_norm ** 3))

    L = [L_1, L_2, L_3]

    X_train = L_0
    for k in range(degree):
        X_train = np.concatenate([X_train, L[k]], axis=0)

    return X_train


def generate_basis(sample_outer, option_type="Vanilla",
                   basis_function="Laguerre",
                   sample_max=1, sample_min=0):

    if option_type == "Asian":

        S_tau = sample_outer[:, :, -1]

        geometric_sum_outer = np.prod(sample_outer[:, :, 1:], axis=2)

        X = np.concatenate([S_tau, geometric_sum_outer], axis=0)

        if basis_function == "Laguerre":
            X_train = Laguerre_polynomial(X)
        else:
            X_train = X

        X_train = X_train.T

    elif option_type == "Barrier_U":

        S_tau = sample_outer[:, :, -1]

        X_U = np.concatenate([S_tau, sample_max], axis=0)

        if basis_function == "Laguerre":
            X_train_U = Laguerre_polynomial(X_U)
        else:
            X_train_U = Laguerre_polynomial(X_U)

        X_train = X_train_U.T

    elif option_type == "Barrier_D":

        S_tau = sample_outer[:, :, -1]

        X_D = np.concatenate([S_tau, sample_min], axis=0)

        if basis_function == "Laguerre":
            X_train_D = Laguerre_polynomial(X_D)
        else:
            X_train_D = Laguerre_polynomial(X_D)

        X_train = X_train_D.T

    else:

        X = sample_outer

        if basis_function == "Laguerre":
            X_train = Laguerre_polynomial(X)
        else:
            X_train = X

        X_train = X_train.T

    return X_train


def regression(M, N, d, S_0, K, mu, sigma, r, tau, T, option_name, option_type, position, test=False):

    """
    Asian and Barrier not supported yet. Using placeholders for now.
    """

    outerScenarios, loss = sns.nestedSimulation(M, N, d, S_0, K, mu, sigma, r, tau, T, option_name, option_type, position)

    X_train = generate_basis(outerScenarios, option_type=option_type)
    y_train = loss

    reg = LinearRegression().fit(X_train, y_train)

    if test:
        outerScenarios = sns.simOuter(M, d, S_0, mu, sigma, tau)
        X_test = generate_basis(outerScenarios)
    else:
        X_test = X_train

    y_test = reg.predict(X_test)

    return y_test


def kNN(M, N, d, S_0, K, mu, sigma, r, tau, T, option_name, option_type, position, test=False, cv=False, k_opt=100):

    """
    Asian and Barrier not supported yet. Using placeholders for now.
    """

    outerScenarios, loss = sns.nestedSimulation(M, N, d, S_0, K, mu, sigma, r, tau, T, option_name, option_type, position)

    X_train = generate_basis(outerScenarios, option_type=option_type)
    y_train = loss

    if cv:
        k_opt = crossValidation(X_train, y_train, "kNN")
        
    kNN = KNeighborsRegressor(n_neighbors=k_opt).fit(X_train, y_train)

    if test:
        outerScenarios = sns.simOuter(M, d, S_0, mu, sigma, tau)
        X_test = generate_basis(outerScenarios)
    else:
        X_test = X_train

    y_test = kNN.predict(X_test)

    return y_test


def kernelRidge(M, N, d, S_0, K, mu, sigma, r, tau, T, option_name, option_type, position, test=False, cv=False,
                alpha_opt=0.01, l_opt=1, nu_opt=0.5):

    """
    Asian and Barrier not supported yet. Using placeholders for now.
    """

    outerScenarios, loss = sns.nestedSimulation(M, N, d, S_0, K, mu, sigma, r, tau, T, option_name, option_type, position)

    X_train = generate_basis(outerScenarios, option_type=option_type)
    y_train = loss

    if cv:
        alpha_opt, l_opt, nu_opt = crossValidation(X_train, y_train, "kernelRidge")

    kernelRidge = KernelRidge(alpha=alpha_opt, kernel=Matern(length_scale=l_opt, nu=nu_opt)).fit(X_train, y_train)

    if test:
        outerScenarios = sns.simOuter(M, d, S_0, mu, sigma, tau)
        X_test = generate_basis(outerScenarios)
    else:
        X_test = X_train

    y_test = kernelRidge.predict(X_test)

    return y_test


def crossValidation(X, y, model_name, n_splits=5):
    
    cv = KFold(n_splits=n_splits)

    if model_name == "kNN":
            
        k_range = np.arange(100, 301, 50)
        n_k = k_range.shape[0]
        cv_score = np.zeros(n_k)    

        for k in range(n_k):
            for train_ind, val_ind in cv.split(X, y):
                X_train = X[train_ind]
                X_val = X[val_ind]
                y_train = y[train_ind]
                y_val = y[val_ind]

                y_hat = KNeighborsRegressor(n_neighbors=k_range[k]).fit(X_train, y_train).predict(X_val)
                cv_score[k] = cv_score[k] + np.sum((y_hat - y_val) ** 2)

        k_opt = k_range[np.argmin(cv_score)]

        res = k_opt
    
    elif model_name == "kernelRidge":

        param_distributions = {"alpha": Real(1e-5, 1e-1, "log-uniform"),
                               "kernel__length_scale": Real(1e-3, 1e3, "log-uniform"),
                               "kernel__nu": Real(5e-1, 5e0, "log-uniform")}
        
        bayesian_search = BayesSearchCV(estimator=KernelRidge(kernel=Matern()),
                                        search_spaces=param_distributions, n_jobs=20, cv=cv)
        
        bayesian_search.fit(X, y)

        alpha = bayesian_search.best_params_["alpha"]
        l = bayesian_search.best_params_["kernel__length_scale"]
        nu = bayesian_search.best_params_["kernel__nu"]

        res = [alpha, l, nu]
    
    else:

        raise ValueError("Model name not recognized.")
    
    return res