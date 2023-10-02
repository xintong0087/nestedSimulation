import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def find_roc(n_vec, data, graph=True):

    roc = np.zeros(data.shape[1])
    constants = np.zeros(data.shape[1])

    for _ in range(data.shape[1]):
        reg = LinearRegression().fit(np.log(n_vec.reshape(-1, 1)), np.log(data[:, _]))

        roc[_] = reg.coef_[0]
        constants[_] = reg.intercept_

    if graph:
        return np.around(roc, 2), np.around(constants, 2)
    else:
        return np.stack([roc, constants]).T


arr = np.load("data/result_kernel.npy")

arr_20 = np.stack([np.load("data/result_114.npy"),
                   np.load("data/result_124.npy"),
                   np.load("data/result_134.npy"),
                   np.load("data/result_144.npy")]).transpose(0, 2, 1)

arr = np.concatenate([arr, np.expand_dims(arr_20, axis=0)], axis=0)

label_0 = ["d = 1",
           "d = 5",
           "d = 10",
           "d = 20"]

label_1 = ["European Call",
           "Geometric Asian",
           "Up and Out Barrier",
           "Down and Out Barrier"]

label_2 = ["Indicator",
           "Hockey-Stick",
           "Quadratic",
           "VaR",
           "CVaR"]

n_front_list = []
for i in range(2):
    n_front_list = n_front_list + list(np.arange(1, 10) * 10 ** (i + 3))
n_front_list = n_front_list + [10 ** (i + 4)]
label_3 = np.array(n_front_list)

for i in range(arr.shape[1]):

    for j in range(arr.shape[2]):
        plt.figure(figsize=(8, 4.5))
        data = arr[:, i, j, :].T
        slope, intercept = find_roc(label_3, data)

        df = pd.DataFrame(data,
                          index=label_3,
                          columns=label_0)

        legends = []
        for l in range(df.columns.shape[0]):
            legends = legends + [df.columns[l] + ": " + str(slope[l]) + ", " + str(intercept[l])]

        plt.loglog(df)
        plt.xlabel(r"Simulation Budget $\Gamma$")
        plt.ylabel("MSE")
        plt.legend(legends)
        plt.title("Kernel - " + str(label_1[i]) + " - " + str(label_2[j]))
        plt.savefig("figures/gbm/kernel/" + str(label_1[i]) + "-" + str(label_2[j]) + ".png")
        plt.close()
