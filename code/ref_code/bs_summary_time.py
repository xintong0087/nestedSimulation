import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Underlying Processes: Black Scholes

# Dimension 0: Option Type
# 1. European Call Option
# 2. Geometric Asian Option
# 3. Barrier Option (Up and Out)
# 4. Barrier Option (Down and Out)

label_0 = ["European Call",
           "Geometric Asian",
           "Up and Out Barrier",
           "Down and Out Barrier"]

# Dimension 1: Methods
# 1. Standard Nested
# 2. Standard Nested with Bootstrap
# 3. Regression
# 4. Kernel
# 5. Likelihood Ratio
# 6. Kernel Ridge Regression

label_1 = ["Standard Nested with Bootstrap",
           "Regression",
           "Kernel",
           "Likelihood Ratio",
           "Kernel Ridge Regression"]

# Dimension 2: Simulation Budget (1000, 2000, ..., 10000)

n_rep = 150

n_front_list = []
for _i in range(1):
    n_front_list = n_front_list + list(np.arange(1, 10) * 10 ** (_i + 3))
n_front_list = n_front_list + [10 ** (_i + 4)]
n_front_vec = np.array(n_front_list)
label_2 = np.array(n_front_list)

# Dimension 3: Risk Measure
# 1. Indicator
# 2. Hockey Stick
# 3. Quadratic
# 4. VaR
# 5. CVaR

label_3 = ["Simulation Cost (for Training)",
           "Simulation Cost (for Prediction)",
           "Modeling Fitting Cost",
           "Prediction Cost",
           "Cross Validation Cost"]

arr_112 = np.load("data/result_112_time.npy")
arr_113 = np.load("data/result_113_time.npy")
arr_114 = np.load("data/result_114_time.npy")
arr_115 = np.load("data/result_115_time.npy")
arr_116 = np.load("data/result_116_time.npy")
arr_11 = np.stack([arr_112, arr_113, arr_114, arr_115, arr_116])

arr_122 = np.load("data/result_122_time.npy")
arr_123 = np.load("data/result_123_time.npy")
arr_124 = np.load("data/result_124_time.npy")
arr_125 = np.load("data/result_125_time.npy")
arr_126 = np.load("data/result_126_time.npy")
arr_12 = np.stack([arr_122, arr_123, arr_124, arr_125, arr_126])

arr_132 = np.load("data/result_132_time.npy")
arr_133 = np.load("data/result_133_time.npy")
arr_134 = np.load("data/result_134_time.npy")
arr_135 = np.load("data/result_135_time.npy")
arr_136 = np.load("data/result_136_time.npy")
arr_13 = np.stack([arr_132, arr_133, arr_134, arr_135, arr_136])

arr_142 = np.load("data/result_142_time.npy")
arr_143 = np.load("data/result_143_time.npy")
arr_144 = np.load("data/result_144_time.npy")
arr_145 = np.load("data/result_145_time.npy")
arr_146 = np.load("data/result_146_time.npy")
arr_14 = np.stack([arr_142, arr_143, arr_144, arr_145, arr_146])

arr_1 = np.stack([arr_11, arr_12, arr_13, arr_14])
arr_1[:, :, :, 4] = arr_1[:, :, :, 4] / n_rep

i_range = arr_1.shape[0]
j_range = arr_1.shape[1]
k_range = arr_1.shape[3]

for i in range(i_range):
    for j in range(j_range):
        plt.figure(figsize=(8, 4.5))
        df = pd.DataFrame(arr_1[i, j, :, :],
                          index=label_2,
                          columns=label_3)
        plt.plot(df)
        plt.xlabel(r"Simulation Budget $\Gamma$")
        plt.ylabel("Time (s)")
        plt.legend(df.columns)
        plt.title(str(label_0[i]) + " - " + str(label_1[j]))
        plt.savefig("figures/gbm_time/" + str(label_0[i]) + "-" + str(label_1[j]) + ".png")
        plt.close()

for j in range(j_range):
    for k in range(k_range):
        plt.figure(figsize=(8, 4.5))
        df = pd.DataFrame(arr_1[:, j, :, k].T,
                          index=label_2,
                          columns=label_0)
        plt.plot(df)
        plt.xlabel(r"Simulation Budget $\Gamma$")
        plt.ylabel("Time (s)")
        plt.legend(df.columns)
        plt.title(str(label_1[j]) + " - " + str(label_3[k]))
        plt.savefig("figures/gbm_time/" + str(label_1[j]) + "-" + str(label_3[k]) + ".png")
        plt.close()

for i in range(i_range):
    for k in range(k_range):
        plt.figure(figsize=(8, 4.5))
        df = pd.DataFrame(arr_1[i, :, :, k].T,
                          index=label_2,
                          columns=label_1)
        plt.plot(df)
        plt.xlabel(r"Simulation Budget $\Gamma$")
        plt.ylabel("Time (s)")
        plt.legend(df.columns)
        plt.title(str(label_0[i]) + " - " + str(label_3[k]))
        plt.savefig("figures/gbm_time/" + str(label_0[i]) + "-" + str(label_3[k]) + ".png")
        plt.close()

for j in range(j_range):
    plt.figure(figsize=(8, 4.5))
    df = pd.DataFrame(np.sum(arr_1[:, j, :, :], axis=2).T,
                      index=label_2,
                      columns=label_0)
    plt.plot(df)
    plt.xlabel(r"Simulation Budget $\Gamma$")
    plt.ylabel("Time (s)")
    plt.legend(df.columns)
    plt.title(str(label_1[j]) + " - total time")
    plt.savefig("figures/gbm_time/" + str(label_1[j]) + " - total_time.png")
    plt.close()

for i in range(i_range):
    plt.figure(figsize=(8, 4.5))
    df = pd.DataFrame(np.sum(arr_1[i, :, :, :], axis=2).T,
                      index=label_2,
                      columns=label_1)
    plt.plot(df)
    plt.xlabel(r"Simulation Budget $\Gamma$")
    plt.ylabel("Time (s)")
    plt.legend(df.columns)
    plt.title(str(label_0[i]) + " - total time")
    plt.savefig("figures/gbm_time/" + str(label_0[i]) + " - total_time.png")
    plt.close()