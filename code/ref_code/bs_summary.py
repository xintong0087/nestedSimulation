import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import rankdata


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

# Dimension 2: Simulation Budget (1000, 2000, ..., 10000, 20000, ..., 100000)

n_front_list = []
for i in range(2):
    n_front_list = n_front_list + list(np.arange(1, 10) * 10 ** (i + 3))
n_front_list = n_front_list + [10 ** (i + 4)]
label_2 = np.array(n_front_list)

# Dimension 3: Risk Measure
# 1. Indicator
# 2. Hockey Stick
# 3. Quadratic
# 4. VaR
# 5. CVaR

label_3 = ["Indicator",
           "Hockey-Stick",
           "Quadratic",
           "VaR",
           "CVaR"]

arr_112 = np.load("data/result_112.npy")
arr_113 = np.load("data/result_113.npy")
arr_114 = np.load("data/result_114.npy")
arr_115 = np.load("data/result_115.npy")
arr_116 = np.load("data/result_116.npy")
arr_11 = np.stack([arr_112, arr_113, arr_114, arr_115, arr_116])

arr_122 = np.load("data/result_122.npy")
arr_123 = np.load("data/result_123.npy")
arr_124 = np.load("data/result_124.npy")
arr_125 = np.load("data/result_125.npy")
arr_126 = np.load("data/result_126.npy")
arr_12 = np.stack([arr_122, arr_123, arr_124, arr_125, arr_126])

arr_132 = np.load("data/result_132.npy")
arr_133 = np.load("data/result_133.npy")
arr_134 = np.load("data/result_134.npy")
arr_135 = np.load("data/result_135.npy")
arr_136 = np.load("data/result_136.npy")
arr_13 = np.stack([arr_132, arr_133, arr_134, arr_135, arr_136])

arr_142 = np.load("data/result_142.npy")
arr_143 = np.load("data/result_143.npy")
arr_144 = np.load("data/result_144.npy")
arr_145 = np.load("data/result_145.npy")
arr_146 = np.load("data/result_146.npy")
arr_14 = np.stack([arr_142, arr_143, arr_144, arr_145, arr_146])

arr_1 = np.stack([arr_11, arr_12, arr_13, arr_14])

i_range = arr_1.shape[0]
j_range = arr_1.shape[1]
k_range = arr_1.shape[3]
pd.set_option('display.max_columns', None)

res_roc = np.zeros([i_range, j_range, k_range, 2])

for i in range(i_range):
    for j in range(j_range):
        plt.figure(figsize=(8, 4.5))
        arr = arr_1[i, j, :, :]
        slope, intercept = find_roc(label_2, arr)
        df = pd.DataFrame(arr,
                          index=label_2,
                          columns=label_3)
        legends = []
        for l in range(df.columns.shape[0]):
            legends = legends + [df.columns[l] + ": " + str(slope[l]) + ", " + str(intercept[l])]
        plt.loglog(df)
        plt.xlabel(r"Simulation Budget $\Gamma$")
        plt.ylabel("MSE")
        plt.legend(legends)
        plt.title(str(label_0[i]) + " - " + str(label_1[j]))
        plt.savefig("figures/gbm/" + str(label_0[i]) + "-" + str(label_1[j]) + ".png")
        plt.close()

        res_roc[i, j, :, :] = find_roc(label_2, arr, graph=False)

rank_roc = rankdata(res_roc, axis=2)

# Dimension 0: Option Type
# Dimension 1: Methods
# Dimension 3: Risk Measure

# ... for ... columns ... index ...
print("UNDER THE SAME OPTION TYPE: RANKING CONVERGENCE OF DIFFERENT RISK MEASURES FOR A GIVEN METHOD")
for i in range(i_range):

    print("Ranking Rate of Convergence of: ", label_0[i])
    df = pd.DataFrame(rank_roc[i, :, :, 0],
                      index=label_1,
                      columns=label_3).astype("int8")
    df.loc["Median Rank"] = np.median(rank_roc[i, :, :, 0], axis=0)
    print(df)
    print()
    df.to_csv("data/rank/" + label_0[i] + "_same_method_rankBy_riskMeasure.csv")

    df = pd.DataFrame(rank_roc[i, :, :, 1],
                      index=label_1,
                      columns=label_3).astype("int8")
    df.loc["Median Rank"] = np.median(rank_roc[i, :, :, 1], axis=0)
    df.to_csv("data/rank/" + label_0[i] + "_same_method_rankBy_riskMeasure_constant.csv")

print("UNDER THE SAME METHOD: RANKING CONVERGENCE OF DIFFERENT RISK MEASURES FOR A GIVEN OPTION TYPE")
for j in range(j_range):

    print("Ranking Rate of Convergence of: ", label_1[j])
    df = pd.DataFrame(rank_roc[:, j, :, 0],
                      index=label_0,
                      columns=label_3).astype("int8")
    df.loc["Median Rank"] = np.median(rank_roc[:, j, :, 0], axis=0)
    print(df)
    print()
    df.to_csv("data/rank/" + label_1[j] + "_same_optionType_rankBy_riskMeasure.csv")

    df = pd.DataFrame(rank_roc[:, j, :, 1],
                      index=label_0,
                      columns=label_3).astype("int8")
    df.loc["Median Rank"] = np.median(rank_roc[:, j, :, 1], axis=0)
    df.to_csv("data/rank/" + label_1[j] + "_same_optionType_rankBy_riskMeasure_constant.csv")

res_roc = np.zeros([i_range, j_range, k_range, 2])

for j in range(j_range):
    for k in range(k_range):
        plt.figure(figsize=(8, 4.5))
        arr = arr_1[:, j, :, k].T
        slope, intercept = find_roc(label_2, arr)
        df = pd.DataFrame(arr,
                          index=label_2,
                          columns=label_0)
        legends = []
        for l in range(df.columns.shape[0]):
            legends = legends + [df.columns[l] + ": " + str(slope[l]) + ", " + str(intercept[l])]
        plt.loglog(df)
        plt.xlabel(r"Simulation Budget $\Gamma$")
        plt.ylabel("MSE")
        plt.legend(legends)
        plt.title(str(label_1[j]) + " - " + str(label_3[k]))
        plt.savefig("figures/gbm/" + str(label_1[j]) + "-" + str(label_3[k]) + ".png")
        plt.close()

        res_roc[:, j, k, :] = find_roc(label_2, arr, graph=False)

rank_roc = rankdata(res_roc, axis=0)

# Dimension 0: Option Type
# Dimension 1: Methods
# Dimension 3: Risk Measure

# ... for ... columns ... index ...
print("UNDER THE SAME METHOD: RANKING CONVERGENCE OF DIFFERENT OPTION TYPES FOR A GIVEN RISK MEASURE")
for j in range(j_range):

    print("Ranking Rate of Convergence of: ", label_1[j])
    df = pd.DataFrame(rank_roc[:, j, :, 0].T,
                      index=label_3,
                      columns=label_0).astype("int8")
    df.loc["Median Rank"] = np.median(rank_roc[:, j, :, 0].T, axis=0)
    print(df)
    print()
    df.to_csv("data/rank/" + label_1[j] + "_same_riskMeasure_rankBy_optionType.csv")

    df = pd.DataFrame(rank_roc[:, j, :, 1].T,
                      index=label_3,
                      columns=label_0).astype("int8")
    df.loc["Median Rank"] = np.median(rank_roc[:, j, :, 1].T, axis=0)
    df.to_csv("data/rank/" + label_1[j] + "_same_riskMeasure_rankBy_optionType_constant.csv")

print("UNDER THE SAME RISK MEASURE: RANKING CONVERGENCE OF DIFFERENT OPTION TYPES FOR A GIVEN METHOD")
for k in range(k_range):

    print("Ranking Rate of Convergence of: ", label_3[k])
    df = pd.DataFrame(rank_roc[:, :, k, 0].T,
                      index=label_1,
                      columns=label_0).astype("int8")
    df.loc["Median Rank"] = np.median(rank_roc[:, :, k, 0].T, axis=0)
    print(df)
    print()
    df.to_csv("data/rank/" + label_3[k] + "_same_method_rankBy_optionType.csv")

    df = pd.DataFrame(rank_roc[:, :, k, 1].T,
                      index=label_1,
                      columns=label_0).astype("int8")
    df.loc["Median Rank"] = np.median(rank_roc[:, :, k, 1].T, axis=0)
    df.to_csv("data/rank/" + label_3[k] + "_same_method_rankBy_optionType_constant.csv")

res_roc = np.zeros([i_range, j_range, k_range, 2])

for i in range(i_range):
    for k in range(k_range):
        plt.figure(figsize=(8, 4.5))
        arr = arr_1[i, :, :, k].T
        slope, intercept = find_roc(label_2, arr)
        df = pd.DataFrame(arr,
                          index=label_2,
                          columns=label_1)
        legends = []
        for l in range(df.columns.shape[0]):
            legends = legends + [df.columns[l] + ": " + str(slope[l]) + ", " + str(intercept[l])]
        plt.loglog(df)
        plt.xlabel(r"Simulation Budget $\Gamma$")
        plt.ylabel("MSE")
        plt.legend(legends)
        plt.title(str(label_0[i]) + " - " + str(label_3[k]))
        plt.savefig("figures/gbm/" + str(label_0[i]) + "-" + str(label_3[k]) + ".png")
        plt.close()

        res_roc[i, :, k, :] = find_roc(label_2, arr, graph=False)

rank_roc = rankdata(res_roc, axis=1)

# Dimension 0: Option Type
# Dimension 1: Methods
# Dimension 3: Risk Measure

# ... for ... columns ... index ...
print("UNDER THE SAME OPTION TYPE: RANKING CONVERGENCE OF DIFFERENT METHODS FOR A GIVEN RISK MEASURE")
for i in range(i_range):

    print("Ranking Rate of Convergence of: ", label_0[i])
    df = pd.DataFrame(rank_roc[i, :, :, 0].T,
                      index=label_3,
                      columns=label_1).astype("int8")
    df.loc["Median Rank"] = np.median(rank_roc[i, :, :, 0].T, axis=0)
    print(df)
    print()
    df.to_csv("data/rank/" + label_0[i] + "_same_riskMeasure_rankBy_methods.csv")

    df = pd.DataFrame(rank_roc[i, :, :, 1].T,
                      index=label_3,
                      columns=label_1).astype("int8")
    df.loc["Median Rank"] = np.median(rank_roc[i, :, :, 1].T, axis=0)
    df.to_csv("data/rank/" + label_0[i] + "_same_riskMeasure_rankBy_methods_constant.csv")

print("UNDER THE SAME RISK MEASURE: RANKING CONVERGENCE OF DIFFERENT METHODS FOR A GIVEN OPTION TYPE")
for k in range(k_range):

    print("Ranking Rate of Convergence of: ", label_3[k])
    df = pd.DataFrame(rank_roc[:, :, k, 0],
                      index=label_0,
                      columns=label_1).astype("int8")
    df.loc["Median Rank"] = np.median(rank_roc[:, :, k, 0], axis=0)
    print(df)
    print()
    df.to_csv("data/rank/" + label_3[k] + "_same_optionType_rankBy_methods.csv")

    df = pd.DataFrame(rank_roc[:, :, k, 1],
                      index=label_0,
                      columns=label_1).astype("int8")
    df.loc["Median Rank"] = np.median(rank_roc[:, :, k, 1], axis=0)
    df.to_csv("data/rank/" + label_3[k] + "_same_optionType_rankBy_methods_constant.csv")
