import method_Broadie

import pandas as pd
import time

seed = 22

K = [101, 110, 114.5]
H = [91, 100, 104.5]

S_0 = 100
mu = 0.08
sigma = 0.2
r = 0.03
T = 1/12
tau = 1/52

alpha = 0.05


M = int(2e7)

start = time.time()
indicator, hockey, quadratic, VaR, CVaR, loss, S_tau = method_Broadie.simTrue(M, S_0, K, H, mu, sigma, r, tau, T, alpha, 22)

df = pd.DataFrame([indicator, hockey, quadratic, VaR, CVaR],
                  index=["Indicator",
                         "Hockey",
                         "Quadratic",
                         "VaR",
                         "CVaR"],
                  columns=[M]).T
print(df)
df.to_csv("./trueValue_Broadie.csv")

print("Total time taken:", time.time() - start, "seconds.")

