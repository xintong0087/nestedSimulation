import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

from sklearn.linear_model import LinearRegression
from scipy.stats import rankdata


def approxCR(df):

    Gammas = np.array(df.index).astype(float)

    slopes = []
    intercepts = []

    for col in df.columns:
        
        col_values = df[col].dropna().values

        length = len(col_values)

        x = np.log(Gammas).reshape(-1, 1)[:length]
        y = np.log(col_values)

        reg = LinearRegression().fit(x, y)

        slopes.append(np.round(reg.coef_[0], 2))
        intercepts.append(np.round(reg.intercept_, 2))
    
    return slopes, intercepts


def plotDF(df, mainCategory, text="", saveFolder="./figures"):

    plt.figure(figsize=(10, 6))
    plt.loglog(df, marker="o", linestyle="dashed")
    plt.xlabel(r"Simulation Budget $\Gamma$")
    plt.ylabel("MSE")

    slopes, intercepts = approxCR(df)

    legend = [f"{col} ({s}, {i})" for col, s, i in zip(df.columns, slopes, intercepts)]

    plt.legend(legend)
    plt.title(f"Comparison of different {mainCategory} {text}")

    if not os.path.exists(saveFolder):
        os.mkdir(saveFolder)

    plt.savefig(f"{saveFolder}/{mainCategory}.png")
    plt.close()

    print(f"Figure saved to {saveFolder}/{mainCategory}.png")

    return None


saveFolder = "./result"
saveFolderHeston = "./resultHeston"

procedures = ["SNS", "Regression", "Kernel", "KRR"]

optionTypes = ["European", "Asian", "BarrierUp", "BarrierDown"]

dimensions = [1, 2, 5, 10, 20]

riskMeasures = ["Indicator", "Hockey-stick", "Quadratic", "VaR", "CVaR"]
levels = ["80%", "90%", "95%", "99%", "99.6%"]
allRiskMeasures = ["Indicator", "Hockey-stick", "Quadratic", 
                   "80%-VaR", "80%-CVaR", "90%-VaR", "90%-CVaR", 
                   "95%-VaR", "95%-CVaR", "99%-VaR", "99%-CVaR", "99.6%-VaR", "99.6%-CVaR"]

possibleInputs = ["procedures", "option types", "dimensions", "risk measures", "levels of VaR and CVaR", "asset models"]

mainCategory = input(f"Please enter the main category you want to compare with: {possibleInputs} > ")

if mainCategory == "procedures":

    o = input(f"Please enter the option type you want to compare from: {optionTypes} > ")
    d = int(input(f"Please enter the dimension you want to compare from: {dimensions} > "))
    r = input(f"Please enter the risk measure you want to compare from: {riskMeasures} > ")

    if (r == "VaR") or (r == "CVaR"): 
        l = input(f"Please enter the level of VaR or CVaR you want to compare from: {levels} > ")
        rmName = f"{l}-{r}"
    else:
        rmName = r

    df_all = pd.DataFrame()
    for p in procedures:
        try:
            df = pd.read_csv(f"{saveFolder}/{p}_{o}_{d}.csv", index_col=0)
            df.columns = 2**np.array(range(len(df.columns))) * 1000
            df.index = allRiskMeasures

            temp = df.loc[rmName, :]
            temp.name = p

            df_all = pd.concat([df_all, temp], axis=1)
        except:
            print(f"{p}_{o}_{d}.csv does not exist.")
    
    plotDF(df_all**2, mainCategory, text=f"({o}, {r}, d = {d})")

elif mainCategory == "option types":

    p = input(f"Please enter the procedure you want to compare from: {procedures} > ")
    d = int(input(f"Please enter the dimension you want to compare from: {dimensions} > "))
    r = input(f"Please enter the risk measure you want to compare from: {riskMeasures} > ")

    if (r == "VaR") or (r == "CVaR"):
        l = input(f"Please enter the level of VaR or CVaR you want to compare from: {levels} > ")
        rmName = f"{l}-{r}"
    else:
        rmName = r
    
    df_all = pd.DataFrame()
    for o in optionTypes:
        try:
            df = pd.read_csv(f"{saveFolder}/{p}_{o}_{d}.csv", index_col=0)
            df.columns = 2**np.array(range(len(df.columns))) * 1000
            df.index = allRiskMeasures

            temp = df.loc[rmName, :]
            temp.name = o

            df_all = pd.concat([df_all, temp], axis=1)
        except:
            print(f"{p}_{o}_{d}.csv does not exist.")

    plotDF(df_all**2, mainCategory, text=f"({p}, {r}, d = {d})")

elif mainCategory == "dimensions":

    p = input(f"Please enter the procedure you want to compare from: {procedures} > ")
    o = input(f"Please enter the option type you want to compare from: {optionTypes} > ")
    r = input(f"Please enter the risk measure you want to compare from: {riskMeasures} > ")

    if (r == "VaR") or (r == "CVaR"):
        l = float(input(f"Please enter the level of VaR or CVaR you want to compare from: {levels} > "))
        rmName = f"{l}-{r}"
    else:
        rmName = r

    df_all = pd.DataFrame()
    for d in dimensions:
        try:
            df = pd.read_csv(f"{saveFolder}/{p}_{o}_{d}.csv", index_col=0)
            df.columns = 2**np.array(range(len(df.columns))) * 1000
            df.index = allRiskMeasures

            temp = df.loc[rmName, :]
            temp.name = d

            df_all = pd.concat([df_all, temp], axis=1)
        except:
            print(f"{p}_{o}_{d}.csv does not exist.")
    
    plotDF(df_all**2, mainCategory, text=f"({p}, {o}, {r})")

elif mainCategory == "risk measures":

    p = input(f"Please enter the procedure you want to compare from: {procedures} > ")
    o = input(f"Please enter the option type you want to compare from: {optionTypes} > ")
    d = int(input(f"Please enter the dimension you want to compare from: {dimensions} > "))
    l = input(f"Please enter the level of VaR or CVaR you want to compare from: {levels} > ")

    df = pd.read_csv(f"{saveFolder}/{p}_{o}_{d}.csv", index_col=0)
    df.columns = 2**np.array(range(len(df.columns))) * 1000
    df.index = allRiskMeasures

    df_all = df.loc[riskMeasures[:3] + [f"{l}-{riskMeasures[3]}", f"{l}-{riskMeasures[4]}"], :]
    df_all = df_all.T    
    
    plotDF(df_all**2, mainCategory, text=f"({p}, {o}, d = {d})")


elif mainCategory == "levels of VaR and CVaR":

    p = input(f"Please enter the procedure you want to compare from: {procedures} > ")
    o = input(f"Please enter the option type you want to compare from: {optionTypes} > ")
    d = int(input(f"Please enter the dimension you want to compare from: {dimensions} > "))

    df = pd.read_csv(f"{saveFolder}/{p}_{o}_{d}.csv", index_col=0)
    df.columns = 2**np.array(range(len(df.columns))) * 1000
    df.index = allRiskMeasures

    df_VaR = df.iloc[2::2].T
    df_CVaR = df.iloc[3::2].T

    plotDF(df_VaR**2, mainCategory, text=f"({p}, {o}, d = {d})")
    plotDF(df_CVaR**2, mainCategory, text=f"({p}, {o}, d = {d})")

elif mainCategory == "asset models":

    o = "European"
    d = 1

    p = input(f"Please enter the procedure you want to compare from: {procedures} > ")
    a = input(f"Compare to BS or within Heston? (BS/Heston) > ")

    if a == "Heston":

        try:
            df_Heston = pd.read_csv(f"{saveFolderHeston}/{p}_Heston.csv", index_col=0)
            df_Heston.columns = 2**np.array(range(len(df_Heston.columns))) * 1000
            df_Heston.index = allRiskMeasures

            df_Heston = df_Heston.T

            plotDF(df_Heston**2, mainCategory, text=f"({p}, {o}, d = {d})")

        except:

            pass

    elif a == "BS":

        r = input(f"Please enter the risk measure you want to compare from: {riskMeasures} > ")

        if (r == "VaR") or (r == "CVaR"):
            l = float(input(f"Please enter the level of VaR or CVaR you want to compare from: {levels} > "))
            rmName = f"{l}-{r}"
        else:
            rmName = r

        try:
            df_BS = pd.read_csv(f"{saveFolder}/{p}_{o}_{d}.csv", index_col=0)
            df_BS.columns = 2**np.array(range(len(df_BS.columns))) * 1000
            df_BS.index = allRiskMeasures

            df_Heston = pd.read_csv(f"{saveFolderHeston}/{p}_Heston.csv", index_col=0)
            df_Heston.columns = 2**np.array(range(len(df_Heston.columns))) * 1000
            df_Heston.index = allRiskMeasures

            df_all = pd.concat([df_BS.loc[rmName, :], df_Heston.loc[rmName, :]], axis=1)

            df_all.columns = ["BS", "Heston"]

            plotDF(df_all**2, mainCategory, text=f"({p}, {o}, {r}, d = {d})")
        
        except:
            pass

