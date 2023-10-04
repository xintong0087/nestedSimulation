# Implementation of Nested Simulation Procedure for Estimating Risk Measures

## Module 1 - Calculate True Values of Risk Measures

1. Calculate the portfolio value at time 0 using analytical formula.
2. Simulate outer scenarios extensively.
3. For each scenario, calculate the option payoffs using analytical formula.
4. Average the scenario-wise portfolio losses over the outer scenarios.
5. Save to file.

## Module 2 - Standard Nested Simulation Procedure

1. Calculate the portfolio value at time 0 using analytical formula.
2. Simulate M outer scenarios.
3. For each outer scenario, simulates N inner replications.
4. Calculate the option payoffs for all inner replications.
5. For each outer scenario, average the portfolio losses over its inner replications.
6. Calculate risk measures from the M portfolio losses in step 3.

## Module 3 - Nested Simulation Procedure with Supervised Learning Proxies

1. Do 1-5 in Module 2.
2. Use the average scenario-wise portfolio loss and the outer scenarios to train a supervised learning proxy.
    * Regression
        * Apply basis functions to the outer scenarios. 
    * Kernel Smoothing (kNN)
        * Perform cross-Validation on k (grid search).
    * KRR
        * Perform cross-Validation on alpha, l, nu (Bayesian search).
3. Use the proxy predictions as the portfolio losses
    * A new set of M outer scenarios can be simulated based on user specification 
4. Calculate risk measures from the M portfolio losses in step 3

## Module 4 - Nested Simulation Procedure with Likelihood Ratio 

1. Do 1-5 in Module 2
2. For each scenario and inner replication, calculate the likelihood ratio.
    * Calculate the conditional density of the inner replication given the outer scenario.
    * Calculate the marginal density of the inner replication. 
    * The likelihood ratio is the ratio of the conditional density to the marginal density (some terms cancel out).
3. For each scenario, average the likelihood-ratio weighted portfolio losses over all inner replications.
4. Calculate risk measures from the M portfolio losses in step 3.

## Module 5 - Run Macro and Calculate MSE 

1. Load the true values for the risk measures from save location of Module 1.
2. Depending on the choice of nested simulation procedure, do Module 2, 3, or 4 for 1000 macro replications.
    * Record the squared errors for all risk measure estimates.
3. Calculate the MSE as the average of the squared errors.
 

## Scripts and Functions

* `main.py` - main script to run the nested simulation procedure
* `optionPricing.py` - functions for calculating option values
    * `priceVanilla`: price vanilla options given the current asset price
    * `priceDiscreteGeoAsian_0`: price geometric Asian options at time 0
    * `priceDiscreteGeoAsian_tau`: price geometric Asian options given an outer path to time tau
    * `priceBarrier_0`: price barrier options at time 0
    * `priceBarrier_tau`: price barrier options given an outer path to time tau
    * `pricePTBarrier`: price partial-time barrier options
* `simTrueValues.py` - functions for calculating true values of risk measures
    * `simulateTrueValues`: simulate true values of risk measures
* `sns.py` - functions for the standard nested simulation procedure
    * `simOuter`: simulate outer scenarios
    * `simInner`: simulate inner replications given outer scenarios
    * `nestedSimulation`: perform the standard nested simulation procedure
* `supervisedLearning.py` - functions for supervised learning
    * `regression`: perform nested simulation procedure with regression
    * `kNN`: perform nested simulation procedure with kernel smoothing
    * `kernelRidge`: perform nested simulation procedure with kernel ridge regression
    * `crossValidation`: perform cross-validation with grid search or Bayesian search
    * `generateBasis`: generate basis functions for regression
* `likelihoodRatio.py` - functions for likelihood ratio
    * `computeLikelihoodRatio`: compute likelihood ratio
    * `nestedSimulationLR`: perform nested simulation procedure with likelihood ratio
* `helpers.py` - helper functions
    * `generate_cor_mat`: generate correlation matrix for the underlying assets
    * `biNormCDF`: calculate bivariate normal CDF
    * `calculatePayoff`: calculate option payoff given full path
    * `calculateRM`: calculate risk measures given portfolio losses
    * `runMacro`: run macro replications of nested simulation procedures