# Theoretical Framework for Asymptotic Analysis of Absolute Difference

In a nested estimation problem, we are interested in the quantity 

$$\rho(g(X)) = \mathbb{E} \left[ h(g(X)) \right]$$

where $X \in \Omega$. 
$g(X)$ can't be directly evaluated, but it is the output of 

$$ g(X) = \mathbb{E}\left[ Y|X=x \right]\vert_{x=X} $$

In supervised learning, $g(\cdot)$ can be approximated by $\hat{g}^{\text{SL}}_{M, N}(\cdot)$. which is based on a chosen function family $\mathcal{G}$ and observations from the standard nested simulation procedure.

### The SNS Procedure 

The standard nested simulation procedure first simulate $M$ iid outer scenarios $X_1, \dots, X_M$ from $F_X$, the distribution of $X$.
For each $X_i$, again simulate $Y_{ij}$, $j = 1, \dots, N$ from $F_{Y|X_i}$, the conditional distribution of $Y$ given $X_i$. Given scenario $X_i$, the $Y_{ij}$ are conditionally iid. The total simulation budget $\Gamma = M \cdot N$. Let $f_X(x)$ denote the density of $X$.

Denote $\mathbf{X} = (X_1, \dots, X_M)$, the standard Monte Carlo estimator of $g(X_i)$ by
$$\hat{g}_N(X_i) = \frac{1}{N} \sum_{j=1}^N Y_{ij}; ~~~ Y_{ij} \sim F_{Y|X_i} $$

Let $(\hat{g}_N(\mathbf{X}))_{[1]}, \dots, (\hat{g}_N(\mathbf{X}))_{[M]}$ be the order statistics of $\hat{g}_N(X_1), \dots \hat{g}_N(X_M)$. 
The SNS estimator of $\rho$ is 
1. Nested expectation form:
$$\hat{\rho}_{M, N} = \frac{1}{M} \sum_{i=1}^M h(\hat{g}_N(X_i)) = \frac{1}{M} \sum_{i=1}^M h(\bar{Y}_{N, i}); ~~~ X_i \sim F_X$$
1. VaR:
$$\hat{\rho}_{M, N} = (\hat{g}_N(\mathbf{X}))_{\lceil \alpha M \rceil}$$
1. CVaR:
$$\hat{\rho}_{M, N} = (\hat{g}_N(\mathbf{X}))_{\lceil \alpha M \rceil} + \frac{1}{(1-\alpha) M} \sum_{i=1}^M \max \{\hat{g}_N(X_i) - (\hat{g}_N(\mathbf{X}))_{\lceil \alpha M \rceil}, 0 \}$$

Consider the observation pairs $(X_i, \hat{g}_N(X_i))$ for $i \in \{1, \dots, M\}$ as training data, we can use supervised learning to approximate $g(\cdot)$ by $\hat{g}^{\text{SL}}_{M, N}(\cdot)$.

Using the $M$ **training** samples, a nested Monte Carlo estimator of $\rho$ is given by

1. Nested expectation form:
$$\hat{\rho}^{\text{SL}, \text{Train}}_{M, N} = \frac{1}{M} \sum_{i=1}^M h(\hat{g}^{\text{SL}}_{M, N}(X_i)); ~~~ X_i \sim F_X$$
2. VaR:
$$\hat{\rho}^{\text{SL}, \text{Train}}_{M, N} = (\hat{g}^{\text{SL}}_{M, N}(\mathbf{X}))_{\lceil \alpha M \rceil}$$
3. CVaR:
$$\hat{\rho}^{\text{SL}, \text{Train}}_{M, N} = (\hat{g}^{\text{SL}}_{M, N}(\mathbf{X}))_{\lceil \alpha M \rceil} + \frac{1}{(1-\alpha) M} \sum_{i=1}^M \max \{\hat{g}^{\text{SL}}_{M, N}(X_i) - (\hat{g}^{\text{SL}}_{M, N}(\mathbf{X}))_{\lceil \alpha M \rceil}, 0 \}$$

where $(\hat{g}_{M, N}(\mathbf{X}))_{\lceil \alpha M \rceil}$ is the $\lceil \alpha M \rceil$-th order statistic of $\hat{g}^{\text{SL}}_{M, N}(X_1), \dots, \hat{g}_{M, N}(X_M)$.

Similarly, with $M'$ **test** samples of $X$, namely $\tilde{\mathbf{X}} = \tilde{X}_1, \dots, \tilde{X}_{M'}$, an estimator is given by

1. Nested expectation form:
$$\hat{\rho}^{\text{SL}, \text{Test}}_{M, N, M'} = \frac{1}{M'} \sum_{i=1}^{M'} h(\hat{g}^{\text{SL}}_{M, N}(\tilde{X}_i)); ~~~ \tilde{X}_i \sim F_X$$
2. VaR:
$$\hat{\rho}^{\text{SL}, \text{Test}}_{M, N, M'} = (\hat{g}^{\text{SL}}_{M, N}(\tilde{\mathbf{X}}))_{\lceil \alpha M' \rceil}$$
3. CVaR:
$$\hat{\rho}^{\text{SL}, \text{Test}}_{M, N, M'} = (\hat{g}^{\text{SL}}_{M, N}(\tilde{\mathbf{X}}))_{\lceil \alpha M' \rceil} + \frac{1}{(1-\alpha) M'} \sum_{i=1}^{M'} \max \{\hat{g}^{\text{SL}}_{M, N}(\tilde{X}_i) - (\hat{g}^{\text{SL}}_{M, N}(\tilde{\mathbf{X}}))_{\lceil \alpha M' \rceil}, 0 \}$$

where $(\hat{g}_{M, N}(\tilde{\mathbf{X}}))_{\lceil \alpha M' \rceil}$ is the $\lceil \alpha M' \rceil$-th order statistic of $\hat{g}^{\text{SL}}_{M, N}(\tilde{X}_1), \dots, \hat{g}_{M, N}(\tilde{X}_{M'})$. 

Note that
$\hat{g}^{\text{SL}}_{M, N}(\cdot)$ is derived from the training samples $(X_1, \hat{g}_N(X_1)), \dots, (X_M, \hat{g}_N(X_M))$.

One benefit of using supervised learning is to avoid the need of re-simulating the inner samples when a new set of outer scenarios is given. 
In the case of having a test set of outer scenarios, the total simulation budget is $\Gamma = M \cdot N + M'$.

We are interested in minimizing the rate of convergence, in probabilistic order, of $\left| \hat{\rho}^{\text{SL}}_{M, N} - \rho \right|$ in terms of the total simulation budget $\Gamma$.

## Our Contribution - Fill in the Gap

Except for the kernel ridge regression, all the other methods have their asymptotic convergence analysis focused on the mean squared error (MSE) of the estimator.
While [Wang, 2022](./papers/Wang_2022.pdf) claims that they have filled in the gap between squared root and cubic root convergence, they have used a different metric to measure the convergence rate.
In this note, we will use the same metric as in [Wang, 2022](./papers/Wang_2022.pdf) to compare the convergence rate of the above methods, by which we believe we have filled in the gap in the "fill the gap" paper.

## Notations and Terminologies
Let $a_n = \mathcal{O}_\mathbb{P}(b_n)$ denote that for any $\epsilon > 0$, there exists $C > 0$ such that $\mathbb{P}(a_n > C b_n) \leq \epsilon$ for all $n$ large enough.

# Asymptotic Analysis - Smooth $h$

## Standard Nested Simulation

**Assumption 1**: $h(g(X))$ has finite second moment, i.e., $\mathbb{E} \left[ \left( h(g(X)) \right)^2 \right] \leq C_2 < \infty$. 

**Collorary 1**: Variance of $h(g(X))$, $\sigma^2 \leq C_2 < \infty$

The absolute difference of the nested Monte Carlo estimator $\hat{\rho}^{\text{SNS}}_{M, N}$ with respect to the risk measure $\rho$ can be decomposed into two terms

Let $\rho_M = \frac{1}{M} \sum_{i=1}^M h(g(X_i))$ be the nested Monte Carlo estimator with the true function $g$.

$$
\begin{align}
\left| \hat{\rho}^{\text{SNS}}_{M, N} - \rho \right|
& \leq \left| \hat{\rho}^{\text{SNS}}_{M, N} - \rho_M \right| + \left| \rho_M - \rho \right| \nonumber \\
& = \left| \frac{1}{M} \sum_{i=1}^M h\left( \hat{g}^{\text{SNS}}_{M, N}(X_i) \right) -  \frac{1}{M} \sum_{i=1}^M h\left(g(X_i) \right)  \right| + \left| \frac{1}{M} \sum_{i=1}^M h\left(g(X_i) \right) - \mathbb{E}\left[ h(g(X))\right] \right| 
\end{align}
$$

where the inequality is from the triangle inequality.

The second term can be bounded by the central limit theorem (CLT) 

$$
\begin{equation}
\sqrt{M} \left( \frac{1}{M} \sum_{i=1}^M h\left(g(X_i) \right) - \mathbb{E}\left[ h(g(X))\right] \right) \xrightarrow{\mathcal{D}} \mathcal{N}(0, \sigma^2)
\end{equation}
$$

which implies that for $M$ large enough,

$$
\begin{align}
\mathbb{P} \left( \left| \frac{1}{M} \sum_{i=1}^M h\left(g(X_i) \right) - \mathbb{E}\left[ h(g(X))\right] \right| > C M^{-\frac{1}{2}} \right)
& = \mathbb{P} \left( \left| \sqrt{M} \left( \frac{1}{M} \sum_{i=1}^M h\left(g(X_i) \right) - \mathbb{E}\left[ h(g(X))\right]  \right) \right| > C \right) \nonumber \\
& \leq 1 - \text{erf}\left( \frac{C}{\sigma \sqrt{2}} \right) 
\end{align}
$$

which means that if we take $C = \sqrt{2} \sigma \text{erf}^{-1} \left( 1- \epsilon \right)$, then $\mathbb{P} \left( \left| \frac{1}{M} \sum_{i=1}^M h\left(g(X_i) \right) - \mathbb{E}\left[ h(g(X))\right] \right| > C M^{-\frac{1}{2}} \right) \leq \epsilon$.

Another way to see this is by noting that the right-hand side of **(3)** approaches $0$ when $C$ goes to infinity.

Hence, we have


$$
\begin{equation}
\left| \rho_M - \rho \right|
= \left| \frac{1}{M} \sum_{i=1}^M h\left(g(X_i) \right) - \mathbb{E}\left[ h(g(X))\right] \right| + \mathcal{O}_{\mathbb{P}}(M^{-\frac{1}{2}}) 
\end{equation}
$$

It remains to analyze the first term, which varies with the choice of supervised learning method and the form of $h$.


**Assumption SF.1** $h$ is a smooth function, i.e., $h$ has bounded first and second derivatives.

The analysis for smooth $h$ will be primarily based on the Taylor expansion of $h$ around $g(X_i)$.

$$
\begin{align}
& \left| \frac{1}{M} \sum_{i=1}^M h\left( \hat{g}^{\text{SNS}}_{M, N}(X_i) \right) -  h\left(g(X_i) \right) \right| \nonumber \\
& = \left| \frac{1}{M} \sum_{i=1}^M h'\left( g(X_i) \right) \left( \hat{g}^{\text{SNS}}_{M, N}(X_i) - g(X_i) \right) +  \frac{1}{2M} \sum_{i=1}^M h''\left( z_i \right) \left( \hat{g}^{\text{SL}}_{M, N}(X_i) - g(X_i) \right)^2 \right|  \nonumber \\
& \leq  \left| \frac{1}{M} \sum_{i=1}^M h'\left( g(X_i) \right) \left( \hat{g}^{\text{SNS}}_{M, N}(X_i) - g(X_i) \right) \right| +  \left| \frac{1}{2M} \sum_{i=1}^M h''\left( z_i \right) \left( \hat{g}^{\text{SNS}}_{M, N}(X_i) - g(X_i) \right)^2 \right|
\end{align}
$$

where $z_i$ is between $\hat{g}^{\text{SNS}}_{M, N}(X_i)$ and $g(X_i)$.

Bounds for terms in **(5)** are derived in the following subsections for different supervised learning methods.


The standard nested simulation estimator of $g(X_i)$ is given by

$$\hat{g}^{\text{SNS}}_{M, N}(X_i) = \hat{g}_N(X_i) = \bar{Y}_{N, i} $$

where $\bar{Y}_{N, i} = \frac{1}{N} \sum_{j=1}^N Y_{ij}$.

The following assumptions are needed for the convergence analysis of the standard nested simulation estimator of $\rho$.

**Assumption SF.SNS.1**: $h$ has bounded first and second order derivative, i.e., there exists $C_{h, 1} > 0$ such that $|h'(x)| \leq C_{h, 1}$ for all $x \in \mathbb{R}$, and there exists $C_{h, 2} > 0$ such that $|h''(x)| \leq C_{h, 2}$ for all $x \in \mathbb{R}$.

**Assumption SF.SNS.2**: $\hat{g}_N(X) = g(X) + \bar{Z}_N(X)$, where the simulation noise $\bar{Z}_N(X)$ has zero mean and variance $\nu(X) / N$, where the conditional variance $\nu(X)$ is bounded, i.e., there exists $C_{\nu, 1} > 0$ such that $\nu(x) \leq C_{\nu, 1}$ for all $x \in \mathbb{R}$. 

**Assumption 1.SF.SNS.3A**: The squared simulation noise $\bar{Z}_N^2(X)$ has variance $\nu_2(X) / N^2$, where $\nu_2(X)$ is bounded, i.e., there exists $C_{\nu,2} > 0$ such that $\nu_2(X) \leq C_{\nu,2}$ for all $x \in \mathbb{R}$.

**Assumption 1.SF.SNS.3B**: There exist $\delta >0$ such that 
$$\mathbb{E} \left[ \left| \sqrt{N} \cdot \bar{Z}_N(X) \right|^{4+\delta} \right] < \infty.$$

This assumption is sufficient for the uniform integrability of $N \cdot \bar{Z}_N^2(X)$ and $N^2 \cdot \bar{Z}_N^4(X)$, which is needed for the convergence analysis of the second term in **(5)**.

Here, we start by showing the convergence order of the first term in **(5)**.

$$
\begin{align}
\left|\frac{1}{M} \sum_{i=1}^M h'\left( g(X_i) \right) \left( \hat{g}_{N}(X_i) - g(X_i) \right) \right| 
& = \left|\frac{1}{M} \sum_{i=1}^M h'\left( g(X_i) \right) \bar{Z}_N(X_i) \right| \nonumber \\
& \leq C_{h, 1} \left|\frac{1}{M} \sum_{i=1}^M \bar{Z}_N(X_i) \right|
\end{align}
$$

The sum in **(6)** can be bounded by noting the boundedness of variance and independence of $\bar{Z}_N(X_i)$.

$$
\begin{align}
\mathbb{E} \left[ \frac{1}{M} \sum_{i=1}^M \bar{Z}_N(X_i) \right] & = 0 \nonumber \\
\text{Variance} \left( \frac{1}{M} \sum_{i=1}^M \bar{Z}_N(X_i) \right) & = \frac{1}{M^2} \sum_{i=1}^M \text{Variance} \left( \bar{Z}_N(X_i) \right) \nonumber \\
& = \frac{1}{M^2} \sum_{i=1}^M \frac{\nu(X_i)}{N} \leq \frac{C_{\nu, 1}}{MN}
\end{align}
$$

Hence, by Chebyshev's inequality, we have

$$
\begin{equation}
\mathbb{P} \left( \left|\frac{1}{M} \sum_{i=1}^M \bar{Z}_N(X_i) \right| > k (MN)^{-\frac{1}{2}} \right) \leq \frac{\text{Variance} \left( \frac{1}{M} \sum_{i=1}^M \bar{Z}_N(X_i) \right)}{k^2 (MN)^{-1}} \leq \frac{C_{\nu, 1}}{k^2} \nonumber \\ 
\end{equation}
$$

Hence, take $a_n = \left|\frac{1}{M} \sum_{i=1}^M \bar{Z}_N(X_i) \right|$ and $b_n = (MN)^{-\frac{1}{2}}$, and $\epsilon$, 
there exist $C = \sqrt{\frac{C_{\nu, 1}}{\epsilon}}$ such that 

$$\mathbb{P} \left( \left|\frac{1}{M} \sum_{i=1}^M \bar{Z}_N(X_i) \right| > C (MN)^{-\frac{1}{2}} \right) \leq \epsilon$$

That is, $\left|\frac{1}{M} \sum_{i=1}^M \bar{Z}_N(X_i) \right| = \mathcal{O}_\mathbb{P}((MN)^{-\frac{1}{2}})$.

Hence, the first term in **(5)** has the same order of convergence.

$$
\begin{equation}
\left|\frac{1}{M} \sum_{i=1}^M h'\left( g(X_i) \right) \left( \hat{g}_{N}(X_i) - g(X_i) \right) \right| = \mathcal{O}_\mathbb{P}((MN)^{-\frac{1}{2}})
\end{equation}
$$

It remains to show the second term in **(5)**.

$$
\begin{align}
\left| \frac{1}{2M} \sum_{i=1}^M h''\left( z_i \right) \left( \hat{g}^{\text{SL}}_{M, N}(X_i) - g(X_i) \right)^2 \right|
& = \left|\frac{1}{2M} \sum_{i=1}^M h''\left( z_i \right) \bar{Z}^2_N(X_i) \right|\nonumber \\
& \leq \frac{C_{h, 2}^2}{2} \left|\frac{1}{M} \sum_{i=1}^M \bar{Z}^2_N(X_i) \right|  \nonumber \\
& \leq \frac{C_{h, 2}^2}{2} \left|\frac{1}{M} \sum_{i=1}^M \bar{Z}^2_N(X_i) - \frac{\nu(X)}{N} \right| + \frac{C_{h, 2}^2}{2} \frac{\nu(X)}{N} 
\end{align}
$$

where **(9)** can be bounded by using Chebyshev's inequality and the boundedness of $\nu_2(X)$.

$$
\begin{align}
\mathbb{P} \left( \left|\frac{1}{M} \sum_{i=1}^M \bar{Z}^2_N(X_i) - \frac{\sigma^2}{N} \right| > C M^{-\frac{1}{2}}N^{-1} \right) 
& \leq \frac{\text{Variance} \left( \frac{1}{M} \sum_{i=1}^M \bar{Z}^2_N(X_i) \right)}{C^2 M^{-1}N^{-2}} \nonumber \\
& = \frac{\frac{1}{M} \text{Variance} \left(\bar{Z}^2_N(X) \right)}{C^2 M^{-1}N^{-2}} \nonumber \\
& = \frac{\text{Variance} \left(\bar{Z}^2_N(X) \right)}{C^2 N^{-2}} \nonumber \\
& = \frac{\nu_2(X)}{C^2} \leq \frac{C_{\nu, 2}}{C^2} 
\end{align}
$$ 

where the boundedness of $\text{Variance} \left(\bar{Z}^2_N(X) \right)$ is from **Assumption 1.SF.SNS.3**. 

Alternatively, the assumption can be shown using the central limit theorem if $\bar{Z}_N(X)$ is the sample mean of inner simulation noises, that is, 

$$
\bar{Z}_N(X) = \frac{1}{N} \sum_{j=1}^N Z_j(X)
$$

Then, by the CLT, we have
$$
\begin{align}
\sqrt{\frac{N}{\nu(X)}} \bar{Z}_N(X) = \sqrt{\frac{N}{\nu(X)}}  \left( \frac{1}{N} \sum_{j=1}^N Z_j(X) \right) 
& \xrightarrow{\mathcal{D}} \mathcal{N}(0, 1) 
\end{align}
$$

<!-- or equivalently,

$$
\begin{align}
\sqrt{N} \cdot \bar{Z}_N(X)
& \xrightarrow{\mathcal{D}} \mathcal{N}(0, \nu(X))
\end{align}
$$ -->


From continuous mapping theorem (Theorem 1.14 in [DasGupta, 2008](./proof_ref/DasGupta_2008.pdf)), we have when $N \to \infty$,

$$
\begin{align}
\left( \sqrt{\frac{N}{\nu(X)}} \cdot \bar{Z}_N(X) \right)^2 & \xrightarrow{\mathcal{D}} \chi^2(1) 
\end{align}
$$

By **Assumption 1.SF.SNS.3B**, the uniform integrability ensures there exists $\delta > 0$ such that

$$
\mathbb{E} \left[ \left| \sqrt{N} \cdot \bar{Z}_N(X) \right|^{4+\delta} \right] < \infty.
$$

Since $\nu(X)$ is a bounded constant, we have

$$
\mathbb{E} \left[ \left| \sqrt{\frac{N}{\nu(X)}} \cdot \bar{Z}_N(X) \right|^{4+\delta} \right] < \infty.
$$

With algebraic manipulation, we have

$$
\begin{equation}
\mathbb{E} \left[ \left| \left(\sqrt{\frac{N}{\nu(X)}} \cdot \bar{Z}_N(X)\right)^2 \right|^{2+\delta} \right] < \infty.
\end{equation}
$$

By Theorem 6.2 in [DasGupta, 2008](./proof_ref/DasGupta_2008.pdf), we have the convergence of the moments of $\left(\sqrt{\frac{N}{\nu(X)}} \cdot \bar{Z}_N(X)\right)^2$.

$$
\begin{align*}
\mathbb{E} \left[ \left( \sqrt{\frac{N}{\nu(X)}} \cdot \bar{Z}_N(X) \right)^2 \right] 
& \to 1 \nonumber \\
\mathbb{E} \left[ \left( \sqrt{\frac{N}{\nu(X)}} \cdot \bar{Z}_N(X) \right)^4 \right] 
& \to 3
\end{align*}
$$

By Proposition 6.1 in [DasGupta, 2008](./proof_ref/DasGupta_2008.pdf), we have
 
$$
\begin{equation}
\text{Variance} \left( \sqrt{\frac{N}{\nu(X)}} \cdot \bar{Z}_N(X) \right)  \nonumber \to 2
\end{equation}
$$


Hence for sufficiently large $N$,
$$
\text{Variance} \left( \bar{Z}^2_N(X)\right) = \mathcal{O}(N^{-2})
$$

Continuing the discussion from **(10)**, take $a_n = \left| \frac{1}{M} \sum_{i=1}^M \bar{Z}^2_N(X_i) - \frac{\sigma^2}{N} \right|$ and $b_n = M^{-\frac{1}{2}}N^{-1}$, and $\epsilon$, there exist $C = \sqrt{\frac{C_{\nu, 2}}{\epsilon}}$ such that

$$
\begin{equation}
\mathbb{P} \left( \left|\frac{1}{M} \sum_{i=1}^M \bar{Z}^2_N(X_i) - \frac{\sigma^2}{N} \right| > C M^{-\frac{1}{2}}N^{-1} \right) \leq \epsilon
\end{equation}
$$

That is, $\left|\frac{1}{M} \sum_{i=1}^M \bar{Z}^2_N(X_i) - \frac{\sigma^2}{N} \right| = \mathcal{O}_\mathbb{P}(M^{-\frac{1}{2}}N^{-1})$.

Therefore, the second term in **(5)** converges at the rate of $\mathcal{O}_\mathbb{P}(N^{-1})$.

$$
\begin{equation}
\left| \frac{1}{2M} \sum_{i=1}^M h''\left( z_i \right) \left( \hat{g}^{\text{SL}}_{M, N}(X_i) - g(X_i) \right)^2 \right| = \mathcal{O}_\mathbb{P}(N^{-1})
\end{equation}
$$

Combining the results from **(4)**, **(8)**, and **(12)**, we have

$$
\begin{equation}
\left| \hat{\rho}^{SNS}_{M, N} - \rho \right| = \mathcal{O}_\mathbb{P}(M^{-\frac{1}{2}} + N^{-1})
\end{equation}
$$

Take $M = \mathcal{O}(\Gamma^{\frac{2}{3}})$, $N = \mathcal{O}(\Gamma^{\frac{1}{3}})$, we have

$$
\begin{equation}
\left| \hat{\rho}^{SNS}_{M, N} - \rho \right| = \mathcal{O}_\mathbb{P}(\Gamma^{-\frac{1}{3}})
\end{equation}
$$

## Regression

The regression-based nested simulation estimates $\rho$ with 

$$
\begin{equation}
\hat{\rho}^{\text{REG}, \text{Test}}_{M, N, M'} = \frac{1}{M'} \sum_{i=1}^{M'} h\left( \hat{g}^{\text{REG}}_{M, N}(\tilde{X}_i) \right)
\end{equation}
$$

where $\tilde{X}_i$'s are the test samples of $X$, $\hat{g}^{\text{REG}}_{M, N}(X) = \Phi(X) \hat{\beta}$, and $\hat{\beta}$ is the solution of the following optimization problem


$$
\begin{equation}
\hat{\beta} \in \argmin_{\beta \in \mathbb{R}^k} \frac{1}{M} \sum_{i=1}^M \left( \Phi(X_i) \beta - \hat{g}_N(X_i) \right)^2
\end{equation}
$$

Let:
* $\bar{Z}_N(X) = \hat{g}_N(X) - g(X)$ be the error from the standard inner simulation.
* $\mathcal{MB}(X) = g(X) - \Phi(X) \beta^*$ be the model error of the regression estimator

where $\beta^*$ is the best regression parameter, the optimal solution of the following optimization problem

$$
\begin{equation}
\beta^* \in \argmin_{\beta \in \mathbb{R}^k} \mathbb{E} \left[ \left( \Phi(X) \beta - g(X) \right)^2 \right]
\end{equation}
$$

In the proof of regression estimator, we will decompose the absolute difference into two terms.

$$
\begin{equation}
\left| \hat{\rho}^{\text{REG}}_{M, N, M'} - \rho \right| \leq \left| \hat{\rho}^{\text{REG}}_{M, N, M'} - \tilde{\rho}_M \right| + \left| \tilde{\rho}_M - \rho \right|
\end{equation}
$$

where $\tilde{\rho}_M = \mathbb{E} \left[ \hat{\rho}^{\text{REG}}_{M, N, M'} \right]  = \mathbb{E} \left[ h\left(\Phi(X) \hat{\beta}\right) | \overrightarrow{X}, \overrightarrow{\bar{Z}_N(X)} \right]$, with $\overrightarrow{X} = (X_1, \dots, X_M)$, $\overrightarrow{\bar{Z}_N(X)} = (\bar{Z}_N(X_1), \dots, \bar{Z}_N(X_M))$.

We will bound each term in **(21)** separately.

<!-- **Assumption 2.SF.REG.1**: The variance $\mathbb{E} \left[ \text{Var}(\hat{g}_N(X) |X) \right] < \infty$. -->

**Assumption 2.SF.REG.1**: The design matrix $\Phi(\overrightarrow{X})$ has full column rank.   

**Assumption 2.SF.REG.2**: Given any training set $\overrightarrow{X}$, 
$$\text{Variance}\left( h\left(\Phi(X) \hat{\beta}\right) | \overrightarrow{X}, \overrightarrow{\bar{Z}_N(X)} \right) \leq C_{\text{REG}} < \infty$$

**Assumption 2.SF.REG.3**: The function $h$ has bounded first second derivative, i.e., there exist $C_{h, 1} > 0$ such that $|h'(x)| \leq C_{h, 1}$ for all $x \in \mathbb{R}$, and there exists $C_{h,2} > 0$ such that $|h''(x)| \leq C_{h,2}$ for all $x \in \mathbb{R}$.

The first terms in **(21)** can be bounded by Chebyshev's inequality.

Since $h\left( \hat{g}^{\text{REG}}_{M, N}(\tilde{X}_1) \right), \dots, h\left( \hat{g}^{\text{REG}}_{M, N}(\tilde{X}_{M'}) \right)$ are independent and identically distributed given the training data $\overrightarrow{X}$ and $\overrightarrow{\bar{Z}_N(X)}$, we have

$$
\begin{align}
\mathbb{E} \left[ h\left( \hat{g}^{\text{REG}}_{M, N}(X) \right) | \overrightarrow{X}, \overrightarrow{\bar{Z}_N(X)} \right]
& = \mathbb{E} \left[ h\left(\Phi(X) \hat{\beta}\right) | \overrightarrow{X}, \overrightarrow{\bar{Z}_N(X)} \right] = \tilde{\rho}_M \nonumber \\
\text{Variance}\left( h\left( \hat{g}^{\text{REG}}_{M, N}(X) \right) | \overrightarrow{X}, \overrightarrow{\bar{Z}_N(X)} \right)
& = \text{Variance}\left( h\left(\Phi(X) \hat{\beta}\right) | \overrightarrow{X}, \overrightarrow{\bar{Z}_N(X)} \right) \leq C_{\text{REG}}
\end{align}
$$

Applying Cheybyshev's inequality, we have

$$
\begin{align}
\mathbb{P} \left( \left| \hat{\rho}^{\text{REG}}_{M, N, M'} - \tilde{\rho}_M \right| > kM'^{-\frac{1}{2}} \right)
& \leq \frac{\text{Variance} \left( \frac{1}{M'} \sum_{i=1}^{M'} h\left( \hat{g}^{\text{REG}}_{M, N}(\tilde{X}_i) \right) \right)}{k^2 M'^{-1}} \nonumber \\
& = \frac{\text{Variance} \left( h\left( \hat{g}^{\text{REG}}_{M, N}(X) \right) \right)}{k^2} \nonumber \\
& \leq C_{\text{REG}} \cdot k^{-2}
\end{align}
$$

Therefore, let $a_n = \left| \hat{\rho}^{\text{REG}}_{M, N, M'} - \tilde{\rho}_M \right|$ and $b_n = M'^{-\frac{1}{2}}$, and $\epsilon > 0$, there exists $C = \sqrt{\frac{C_{\text{REG}}}{\epsilon}}$ such that

$$
\begin{equation}
\mathbb{P} \left( \left| \hat{\rho}^{\text{REG}}_{M, N, M'} - \tilde{\rho}_M \right| > C M'^{-\frac{1}{2}} \right) \leq \epsilon
\end{equation}
$$

Hence,

$$
\begin{equation}
\left| \hat{\rho}^{\text{REG}}_{M, N, M'} - \tilde{\rho}_M \right| = \mathcal{O}_\mathbb{P}(M'^{-\frac{1}{2}})
\end{equation}
$$

The **Assumption 2.SF.REG.2** is somewhat ambiguous. However, it is currently a step ahead from [Broadie, 2015](papers/Broadie_2015.pdf), where the convergence from $\hat{\rho}^{\text{REG}}_{M, N, M'}$ to $\tilde{\rho}_M$ is not explicitly shown.

It remains to bound the second term in **(21)**. 
By Taylor expansion, we have

$$
\begin{align}
\left| \tilde{\rho}_M - \rho \right|
& = \left| \mathbb{E} \left[ h\left(\Phi(X) \hat{\beta}\right) | \overrightarrow{X}, \overrightarrow{\bar{Z}_N(X)} \right] - \mathbb{E} \left[ h(g(X)) \right] \right| \nonumber \\
& = \left| \mathbb{E} \left[ h\left(\Phi(X) \hat{\beta}\right) - h(g(X)) | \overrightarrow{X}, \overrightarrow{\bar{Z}_N(X)} \right] \right| \nonumber \\
& = \left| \mathbb{E} \left[ h'\left( g(X) \right) \left( \Phi(X) \hat{\beta} - g(X) \right) | \overrightarrow{X}, \overrightarrow{\bar{Z}_N(X)} \right] + \mathbb{E} \left[ \frac{1}{2} h''\left( z \right) \left( \Phi(X) \hat{\beta} - g(X) \right)^2 | \overrightarrow{X}, \overrightarrow{\bar{Z}_N(X)} \right] \right| 
\end{align}
$$

We bound the first term of **(26)** by further decomposing it into two terms.

$$
\begin{align}
\mathbb{E} \left[ h'\left( g(X) \right) \left( \hat{g}^{\text{REG}}_{M, N}(X) - g(X) \right) | \overrightarrow{X}, \overrightarrow{\bar{Z}_N(X)}\right]
& = \mathbb{E} \left[ h'\left( g(X) \right) \left( \Phi({X}) \hat{\beta} - \Phi(X) \beta^* + \Phi(X) \beta^* - g(X) \right) | \overrightarrow{X}, \overrightarrow{\bar{Z}_N(X)}\right] \nonumber \\
& = \mathbb{E} \left[ h'\left( g(X) \right) \left( \Phi(X) \hat{\beta} - \Phi(X) \beta^* \right) | \overrightarrow{X}, \overrightarrow{\bar{Z}_N(X)}\right] + \mathbb{E} \left[ h'\left( g(X) \right) \left( \Phi(X) \beta^* - g(X) \right) \right] \nonumber \\
& = \mathbb{E} \left[ h'\left( g(X) \right) \left( \Phi(X)  (\hat{\beta} - \beta^*) \right) | \overrightarrow{X}, \overrightarrow{\bar{Z}_N(X)} \right] - \mathbb{E} \left[ h'\left( g(X) \right) (\mathcal{MB}(X)) \right] \nonumber \\
& = \mathbb{E} \left[ h'\left( g(X) \right) \Phi(X) \right] (\hat{\beta} - \beta^*) - \mathbb{E} \left[ h'\left( g(X) \right) (\mathcal{MB}(X)) \right]
\end{align}
$$

The last equality is due to $\hat{\beta}$ being a number when conditioning on $\overrightarrow{X}$ and $\overrightarrow{\bar{Z}_N(X)}$, and it does not depend on $X$. 
The first term of **(27)** can be bounded by applying regression theory, and the second term is combined into the model error.

From White, 2001, as $n\to \infty$, we have

$$
\begin{equation}
\sqrt{M} (\hat{\beta} - \beta^*) \xrightarrow{\mathcal{D}} \mathcal{N}\left(0, \Sigma_{\mathcal{MB}} + \frac{\Sigma_\nu}{N} \right)
\end{equation}
$$

and 

$$
\begin{equation}
\sqrt{M}  \mathbb{E} \left[ h'\left( g(X) \right) \Phi(X) \right] \left( \hat{\beta} - \beta^* \right) \xrightarrow{\mathcal{D}} \mathcal{N}\left((0,  \mathbb{E} \left[ h'\left( g(X) \right) \Phi(X) \right] \left(  \Sigma_{\mathcal{MB}} + \frac{\Sigma_{\mathcal{\nu}}}{N} \right) \mathbb{E} \left[ h'\left( g(X) \right) \Phi(X) \right]^\top \right)
\end{equation}
$$

where $\Sigma_{\mathcal{MB}} = \mathbb{E} \left[ \mathcal{MB}(X)^2 \Phi(X)^\top \Phi(X) \right]$ and $\Sigma_{\mathcal{\nu}} = \mathbb{E} \left[ \nu(X) \Phi(X)^\top \Phi(X) \right]$, and $\nu(X)/N$ is the variance of the simulation noise $\bar{Z}_N(X)$ from the standard inner simulation.

Let $\mathcal{B}_{\mathcal{MB}, M}$ be the bias of the regression estimator with $M$ outer scenarios, i.e., 
$$\mathcal{B}_{\mathcal{MB}, M} = \mathbb{E} \left[ h'\left( g(X) \right) (\mathcal{MB}(X)) \right] - \mathbb{E} \left[ \frac{1}{2} h''\left( z \right) \left( \Phi(X) \hat{\beta} - g(X) \right)^2 | \overrightarrow{X}, \overrightarrow{\bar{Z}_N(X)} \right]$$

then we have

$$
\begin{equation}
\sqrt{M}\cdot \left( \tilde{\rho}_M - \rho - \mathcal{B}_{\mathcal{MB}, M} \right) \overset{\mathcal{D}}{\to} \mathcal{N} \left( 0, \mathbb{E} \left[ h'\left( g(X) \right) \left( \Phi(X) \right) \right] \left( \Sigma_\mathcal{M} + \frac{\Sigma_\nu}{N} \right) \left( \mathbb{E} \left[ h'\left( g(X) \right) \left( \Phi(X) \right) \right]\right)^\top \right)
\end{equation}
$$

It remains to find the asymptotic model bias $\mathcal{B}_{\mathcal{MB}}^*$ when $M \to \infty$.

From **(28)**, continuous mapping theorem implies that

$$
\begin{equation}
\tilde{\rho}_M = \mathbb{E} \left[ h\left(\Phi(X) \hat{\beta}\right) | \overrightarrow{X}, \overrightarrow{\bar{Z}_N(X)} \right] \xrightarrow{\mathbb{P}} \mathbb{E} \left[ h(\Phi(X) \beta^*)\right]
\end{equation}
$$

and **(29)** gives

$$
\begin{equation}
\left( \tilde{\rho}_M - \rho - \mathcal{B}_{\mathcal{MB}, M} \right) \overset{\mathbb{P}}{\to} 0
\end{equation}
$$

Combining **(31)** and **(32)**, we have as $M \to \infty$,

$$
\begin{equation}
\mathcal{B}_{\mathcal{MB}, M} \overset{\mathbb{P}}{\to} \mathbb{E} \left[ h(\Phi(X) \beta^*)\right] - \rho
\end{equation}
$$

It remains to bound the asymptotic model bias $\mathcal{B}_{\mathcal{MB}, M}$.

$$
\begin{align}
\mathcal{B}_{\mathcal{MB}, M}
& = \mathbb{E} \left[ h'\left( g(X) \right) (\mathcal{MB}(X)) \right] - \mathbb{E} \left[ \frac{1}{2} h''\left( z \right) \left( \Phi(X) \hat{\beta} - g(X) \right)^2 | \overrightarrow{X}, \overrightarrow{\bar{Z}_N(X)} \right] \nonumber \\
& \leq \left|\mathbb{E} \left[ h'\left( g(X) \right) (\mathcal{MB}(X)) \right]\right| + \left| \mathbb{E} \left[ \frac{1}{2} h''\left( z \right) \left( \Phi(X) \hat{\beta} - g(X) \right)^2 | \overrightarrow{X}, \overrightarrow{\bar{Z}_N(X)} \right]\right| \nonumber \\
& = \left|\mathbb{E} \left[ h'\left( g(X) \right) (\mathcal{MB}(X)) \right] \right| + \left| \mathbb{E} \left[ \frac{1}{2} h''\left( z \right) \left( \Phi(X) (\hat{\beta} - \beta^*) + \Phi(X)\beta^* - g(X) \right)^2 | \overrightarrow{X}, \overrightarrow{\bar{Z}_N(X)} \right] \right| \nonumber \\
& \leq C_{h, 1} \left| \mathbb{E}  \left[\mathcal{MB}(X)\right] \right| + \frac{C_{h, 2}}{2} \mathbb{E} \left| \left[ \left(\Phi(X) (\hat{\beta} - \beta^*) - \mathcal{MB}(X) \right)^2 | \overrightarrow{X}, \overrightarrow{\bar{Z}_N(X)} \right] \right| \nonumber \\
& \leq C_{h, 1} \left| \mathbb{E}  \left[\mathcal{MB}(X)\right] \right| + \frac{C_{h, 2}}{2} \mathbb{E} \left[ \left| \left(\Phi(X) (\hat{\beta} - \beta^*) - \mathcal{MB}(X) \right)^2\right| | \overrightarrow{X}, \overrightarrow{\bar{Z}_N(X)} \right] \nonumber \\
& \leq C_{h, 1} \left| \mathbb{E}  \left[\mathcal{MB}(X)\right] \right| + \frac{C_{h, 2}}{2} \mathbb{E} \left[\left(\Phi(X) (\hat{\beta} - \beta^*) \right)^2 | \overrightarrow{X}, \overrightarrow{\bar{Z}_N(X)} \right] + \frac{C_{h, 2}}{2} \mathbb{E} \left[  \mathcal{MB}(X) ^2 \right] 
\end{align}
$$

where the first and third terms in **(34)** are the model bias, and the second term can be further bounded by

$$
\begin{align}
\mathbb{E} \left[\left(\Phi(X) (\hat{\beta} - \beta^*) \right)^2 | \overrightarrow{X}, \overrightarrow{\bar{Z}_N(X)}\right]
& = \left(\hat{\beta} - \beta^*\right)^\top \mathbb{E} \left[ \Phi(X)^\top \Phi(X) \right] \left(\hat{\beta} - \beta^*\right) \nonumber \\
& = \mathcal{O}_\mathbb{P}(M^{-1})
\end{align}
$$

where the last equality is from **(28)** and the $\mathbb{E} \left[ \Phi(X)^\top \Phi(X) \right]$ being a scalar.

To summarize, we have characterized the asymptotic distribution of the regression estimator of $\rho$ to its own expectation, and the difference between its expectation and $\rho$, which involves the model bias with the remaining terms converges in the order of $M$.

$$
\begin{align}
\left| \tilde{\rho}_M - \rho \right| 
& = \left| \tilde{\rho}_M - \rho - \mathcal{B}_{\mathcal{MB}, M} + \mathcal{B}_{\mathcal{MB}, M} \right| \nonumber \\
& \leq \left| \tilde{\rho}_M - \rho - \mathcal{B}_{\mathcal{MB}, M} \right| + \left| \mathcal{B}_{\mathcal{MB}, M} \right| \nonumber \\
& \leq \mathcal{O}_\mathbb{P}(M^{-\frac{1}{2}}) + \mathcal{O}_\mathbb{P}(M^{-1}) + C_{h, 1} \left| \mathbb{E}  \left[\mathcal{MB}(X)\right] \right| + \frac{C_{h, 2}}{2} \mathbb{E} \left[  \mathcal{MB}(X) ^2 \right] 
\end{align}
$$

Therefore, we have the result for the regression estimator.

$$
\begin{equation}
\left| \hat{\rho}^{\text{REG}}_{M, N, M'} - \rho \right| = \mathcal{O}_\mathbb{P}(M'^{-\frac{1}{2}}) + \mathcal{O}_\mathbb{P}(M^{-\frac{1}{2}}) + C_{h, 1} \left| \mathbb{E}  \left[\mathcal{MB}(X)\right] \right| + \frac{C_{h, 2}}{2} \mathbb{E} \left[  \mathcal{MB}(X) ^2 \right]
\end{equation}
$$

when $\mathcal{MB}(X) = 0$, we have

$$
\begin{equation}
\left| \hat{\rho}^{\text{REG}}_{M, N, M'} - \rho \right| = \mathcal{O}_\mathbb{P}(M'^{-\frac{1}{2}}) + \mathcal{O}_\mathbb{P}(M^{-\frac{1}{2}})
\end{equation}
$$

Set $M = \mathcal{O}(\Gamma)$ and $M' = \mathcal{O}(\Gamma)$, we have

$$
\begin{equation}
\left| \hat{\rho}^{\text{REG}}_{M, N, M'} - \rho \right| = \mathcal{O}_\mathbb{P}(\Gamma^{-\frac{1}{2}})
\end{equation}
$$

### Alternative Proof

From **Theorem 1** in [Own Result, 2023](./convergenceRes.md), we have the convergence of the regression estimator.

When $\mathcal{MB}(X) = 0$, we have

$$
\begin{equation}
\left| \hat{\rho}^{\text{REG}}_{M, N, M'} - \rho \right| = \mathcal{O}_\mathbb{P}(\Gamma^{-\frac{1}{2}})
\end{equation}
$$

## Kernel Smoothing with k-Nearest Neighbors

In [Hong, 2017](./papers/Hong_2017.pdf), the convergence results were shown for the Nadaraya-Watson kernel smoothing estimator. 
However, the authors use kNN in their numerical studies.

In this section, we will show the convergence of the kernel smoothing estimator with kNN.

Let $\hat{g}^{\text{kNN}}_{M, N}(X)$ be the kNN estimator of $g(X)$ with $M$ outer scenarios and $N$ inner samples. 
The kNN-based nested simulation estimators of $\rho$ is defined as

$$\hat{\rho}^{\text{kNN}, \text{Train}}_{M, N} = \frac{1}{M} \sum_{i=1}^M h(\hat{g}^{\text{kNN}}_{M, N}(X_i)); ~~~ X_i \sim F_X$$

$$\hat{\rho}^{\text{kNN}, \text{Test}}_{M, N, M'} = \frac{1}{M'} \sum_{i=1}^{M'} h(\hat{g}^{\text{kNN}}_{M, N}(\tilde{X}_i)); ~~~ \tilde{X}_i \sim F_X$$

In order to show the convergence of $\hat{\rho}^{kNN}_{M, N}$, we will first show the convergence of the inner estimator $\hat{g}^{\text{kNN}}_{M, N}(X)$.

### kNN Estimator

We will first show the convergence of the kNN estimator at a fixed point $x$, and use similar techniques as in [Hong, 2017](./papers/Hong_2017.pdf) to show the convergence of the overall estimator of $\rho$.

Treating the outer scenarios and the corresponding standard Monte Carlo estimators $\hat{g}_N(X_i)$'s as the training data, we have the kNN estimator at a fix point $x$ as

$$
\begin{equation}
\hat{g}^{\text{kNN}}_{M, N}(x) = \sum_{i=1}^M v_{M, i} \cdot \hat{g}_N(X_i)
\end{equation}
$$

where $v_{M, 1}, \dots, v_{M, M}$ is a probability vector. 

**Example**: Let $v_{M, i} = a_{M, j}$ if $X_i$ is the $j$-th nearest neighbor of $x$, with $a_{M, 1} \geq a_{M, 2} \geq \dots \geq a_{M, M}$, where $a_{M, j} = \frac{1}{k}$ if $j \leq k$ and $0$ otherwise gives the kNN estimator with $k$ nearest neighbors.

For the kNN regressor to converge, we need the following assumptions.

**Assumption SF.KNN.1**: The function $g$ is bounded.
$$
g(x) \leq B ~~~ \forall x \in \mathbb{R}^d
$$

**Assumption SF.KNN.2**: $g$ is locally Lipschitz continuous on at $x$, that is, 

$$
\left| g(x) - g(x') \right| \leq K \left| x - x' \right| 
$$

The above Lipschitz condition only needs to be satisfied locally on an open neighborhood of the fixed $x$.

**Assumption SF.KNN.3**: Suppose $\sum_{i=k+1}^M v_{M, i} = \mathcal{O}(b_M)$ almost surely, and let $\| v_M\|_2 = \sqrt{\sum_{i=1}^M v_{M, i}^2}$. 
Assume $b_M \to 0$, $\| v_M \|_2 \to 0$, $\frac{k}{M} \to 0$, and $\frac{k}{\log(M)} \to 0$ as $M \to \infty$.

**Assumption SF.KNN.4A**: $\mathbb{E} \left[ |\bar{Z}_N(X)|^r \right] < \infty$ for some $r > 2$.

**Assumption SF.KNN.4B**: For any $a>0$, $\mathbb{P} \left( |\bar{Z}_N(X)| > a \right) \leq e^{-Ca^p}$ for some $C>0$ amd $p>0$.

From [Lian, 2011](./proof_ref/Lian_2011.pdf), we have the following results.

**Lemma SF.KNN.1**: Suppose Assumptions SF.KNN.1, SF.KNN.2, SF.KNN.3, and SF.KNN.4A hold. 
Then, for any $x \in \mathbb{R}^d$, we have

$$
| \hat{g}^{\text{kNN}}_{M, N}(x) - g(x) | = \mathcal{O} \left( \left(\phi^{-1}(2k/M)\right)^\alpha + \left(\log(M)\right)^{1/2} k^{-1/2} \right)
$$

**Lemma SF.KNN.2**: Suppose Assumptions SF.KNN.1, SF.KNN.2, SF.KNN.3, and SF.KNN.4B hold. 
Then, for any $x \in \mathbb{R}^d$, we have

$$
| \hat{g}^{\text{kNN}}_{M, N}(x) - g(x) | = \mathcal{O} \left( \left(\phi^{-1}(2k/M)\right)^\alpha + \left(\log(M)\right)^{1+1/p} k^{-1/2} \right)
$$

### Alternative Proof for Nadaraya-Watson Estimator

The Nadaraya-Watson estimator $\hat{g}^{\text{KS}}_{M, N}(X)$ is defined as

$$
\begin{equation}
\hat{g}^{\text{KS}}_{M, N}(X) = \frac{\sum_{i=1}^M K_w(X - X_i) \hat{g}_N(X_i)}{\sum_{i=1}^M K_w(X - X_i)}
\end{equation}
$$

where $K_w(X - X_i) = \frac{1}{w^d} K\left(\frac{|X - X_i|}{w}\right)$ is the kernel function with bandwidth $w$.

The kernel smoothing estimator $\hat{g}^{\text{KS}}_{M, N}(X)$ is a weighted average of the standard inner simulation estimator $\hat{g}_N(X_i)$, where the weight is determined by the kernel function $K_w(X - X_i)$.

From **Theorem 1** in [Own Result, 2023](./convergenceRes.md), we have the convergence of the kernel smoothing estimator.

$$
\begin{equation}
\left| \hat{\rho}^{\text{KS}}_{M, N} - \rho \right| = \mathcal{O}_\mathbb{P}(\Gamma^{-\min \{\frac{1}{2}, \frac{2}{d+2} \}})
\end{equation}
$$
