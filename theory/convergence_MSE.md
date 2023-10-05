# Theoretical Framework

In a nested estimation problem, we are interested in the quantity 

$$\rho(g(X)) = \mathbb{E} \left[ h(g(X)) \right]$$

where $X \in \Omega$. 
$g(X)$ can't be directly evaluated, but it is the output of 

$$ g(X) = \mathbb{E}\left[ Y|X=x \right]\vert_{x=X} $$

In supervised learning, $g(\cdot)$ can be approximated by $\hat{g}^{\text{SL}}_{M, N}(\cdot)$. which is based on a chosen function family $\mathcal{G}$ and observations from the standard nested simulation procedure.

### The SNS Procedure 

The standard nested simulation procedure first simulate $M$ iid outer scenarios $X_1, \dots, X_M$ from $F_X$, the distribution of $X$.
For each $X_i$, again simulate $Y_{ij}$, $j = 1, \dots, N$ from $F_{Y|X_i}$, the conditional distribution of $Y$ given $X_i$. Given scenario $i$, the $Y_{ij}$ are conditionally iid. The total simulation budget $\Gamma = M \cdot N$. Let $f_X(x)$ denote the density of $X$.

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
$$\hat{\rho}^{\text{SL}, \text{Test}}_{M, N, M'} = \frac{1}{M} \sum_{i=1}^M h(\hat{g}^{\text{SL}}_{M, N}(\tilde{X}_i)); ~~~ \tilde{X}_i \sim F_X$$
2. VaR:
$$\hat{\rho}^{\text{SL}, \text{Test}}_{M, N, M'} = (\hat{g}^{\text{SL}}_{M, N}(\tilde{\mathbf{X}}))_{\lceil \alpha M \rceil}$$
3. CVaR:
$$\hat{\rho}^{\text{SL}, \text{Test}}_{M, N, M'} = (\hat{g}^{\text{SL}}_{M, N}(\tilde{\mathbf{X}}))_{\lceil \alpha M \rceil} + \frac{1}{(1-\alpha) M} \sum_{i=1}^M \max \{\hat{g}^{\text{SL}}_{M, N}(\tilde{X}_i) - (\hat{g}^{\text{SL}}_{M, N}(\tilde{\mathbf{X}}))_{\lceil \alpha M \rceil}, 0 \}$$

where $(\hat{g}_{M, N}(\tilde{\mathbf{X}}))_{\lceil \alpha M \rceil}$ is the $\lceil \alpha M \rceil$-th order statistic of $\hat{g}^{\text{SL}}_{M, N}(\tilde{X}_1), \dots, \hat{g}_{M, N}(\tilde{X}_M)$. 

Note that
$\hat{g}^{\text{SL}}_{M, N}(\cdot)$ is derived from the training samples $(X_1, \hat{g}_N(X_1)), \dots, (X_M, \hat{g}_N(X_M))$.

We are interested in minimizing the mean squared error (MSE) of the nested Monte Carlo estimator with supervised learning $\hat{\rho}^{\text{SL}, \text{Train}}_{M, N}$ and $\hat{\rho}^{\text{SL}, \text{Test}}_{M, N, M'}$ subject to the total simulation budget $\Gamma$

$$ 
\begin{aligned}
& \min_{M, N}  & \text{MSE}(\hat{\rho}^{\text{SL}}_{M, N}) = \mathbb{E} \left[ \left( \hat{\rho}^{\text{SL}}_{M, N} - \rho \right)^2 \right] \\
& \text{subject to} & M \cdot N = \Gamma 
\end{aligned}
$$

### Existing Literature 

* **Standard Nested Simulation**

    $$\hat{g}^{\text{SNS}}_{M, N}(X_i) = \hat{g}_N(X_i) = \bar{Y}_{N, i} $$


* **Regression**: 
    $$\hat{g}^{\text{REG}}_{M, N}(X) = \Phi(X) \hat{\beta}$$
    where $\Phi$ is the chosen basis, and $\hat{\beta}$ is estimated from the training samples: $(X_1, \hat{g}_N(X_1)), \dots, (X_M, \hat{g}_N(X_M))$.

* **Kernel Smoothing**:
    $$\hat{g}^{\text{KS}}_{M}(X) = \frac{\sum_{i=1}^M Y_i K_h(X - X_i)}{\sum_{i=1}^M K_h(X - X_i)}$$
    where $K_h$ is the kernel function with bandwidth $h$.

    Note that the kernel smoothing estimator is not a function of $N$. 
    Instead, it is interested in observation pairs $(X_i, \phi(X_i, Y_i))$ for $i \in \{1, \dots, M\}$.

* **Kernel Ridge Regression**:
    $$\hat{g}^{\text{KRR}}_{M, N}(X) = \argmin_{g \in \mathcal{N}_{\Psi}(\Omega)} \left( \frac{1}{M} \sum_{i=1}^M (\hat{g}_N(X_i) - g(X_i))^2 + \lambda \|g\|_{\mathcal{N}_{\Psi}(\Omega)}^2\right)$$
    where $\mathcal{N}_{\Psi}(\Omega)$ is the reproducing kernel Hilbert space (RKHS) with kernel $\Psi$ defined on the domain of $X$, and $\lambda$ is the regularization parameter as in ridge regression. 
    More specifically, $\Phi$ is a Matérn kernel with smoothness parameter $\nu$ and length scale parameter $\ell$.

# Asymptotic Analysis - Nested Expectation Form

In this section, we conduct asymptotic analysis of $\hat{\rho}^{\text{SL}, \text{Train}}_{M, N}$ the nested Monte Carlo estimator with supervised learning and training data for the first case, i.e., nested expectation form, where 

$$\hat{\rho}^{\text{SL}, \text{Train}}_{M, N} = \frac{1}{M} \sum_{i=1}^M h(\hat{g}^{\text{SL}}_{M, N}(X_i)); ~~~ X_i \sim F_X$$

# Decomposition 1

**Assumption 1**: $h(g(X))$ has finite second moment, i.e., $\mathbb{E} \left[ \left( h(g(X)) \right)^2 \right] < \infty$.

The MSE of the nested Monte Carlo estimator $\hat{\rho}^{\text{SL}}_{M, N}$ can be decomposed into two terms

Let $\rho_M = \frac{1}{M} \sum_{i=1}^M h(g(X_i))$ be the nested Monte Carlo estimator with the true function $g$.

$$
\begin{align}
\mathbb{E} \left[ \left( \hat{\rho}^{\text{SL}}_{M, N} - \rho \right)^2 \right] 
& \leq 2 \mathbb{E} \left[ \left( \hat{\rho}^{\text{SL}}_{M, N} - \rho_M \right)^2 \right] 
        + 2  \mathbb{E} \left[ \left(\rho_M - \rho \right)^2 \right]  \nonumber \\
& = 2 \mathbb{E} \left[  \left( \frac{1}{M} \sum_{i=1}^M h\left( \hat{g}^{\text{SL}}_{M, N}(X_i) \right) -  \frac{1}{M} \sum_{i=1}^M h\left(g(X_i) \right)  \right)^2\right] + 2  \mathbb{E} \left[ \left(\frac{1}{M} \sum_{i=1}^M h\left(g(X_i) \right) - \mathbb{E}\left[ h(g(X))\right] \right)^2 \right]  \nonumber \\
& = 2 \mathbb{E} \left[  \left( \frac{1}{M} \sum_{i=1}^M h\left( \hat{g}^{\text{SL}}_{M, N}(X_i) \right) -  h\left(g(X_i) \right)  \right)^2\right] + \frac{2}{M} \text{Var}(h(g(X))) \nonumber \\
& = 2 \mathbb{E} \left[  \left( \frac{1}{M} \sum_{i=1}^M h\left( \hat{g}^{\text{SL}}_{M, N}(X_i) \right) -  h\left(g(X_i) \right)  \right)^2\right] + \mathcal{O}(M^{-1})
\end{align}
$$

The analysis of the first term is different for different forms of $h$. We will analyze them separately.

## Smooth $h$

**Assumption 1.SF**: The function $h$ has first and second order derivatives.

### Taylor Expansion

Taylor expansion of $h$ around $g(X_i)$ gives

$$
\begin{align}
& \mathbb{E} \left[  \left( \frac{1}{M} \sum_{i=1}^M h\left( \hat{g}^{\text{SL}}_{M, N}(X_i) \right) -  h\left(g(X_i) \right)  \right)^2\right] \nonumber \\
& = \mathbb{E} \left[ \left( \frac{1}{M} \sum_{i=1}^M h'\left( g(X_i) \right) \left( \hat{g}^{\text{SL}}_{M, N}(X_i) - g(X_i) \right) +  \frac{1}{2M} \sum_{i=1}^M h''\left( z_i \right) \left( \hat{g}^{\text{SL}}_{M, N}(X_i) - g(X_i) \right)^2 \right)^2\right] \nonumber \\
& \leq 2 \mathbb{E} \left[ \left( \frac{1}{M} \sum_{i=1}^M h'\left( g(X_i) \right) \left( \hat{g}^{\text{SL}}_{M, N}(X_i) - g(X_i) \right) \right)^2\right] + 2 \mathbb{E} \left[ \left( \frac{1}{2M} \sum_{i=1}^M h''\left( z_i \right) \left( \hat{g}^{\text{SL}}_{M, N}(X_i) - g(X_i) \right)^2 \right)^2\right]  \\
\end{align}
$$

where the last inequality is due to $2ab \leq a^2 + b^2$ for any $a, b \in \mathbb{R}$. For different methods of nested estimation, each of the two terms on the right hand side of **(2)** can be analyzed separately.

### Standard Monte Carlo Estimator: $\hat{g}_{M, N}^{\text{SL}}(X) = \hat{g}_N(X)$

**Assumption 1.SF.SNS.1**: $h$ has bounded first and second order derivative, i.e., there exists $C_1 > 0$ such that $|h'(x)| \leq C_1$ for all $x \in \mathbb{R}$, and there exists $C_2 > 0$ such that $|h''(x)| \leq C_2$ for all $x \in \mathbb{R}$.

**Assumption 1.SF.SNS.2**: $\hat{g}_N(X) = g(X) + \bar{Z}_N(X)$, where the simulation noise $\bar{Z}_N(X)$ has zero mean and variance $\nu(X) / N$, where the conditional variance $\nu(X)$ is bounded, i.e., there exists $C_\nu > 0$ such that $\nu(x) \leq C_\nu$ for all $x \in \mathbb{R}$. 

**Assumption 1.SF.SNS.3**: The fourth moment of simulation noise $\bar{Z}_N(X)$ follows $\mathbb{E} \left[ \left( \bar{Z}_N(X) \right)^4 \right] = \nu_2(X) / N^2$, where $\nu_2(X)$ is bounded, i.e., there exists $C_{\nu,2} > 0$ such that $\nu_2(X) \leq C_{\nu,2}$ for all $x \in \mathbb{R}$.

$$
\begin{align}
\mathbb{E} \left[ \left(\frac{1}{M} \sum_{i=1}^M h'\left( g(X_i) \right) \left( \hat{g}_{N}(X_i) - g(X_i) \right) \right)^2 \right]
& = \mathbb{E} \left[ \left(\frac{1}{M} \sum_{i=1}^M h'\left( g(X_i) \right) \bar{Z}_N(X_i) \right)^2 \right] \nonumber \\
& \leq C_1^2 \mathbb{E} \left[ \left(\frac{1}{M} \sum_{i=1}^M \bar{Z}_N(X_i) \right)^2 \right] \nonumber \\
& = C_1^2 \mathbb{E} \left[ \frac{1}{M^2} \sum_{i=1}^M \sum_{j=1}^M \bar{Z}_N(X_i) \bar{Z}_N(X_j) \right] \nonumber \\
& = C_1^2 \mathbb{E} \left[ \frac{1}{M^2} \sum_{i=1}^M \bar{Z}_N^2(X_i) + \frac{1}{M^2} \sum_{i=1}^M \sum_{j \neq i}^M \bar{Z}_N(X_i) \bar{Z}_N(X_j) \right] \nonumber \\
& \leq  \frac{C_1^2 C_{\nu, 1}}{MN}  = \mathcal{O}(M^{-1} N^{-1})
\end{align}
$$

where the last inequality is due to **Assumption SF.SNS.2**, independence of $X_i$ and $X_j$ for $i \neq j$, and the fact that $\mathbb{E} \left[ \bar{Z}_N(X) \right] = 0$.

$$
\begin{align}
\mathbb{E} \left[ \left( \frac{1}{2M} \sum_{i=1}^M h''\left( z_i \right) \left( \hat{g}^{\text{SL}}_{M, N}(X_i) - g(X_i) \right)^2 \right)^2\right] 
& = \mathbb{E} \left[ \left(\frac{1}{2M} \sum_{i=1}^M h''\left( z_i \right) \bar{Z}^2_N(X_i) \right)^2 \right] \nonumber \\
& \leq C_2^2 \mathbb{E} \left[ \left(\frac{1}{2M} \sum_{i=1}^M \bar{Z}^2_N(X_i) \right)^2 \right] \nonumber \\
& = C_2^2 \mathbb{E} \left[ \frac{1}{2M^2} \sum_{i=1}^M \sum_{j=1}^M \bar{Z}^2_N(X_i) \bar{Z}^2_N(X_j) \right] \nonumber \\
& = C_2^2 \mathbb{E} \left[ \frac{1}{2M^2} \sum_{i=1}^M \bar{Z}_N^4(X_i) + \frac{1}{2M^2} \sum_{i=1}^M \sum_{j \neq i}^M \bar{Z}_N^2(X_i) \bar{Z}_N^2(X_j) \right] \nonumber \\
& \leq  C_2^2 \left(\frac{ C_{\nu, 2} M}{2M^2N^2} + \frac{C_{\nu,1}^2M(M-1)}{2M^2N^2}\right) = \mathcal{O}(N^{-2})
\end{align}
$$
    
where the second inequality is due to **Assumption SF.SNS.1**, and the last equality is due $\hat{g}_{N}(X)$ is a standard Monte Carlo estimator of $g(X)$.

Combining **(1)**, **(2)**, **(3)**, and **(4)**, we have

$$
\begin{equation}
\mathbb{E} \left[ \left( \hat{\rho}^{\text{SNS}}_{M, N} - \rho \right)^2 \right] = \mathcal{O}(M^{-1}) + \mathcal{O}(N^{-2})
\end{equation}
$$

### Regression Estimator: $\hat{g}_{M, N}^{\text{SL}}(X) = \hat{g}_{M,N}^{\text{REG}}(X; \hat{\beta}) = \Phi(\mathbf{X}) \hat{\beta}$

Let $k$ be the number of predictors, and $\beta^*$ be the optimal solution of the following optimization problem

$$
\begin{equation}
\beta^* \in \argmin_{\beta \in \mathbb{R}^k} \mathbb{E} \left[ \left( \Phi(\mathbf{X}) \beta - g(X) \right)^2 \right]
\end{equation}
$$

and let $\hat{\beta}$ be the solution of the following optimization problem

$$
\begin{equation}
\hat{\beta} \in \argmin_{\beta \in \mathbb{R}^k} \frac{1}{M} \sum_{i=1}^M \left( \Phi(\mathbf{X}) \beta - \hat{g}_N(X_i) \right)^2
\end{equation}
$$

Then, the first term in **(2)** can be decomposed as
$$
\begin{align}
\mathbb{E} \left[ \left( \frac{1}{M} \sum_{i=1}^M h'\left( g(X_i) \right) \left( \hat{g}^{\text{REG}}_{M, N}(X_i) - g(X_i) \right) \right)^2\right] 
& = \mathbb{E} \left[ \left( \frac{1}{M} \sum_{i=1}^M h'\left( g(X_i) \right) \left( \left(\Phi(\mathbf{X}) \hat{\beta}\right)_i - g(X_i) \right) \right)^2\right] \nonumber \\
& = \mathbb{E} \left[ \left( \frac{1}{M} \sum_{i=1}^M h'\left( g(X_i) \right) \left( \left(\Phi(\mathbf{X}) \hat{\beta}\right)_i - \left(\Phi(\mathbf{X}) \beta^*\right)_i + \left(\Phi(\mathbf{X}) \beta^*\right)_i - g(X_i) \right) \right)^2\right] \nonumber \\
& \leq 2 \mathbb{E} \left[ \left( \frac{1}{M} \sum_{i=1}^M h'\left( g(X_i) \right) \left( \left(\Phi(\mathbf{X}) \hat{\beta}\right)_i - \left(\Phi(\mathbf{X}) \beta^*\right)_i  \right) \right)^2\right] + 2 \mathbb{E} \left[ \left( \frac{1}{M} \sum_{i=1}^M h'\left( g(X_i) \right) \left( \left(\Phi(\mathbf{X}) \beta^*\right)_i  - g(X_i) \right) \right)^2\right]  
\end{align}
$$

The first term can further be decomposed as

$$
\begin{align}
\mathbb{E} \left[ \left( \frac{1}{M} \sum_{i=1}^M h'\left( g(X_i) \right) \left( \left(\Phi(\mathbf{X}) \hat{\beta}\right)_i - \left(\Phi(\mathbf{X}) \beta^*\right)_i  \right) \right)^2\right]
& = \mathbb{E} \left[ \left( \frac{1}{M} \sum_{i=1}^M h'\left( g(X_i) \right) \Phi(\mathbf{X}) \left( \hat{\beta} - \beta^*  \right)_i \right)^2\right] \nonumber \\
\end{align}
$$

No further progress can be made at the moment.

### Kernel Smoothing Estimator: $\hat{g}^{\text{KS}}_{M}(X) = \frac{\sum_{i=1}^M Y_i K_h(X - X_i)}{\sum_{i=1}^M K_h(X - X_i)}$

Similarly, found no theory to proceed from **(8)**.

### KRR Estimator: $\hat{g}^{\text{KRR}}_{M, N}(X)$

**Assumption 1.SF.KRR.1**: $h$ has bounded first and second order derivative, i.e., there exists $C_1 > 0$ such that $|h'(x)| \leq C_1$ for all $x \in \mathbb{R}$, and there exists $C_2 > 0$ such that $|h''(x)| \leq C_2$ for all $x \in \mathbb{R}$.

**Assumption 1.SF.KRR.2**. The data generating process is $y_{ij} = g(X_i) + \epsilon_{ij}$ and $\epsilon_i \sim \text{subG}(0, \sigma^2)$.

**Comment**: **Assumption 1.SF.KRR.2** is a stronger assumption than **Assumption 1.SF.SNS.2**. It has made assumptions on the tail behavior of $\epsilon_{ij}$ and this affect $\sum_{j=1}^N \epsilon_{ij}$, the $\bar{Z}_N$ in the standard Monte Carlo case.

In probability theory, a sub-Gaussian distribution is a probability distribution with strong tail decay. Informally, the tails of a sub-Gaussian distribution are dominated by (i.e. decay at least as fast as) the tails of a Gaussian. 

Link to wikipedia: https://en.wikipedia.org/wiki/Sub-Gaussian_distribution

**Assumption 1.SF.KRR.3**. $h \in \mathcal{H}_{K_\nu}(\Omega)$, where $\mathcal{H}_{K_\nu}(\Omega)$ is the reproducing kernel Hilbert space (RKHS) with the Matérn kernel $K$ of smoothness parameter $\nu$ and domain $\Omega$.



The KRR estimator of $g(X)$ is defined as

$$\hat{g}^{\text{KRR}}_{M, N}(X) = \argmin_{g \in \mathcal{N}_{\Psi}(\Omega)} \left( \frac{1}{M} \sum_{i=1}^M (\hat{g}_N(X_i) - g(X_i))^2 + \lambda \|g\|_{\mathcal{H}_{\Psi}(\Omega)}^2\right)$$

By discussion from the last meeting,
$$
\begin{equation}
\left| \frac{1}{M} \sum_{i=1}^M h'\left( g(X_i) \right) \left( \hat{g}^{\text{KRR}}_{M, N}(X_i) - g(X_i) \right) \right| 
= \mathcal{O}_\mathbb{P} \left(\lambda^{1/2} + (MN)^{-1/2} \right) 
\end{equation}
$$

$$
\begin{equation}
\left| \frac{1}{M} \sum_{i=1}^M h''\left( z_i \right) \left( \hat{g}^{\text{KRR}}_{M, N}(X_i) - g(X_i) \right)^2  \right| 
= \mathcal{O}_\mathbb{P} \left(\lambda + (MN)^{-1} \lambda^{-\frac{d}{2\nu+d}} + (MN)^{-\frac{2\nu + d}{2\nu + 2d}} \right)
\end{equation}
$$

Hence, the MSE of the KRR estimator is

$$
\begin{equation}
\mathbb{E} \left[ \left( \hat{g}^{\text{KRR}}_{M, N}(X) - g(X) \right)^2 \right]
= \mathcal{O} \left(M^{-1} + \lambda + (MN)^{-2} \lambda^{-\frac{2d}{2\nu+d}} \right)
\end{equation}
$$

Most KRR research establish bounds on the squared $\mathcal{L}_2$ error of the KRR estimator:

$$
\begin{equation}
\frac{1}{M} \sum_{i=1}^M \left( \hat{g}^{\text{KRR}}_{M, N}(X_i) - g(X_i) \right)^2
\end{equation}
$$

The data generating process is $y_{ij} = g(X_i) + \epsilon_{ij}$.

The result in [Wang, 2022](papers/Wang_2022.pdf) is derived using the Matérn kernel with smoothness $\nu$ and sub-Gaussian noise terms, other result on the convergence of polynomial kernel and Gaussian kernel for $\epsilon_i \sim \mathcal{N}(0, \sigma)$ can be found in **Section 2** of [Yang, 2015](papers/Yang_2015.pdf).

# Decomposition 2


$$
\begin{align}
\mathbb{E} \left[ \left( \hat{\rho}^{\text{SL}}_{M, N} - \rho \right)^2 \right] 
& = \mathbb{E} \left[ \left( \hat{\rho}^{\text{SL}}_{M, N} - \mathbb{E} \left[ \hat{\rho}^{\text{SL}}_{M, N} \right] \right)^2 \right] + \left( \mathbb{E} \left[ \hat{\rho}^{\text{SL}}_{M, N} \right] - \rho \right)^2 \nonumber \\
& = \text{Var}(\hat{\rho}^{\text{SL}}_{M, N}) + \text{Bias}(\hat{\rho}^{\text{SL}}_{M, N})^2
\end{align}
$$

where $\text{Var}(\hat{\rho}^{\text{SL}}_{M, N})$ is the variance of the estimator and $\text{Bias}(\hat{\rho}^{\text{SL}}_{M, N})$ is the bias of the estimator about $\rho$. We will analyze them separately.

## Analysis of Bias

$$
\begin{align}
\text{Bias}(\hat{\rho}^{\text{SL}}_{M, N})
& = \mathbb{E} \left[ \hat{\rho}^{\text{SL}}_{M, N} \right] - \rho \nonumber \\
& = \mathbb{E} \left[ \frac{1}{M} \sum_{i=1}^M h\left( \hat{g}^{\text{SL}}_{M, N}(X_i) \right) \right] - \mathbb{E} \left[ h\left( g(X) \right) \right]  \nonumber \\
& = \mathbb{E} \left[ h\left( \hat{g}^{\text{SL}}_{M, N}(X) \right) - h\left( g(X) \right) \right] \\
\end{align}
$$

## Analysis of Variance

$$
\begin{align}
\text{Var}(\hat{\rho}^{\text{SL}}_{M, N}) 
& = \mathbb{E} \left[ \left( \hat{\rho}^{\text{SL}}_{M, N} - \mathbb{E} \left[ \hat{\rho}^{\text{SL}}_{M, N} \right] \right)^2 \right] \nonumber \\
& = \mathbb{E} \left[ \left( \frac{1}{M} \sum_{i=1}^M h\left( \hat{g}^{\text{SL}}_{M, N}(X_i) \right)- \mathbb{E} \left[ \frac{1}{M} \sum_{i=1}^M h\left(\hat{g}^{\text{SL}}_{M, N}(X_i) \right) \right] \right)^2  \right] \nonumber \\
& = \mathbb{E} \left[ \left( \frac{1}{M} \sum_{i=1}^M h\left( \hat{g}^{\text{SL}}_{M, N}(X_i) \right)- \mathbb{E} \left[ h\left(\hat{g}^{\text{SL}}_{M, N}(X) \right) \right] \right)^2  \right] 
\end{align}
$$

where the third equality is due to the i.i.d. assumption of $X_i$.

However, the quantities $h(\hat{g}_{M,N}^{SL}(X_i))$ are not necessarily independent. We will analyze the variance of the estimator for different forms of $h$ separately.

If they are, then
$$
\text{Var}(\hat{\rho}^{\text{SL}}_{M, N}) = \frac{1}{M} \text{Var}(h\left(\hat{g}^{\text{SL}}_{M, N}(X) \right)) 
$$

## Smooth $h$ - Bias

**Assumption 2.B.SF.1**: The function $h$ has first and second order derivatives.

### Taylor Expansion

From **(4)**, Taylor expansion of $h$ around $g(X)$ gives

$$
\begin{align}
\text{Bias}(\hat{\rho}^{\text{SL}}_{M, N})
& = \mathbb{E} \left[ h\left( \hat{g}^{\text{SL}}_{M, N}(X) \right) - h\left( g(X) \right) \right]  \nonumber \\
& = \mathbb{E} \left[h'\left( g(X) \right) \left( \hat{g}^{\text{SL}}_{M, N}(X) - g(X) \right) + \frac{1}{2} h''\left( z \right) \left( \hat{g}^{\text{SL}}_{M, N}(X) - g(X) \right)^2 \right]  
\end{align}
$$

where the last inequality is due to $2ab \leq a^2 + b^2$ for any $a, b \in \mathbb{R}$. Each of the two terms on the right hand side of **(5)** can be analyzed separately.

**Assumption 2.B.SF.2**: $h$ has bounded second order derivative, i.e., there exists $C_{f,2} > 0$ such that $|h''(x)| \leq C$ for all $x \in \mathbb{R}$.

### Standard Monte Carlo Estimator: $\hat{g}_{M, N}^{\text{SL}}(X) = \hat{g}_N(X)$

$$
\begin{align}
\mathbb{E} \left[ h'\left( g(X) \right) \left( \hat{g}^{\text{SL}}_{M, N}(X) - g(X) \right) \right]
& = \mathbb{E} \left[ h'\left( g(X) \right) \left( \hat{g}_N(X) - g(X) \right) \right] \nonumber \\
& = \mathbb{E} \left[ \mathbb{E} \left[ h'\left( g(X) \right) \left( \hat{g}_N(X) - g(X) \right)  | X \right] \right] \nonumber \\
& = \mathbb{E} \left[ h'\left( g(X) \right) \mathbb{E} \left[ \hat{g}_N(X) - g(X)  | X \right] \right] = 0
\end{align}
$$

where the third equality is due to the unbiasedness of $\hat{g}_N(X)$ about $g(X)$.

$$
\begin{align}
\mathbb{E} \left[ h''\left( z \right) \left( \hat{g}^{\text{SL}}_{M, N}(X) - g(X) \right)^2 \right]
& \leq \mathbb{E} \left[ |h''\left( z \right)| \left( \hat{g}_{N}(X) - g(X) \right)^2 \right] \nonumber \\
& \leq C_{f,2} \mathbb{E} \left[ \left( \hat{g}_{N}(X) - g(X) \right)^2 \right] = C_{f,2} \text{Var}(\hat{g}_{N}(X)) = \mathcal{O}\left( N^{-1} \right)
\end{align}
$$
    
where the second inequality is due to **Assumption 2.SF.1**, and the last equality is due to $\hat{g}_{N}(X)$ being a standard Monte Carlo estimator of $g(X)$.

### Regression Estimator: $\hat{g}_{M, N}^{\text{SL}}(X) = \hat{g}_{M,N}^{\text{REG}}(X; \hat{\beta}) = \Phi(\mathbf{X}) \hat{\beta}$

**Assumption 2.B.SF.REG.1**: The variance $\mathbb{E} \left[ \text{Var}(g(X)) \right] < \infty$.

**Assumption 2.B.SF.REG.2**: The design matrix $\Phi(\mathbf{X})$ is full rank.   

**Assumption 2.B.SF.REG.3**: The function $h$ has bounded second derivative, i.e., there exists $C_{f,2} > 0$ such that $|h''(x)| \leq C$ for all $x \in \mathbb{R}$.

Consider the regression estimator $\hat{g}^{\text{SL}}_{M, N}(X) = \hat{g}^{\text{REG}}_{M, N}(X; \hat{\beta})$ where $\hat{\beta}$ is the solution of the following optimization problem

$$
\begin{equation}
\hat{\beta} \in \argmin_{\beta \in \mathbb{R}^k} \frac{1}{M} \sum_{i=1}^M \left( \Phi(\mathbf{X}) \beta - \hat{g}_N(X_i) \right)^2
\end{equation}
$$

Let:
* $\bar{Z}_N(X) = \hat{g}_N(X) - g(X)$ be the error from the standard inner simulation.
* $\mathcal{M}(X) = g(X) - \Phi(\mathbf{X}) \beta^*$ be the model error of the regression estimator

where $\beta^*$ is the best regression parameter. Then, the bias of the regression estimator can be written as


$$
\begin{align}
\mathbb{E} \left[ h'\left( g(X) \right) \left( \hat{g}^{\text{SL}}_{M, N}(X) - g(X) \right) \right]
& = \mathbb{E} \left[ h'\left( g(X) \right) \left( \hat{g}^{\text{REG}}_{M, N}(X; \hat{\beta}) - g(X) \right) \right] \nonumber \\
& = \mathbb{E} \left[ h'\left( g(X) \right) \left( \Phi(\mathbf{X}) \hat{\beta} - \Phi(\mathbf{X}) \beta^* + \Phi(\mathbf{X}) \beta^* - g(X) \right) \right] \nonumber \\
& = \mathbb{E} \left[ h'\left( g(X) \right) \left( \Phi(\mathbf{X}) \hat{\beta} - \Phi(\mathbf{X}) \beta^* \right) \right] + \mathbb{E} \left[ h'\left( g(X) \right) \left( \Phi(\mathbf{X}) \beta^* - g(X) \right) \right] \nonumber \\
& = \mathbb{E} \left[ h'\left( g(X) \right) \left( \Phi(\mathbf{X})  (\hat{\beta} - \beta^*) \right) \right] - \mathbb{E} \left[ h'\left( g(X) \right) (\mathcal{M}(X)) \right] 
\end{align}
$$

From regression theory, we know that as $M \rightarrow \infty$,
$$
\begin{equation}
\text{Cov} \left( \frac{1}{\sqrt{M}} \Phi^\top(\mathbf{X}) \left( \mathcal{M}(\mathbf{X}) + \bar{Z}_N(\mathbf{X}) \right) \right)^{-\frac{1}{2}} \sqrt{M} \left(\hat{\beta} - \beta \right) \overset{\mathcal{D}}{\to} \mathcal{N}(0, \mathbf{I})
\end{equation}
$$
where $\bar{Z}_N(\mathbf{X}) = \left(\bar{Z}_N(X_1), \dots, \bar{Z}_N(X_M) \right)^\top$, $\mathcal{M}(\mathbf{X}) = \left(\mathcal{M}(X_1), \dots, \mathcal{M}(X_M) \right)^\top$ is the vectors of the simulation error terms and the model error terms, respectively.

From direct calculation, the $\left(\hat{\beta} - \beta \right)$ term correspond to the middle term inside the variance.
$$
\begin{equation}
\sqrt{M} \cdot \mathbb{E} \left[ h'\left( g(X) \right) \left( \Phi(\mathbf{X}) \right) \right] (\hat{\beta} - \beta^*) \overset{\mathcal{D}}{\to} \mathcal{N} \left( 0, \mathbb{E} \left[ h'\left( g(X) \right) \left( \Phi(\mathbf{X}) \right) \right] \left( \Sigma_\mathcal{M} + \frac{\Sigma_V}{N} \right) \left( \mathbb{E} \left[ h'\left( g(X) \right) \left( \Phi(\mathbf{X}) \right) \right]\right)^\top \right)
\end{equation}
$$

where $\Sigma_\mathcal{M} = \mathbb{E} \left[ \mathcal{M}(X)^2 \Phi(X)^\top \Phi(X)\right]$ and $\Sigma_V = \mathbb{E} \left[ \text{Var}(g_N(X)|X) \Phi(X)^\top \Phi(X) \right]$.

* Note: $\mathcal{M}(X)$ and $\text{Var}(g_N(X)|X)$ are 1-dimensional.

$$
\begin{equation}
\sqrt{M}\cdot \left( \mathbb{E}\left[h(\hat{g}_{M,N}^{\text{REG}}(X; \hat{\beta})) - h\left( g(X) \right) \right] - \mathcal{B}_{\mathcal{M}, M} \right) \overset{\mathcal{D}}{\to} \mathcal{N} \left( 0, \mathbb{E} \left[ h'\left( g(X) \right) \left( \Phi(\mathbf{X}) \right) \right] \left( \Sigma_\mathcal{M} + \frac{\Sigma_V}{N} \right) \left( \mathbb{E} \left[ h'\left( g(X) \right) \left( \Phi(\mathbf{X}) \right) \right]\right)^\top \right)
\end{equation}
$$

Combine the remaining error term $\mathbb{E} \left[ h'\left( g(X) \right) (\mathcal{M}(X)) \right]$ with the second order Taylor expansion of $h$ at $g(X)$ into the regression bias $\mathcal{B}_{\mathcal{M}, M}$, i.e.,

$$
B_{\mathcal{M},M} = 
\mathbb{E} \left[ h'(g(X)) \mathcal{M}(X) \right]
- \mathbb{E} \left[ \frac{1}{2} h''(z)(\Phi(X)\hat{\beta} - g(X))^2 \right]
$$

Note from **(22)**,

$$
\hat{\rho}_{M,N}^{\text{REG}} - \rho - \mathcal{B}_{\mathcal{M},M} \overset{\mathbb{P}}{\to} 0
$$

Combined with the fact that 

$$
\hat{\rho}_{M,N}^{\text{REG}} \overset{\mathbb{P}}{\to} \mathbb{E} \left[ h(\Phi(X)) \beta^*\right]
$$

We are able to characterize the bias of the regression estimator $\mathcal{B}_{\mathcal{M}}^*$ via

$$
\mathcal{B}_{\mathcal{M},M}  \overset{\mathbb{P}}{\to} \mathcal{B}_{\mathcal{M}}^* = \mathbb{E} \left[ h(\Phi(X)) \beta^*\right] - \rho
$$

From **Assumption 2.B.SF.REG.3**, we can show the regression bias $\mathcal{B}_{\mathcal{M}}$ follows
$$
\begin{equation}
\left| \mathcal{B}_{\mathcal{M},} - \mathbb{E} \left[ h'\left( g(X) \right) (\mathcal{M}(X)) \right] \right| \leq \frac{C_{f,2}}{2} \mathbb{E} \left[ \mathcal{M}^2(X) \right]
\end{equation}
$$

### Kernel Smoothing Estimator: $\hat{g}^{\text{SL}}_{M, N}(X) = \hat{g}^{\text{KS}}_{M}(X)$

**Assumption 2.B.SF.KS.1**: The variance $\mathbb{E} \left[ g^2(X) \right] < \infty$.

**Assumption 2.B.SF.KS.2**: $w \to 0$, $Mw^d \to \infty$, and $Mw^{d+4} \to 0$ as $n \to \infty$.

**Assumption 2.SF.KS.3**: $h$ has bounded first and second order derivatives

For kernel smoothing estimator, we have

$$
\begin{align}
\mathbb{E} \left[ h'\left( g(X) \right) \left( \hat{g}^{\text{KS}}_{M, N}(X) - g(X) \right) \right]
& = \mathbb{E} \left[ h'\left( g(X) \right) \mathbb{E} \left[ \hat{g}^{\text{KS}}_{M, N}(X) - g(X)  |X \right] \right] \nonumber \\
& = \int_{\Omega}  h'\left( g(x) \right) \mathcal{B}_M(x) f_X(x) dx \nonumber \\
& = \int_{\Omega}  h'\left( g(x) \right) \mathcal{B}(x) w^2 \mathcal{O}_x(1) f_X(x) dx \nonumber \\
& = \mathcal{O}(w^2) \mathbb{E} \left[ h'\left( g(X)  \right) \mathcal{B}(x)  \right] 
\end{align}
$$

Similarly
$$
\begin{align}
\mathbb{E} \left[ h''\left( g(X) \right) \left( \hat{g}^{\text{KS}}_{M, N}(X) - g(X) \right)^2 \right]
& = \mathbb{E} \left[ h''\left( g(X) \right) \mathbb{E} \left[ (\hat{g}^{\text{KS}}_{M, N}(X) - g(X))^2  |X \right] \right] \nonumber \\
& = \int_{\Omega}  h'\left( g(x) \right) \left(\mathcal{B}_M^2(x) + \mathcal{V}_M(x)\right) f_X(x) dx \nonumber \\
& = \mathcal{O}(w^4) \mathbb{E} \left[ h''\left( g(X)  \right) \mathcal{B}^2(x)  \right] + \mathcal{O}(M^{-1} w^{-d}) \mathbb{E} \left[ h''\left( g(X)  \right) \mathcal{V}(x)  \right] \nonumber \\
\end{align}
$$

where $\mathcal{B}_M(x) = \mathbb{E} \left[ \hat{g}^{\text{KS}}_{M, N}(X) - g(X)  |X=x\right]$ and $\mathcal{V}_M(x) = \text{Var} \left[ \hat{g}^{\text{KS}}_{M, N}(X) - g(X)  |X=x \right]$, and 

$$
\begin{align}
\mathcal{B}_M(x) & = \mathcal{B}(x) \mathcal{O}(w^2) \nonumber \\
\mathcal{V}_M(x) & = \mathcal{V}(x) \mathcal{O}(M^{-1} w^{-d}) \nonumber \\
\end{align}
$$

where $\mathcal{B}(x)$ and $\mathcal{V}(x)$ does not depend on $M$.

Therefore, we have

$$
\begin{align}
\text{Bias}(\hat{\rho}^{\text{KS}}_{M, N}) = \mathcal{O}(w^2) + \mathcal{O}(M^{-1}w^{-d})
\end{align}
$$

### KRR Estimator

Most KRR research establish bounds on the squared $\mathcal{L}_2$ error of the KRR estimator:

$$
\frac{1}{M} \sum_{i=1}^M \left( \hat{g}^{\text{KRR}}_{M, N}(X_i) - g(X_i) \right)^2
$$

This makes it difficult to apply to our setting, where we want to bound

$$
|\hat{g}^{\text{KRR}}_{M, N}(X) - g(X)|
$$



## Smooth $h$ - Variance

### Standard Monte Carlo Estimator

For smooth function $h$, the variance cannot be reduced without making assumptions on the second moment of $h(\hat{g}_{N}(X))$.

Applying **Assumption V.1**, we have 

$$ 
\begin{equation}
\text{Var}(\hat{\rho}^{\text{SL}}_{M, N}) = \frac{1}{M} \text{Var}(h\left(\hat{g}_{N}(X) \right)) = \mathcal{O}(M^{-1})
\end{equation}
$$

### Regression Estimator

$$ 
\begin{equation}
\text{Var}(\hat{\rho}^{\text{SL}}_{M, N}) = \frac{1}{M} \text{Var}(h\left(\hat{g}^{\text{REG}}_{M, N}(X) | \mathbf{X}, \mathbf{g}_N(\mathbf{X}) \right)) = \mathcal{O}(M^{-1})
\end{equation}
$$

In [Broadie, 2015](papers/Broadie_2015.pdf), the proof is done with the training data $\mathbf{X}, \mathbf{g}_N(\mathbf{X})$ given

### Kernel Smoothing Estimator

**Assumption 2.V.SF.KS.1**: $\hat{g}^{\text{KS}}_M(X)$ converge to $g(X)$ in probability as $M \to \infty$.

**Assumption 2.V.SF.KS.2**: The discontinuity of $h$, $\mathcal{D}_h$, satisfies $\mathbb{P}(\{g(X) \in \mathcal{D}_h\}) = 0$, and there exist a constant $C_h$ and an integer $p \geq 0$ such that $|h(t)| \leq C_h|t|^p$ for all $t \in \mathcal{D}_h$.

**Assumption 2.V.SF.KS.3**: There exist some $\delta > 0$ such that 
$$\sup_M \mathbb{E} \left[ |\hat{g}^{\text{KS}}_M(X) | ^{2p+\delta}\right]$$

**Assumption 2.V.SF.KS.4**: There exist some $C_f$ such that $f_X(x) \leq C_f$ for all $x \in \Omega$.

**Assumption 2.V.SF.KS.5**: $\sup_M \mathbb{E} \left[ Z^4_M(X) \right] < \infty$

[Hong, 2017](papers/Hong_2017.pdf) is the only paper to find a bound on the variance of the estimator. 

$$
\begin{align}
\text{Var}(\hat{\rho}^{\text{SL}}_{M, N}) 
& = \frac{1}{M} \text{Var}\left( h\left(\hat{g}^{\text{KS}}_M(X) \right)\right) + \left( 1 - \frac{1}{M} \right) \text{Cov} \left( \hat{g}^{\text{KS}}_M(X_1), \hat{g}^{\text{KS}}_M(X_2)\right)  
\end{align}
$$

where each term in **(28)** is bounded separately.

$$
\begin{align}
\text{Var}\left( h\left(\hat{g}^{\text{KS}}_M(X) \right)\right) 
& = \text{Var}\left( h\left(\hat{g}^{\text{KS}}_M(X) \right)\right)
- \text{Var} \left( h\left(g(X) \right)\right) + \text{Var} \left( h\left(g(X) \right)\right) \nonumber \\
& = \text{Var} \left( h\left(g(X) \right)\right) 
+ \mathbb{E} \left[ h^2\left( \hat{g}^{\text{KS}}_M(X) \right) \right] - \mathbb{E}^2 \left[ h\left( \hat{g}^{\text{KS}}_M(X) \right) \right] 
+ \mathbb{E}^2 \left[ h\left( g(X) \right) \right] - \mathbb{E}^2 \left[ h\left( g(X) \right) \right] \nonumber \\
& = \text{Var} \left( h\left(g(X) \right)\right) + \mathcal{o}_M(1)
\end{align}
$$

where **Assumption 2.V.SF.KS.1**, **Assumption 2.V.SF.KS.2**, **Assumption 2.V.SF.KS.3** ensures the convergence of $\hat{g}^{\text{KS}}_M(X)$, $h(\hat{g}^{\text{KS}}_M(X))$ and $\mathbb{E}\left[h^p\left(\hat{g}^{\text{KS}}_M(X) \right)\right]$.

It remains to bound the covariance term $\text{Cov} \left( \hat{g}^{\text{KS}}_M(X_1), \hat{g}^{\text{KS}}_M(X_2)\right)$.

We first show that the covariance term is nonnegative.

Since $X_1$ and $X_2$ are independent, when the training data $\mathbf{X}, \mathbf{g}_N(\mathbf{X})$ is given, the covariance term is zero.

$$
\begin{align}
\text{Cov} \left( \hat{g}^{\text{KS}}_M(X_1), \hat{g}^{\text{KS}}_M(X_2)\right)
& = \text{Cov} \left( \mathbb{E}\left[\hat{g}^{\text{KS}}_M(X_1) | \mathbf{X}, \mathbf{g}_N(\mathbf{X}) \right],  \mathbb{E}\left[\hat{g}^{\text{KS}}_M(X_2) | \mathbf{X}, \mathbf{g}_N(\mathbf{X}) \right] \right) + \mathbb{E} \left[ \text{Cov} \left( \hat{g}^{\text{KS}}_M(X_1), \hat{g}^{\text{KS}}_M(X_2) | \mathbf{X}, \mathbf{g}_N(\mathbf{X}) \right) \right] \nonumber \\
& = \text{Var} \left( \mathbb{E}\left[\hat{g}^{\text{KS}}_M(X) | \mathbf{X}, \mathbf{g}_N(\mathbf{X}) \right] \right) \geq 0
\end{align}
$$

where the last inequality is due to the fact that the variance is always non-negative.

Then, it remains to bound the covariance. From **Assumption 2.V.SF.KS.4**, we have

$$
\begin{align}
M \text{Cov} \left( \hat{g}^{\text{KS}}_M(X_1), \hat{g}^{\text{KS}}_M(X_2)\right) 
& = M \int_\Omega\int_\Omega \text{Cov}(h(\hat{g}_M(x_1), \hat{g}_M(x_2))) f(x_1) f(x_2) \mathbb{I}_{\{x_2 \in H(x_1)\}} dx_1 dx_2 \nonumber \\
& \leq \frac{M}{2} \int_\Omega\int_\Omega \left( \text{Var}(h(\hat{g}_M(x_1))) + \text{Var}(h(\hat{g}_M(x_2))) \right) f(x_1) f(x_2) \mathbb{I}_{\{x_2 \in H(x_1)\}} dx_1 dx_2  \nonumber \\
& = M \int_\Omega \int_\Omega \text{Var}(h(\hat{g}_M(x_1))) f(x_1) f(x_2) \mathbb{I}_{\{x_2 \in H(x_1)\}} dx_1 dx_2 \nonumber \\
& = M \int_\Omega \text{Var}(h(\hat{g}_M(x_1))) f(x_1) \int_\Omega f(x_2) \mathbb{I}_{\{x_2 \in H(x_1)\}} dx_2 dx_1 \nonumber \\
& \leq M \cdot C_f w^d  \int_\Omega \text{Var}(h(\hat{g}_M(x))) f(x) dx \nonumber \\ 
& \leq M \cdot C_f w^d  \int_\Omega \mathbb{E} \left[ \left( h(\hat{g}_M(x)) - h(g(x)) \right)^2 \right] f(x) dx \nonumber \\
& = M \cdot C_f w^d  \int_\Omega \mathbb{E} \left[ \left( h'(g(x))\left( h(\hat{g}_M(x)) - h(g(x)) \right) + \frac{1}{2} h''(z)\left( h(\hat{g}_M(x)) - h(g(x)) \right)^2 \right)^2 \right] f(x) dx \nonumber \\
& \leq 2 M \cdot C_f w^d \int_\Omega \left( \mathbb{E} \left[ h'(g(x))^2 \left( h(\hat{g}_M(x)) - h(g(x)) \right)^2  \right] + \frac{C_{g, 2}}{4} \mathbb{E} \left[ \left( h(\hat{g}_M(x)) - h(g(x)) \right)^4 \right] \right) f(x) dx \nonumber \\ 
& = 2 C_f \left( \mathbb{E} \left[ h'(g(x))^2 Z_M^2(X)\right] + \frac{C_{g, 2}^2 \mathbb{E} \left[ Z_M^4(X)\right]}{2Mw^d} \right) \nonumber \\
& \leq 2 C_f \left( \sqrt{\mathbb{E} \left[ (h'(g(X)))^4 \right] \mathbb{E} \left[ (Z_M(X))^4 \right] } \right) + \frac{C_{g, 2}^2 \mathbb{E} \left[ Z_M^4(X)\right]}{2Mw^d}  
\end{align}
$$

where $Z_M(X) = \sqrt{Mw^d} \left( \hat{g}_M^{\text{KS}}(X) - g(X) \right)$, and the last inequality is due to the Hölder's inequality.


### KRR Estimator

Similar to the above, applying **Assumption V.1**, we have

$$
\begin{equation}
\text{Var}(\hat{\rho}^{\text{SL}}_{M, N}) = \frac{1}{M} \text{Var}(h\left(\hat{g}^{\text{KRR}}_{M, N}(X) \right)) = \mathcal{O}(M^{-1})
\end{equation}
$$


## Neural Network

No result has been found for neural networks except for robustness to additive noise, see [Lee, 1994](./papers/Lee_1994.pdf).


## Classification of Critical Assumptions

Nested Simulation Procedures | Assumptions on Simulation Noise | Risk Measures | Convergence of $\hat{g}_{M,N}^{\text{SL}}(X)$ | Convgergence of MSE of $\hat{\rho}_{M,N}$ |
| --- | --- | --- | --- | --- |
| Standard Monte Carlo |  **Assumption 1.SF.SNS.2**: $\bar{Z}_N(X)$ has zero mean and variance decay in $N$; <br />**Assumption 1.SF.SNS.3**: The fourth moment of $\bar{Z}_N(X)$ decays in $N^{-2}$ | **Assumption 1.SF.SNS.1**: $h$ has bounded first and second order derivatives | N/A | $\mathcal{O}(M^{-1}) + \mathcal{O}(N^{-2})$ |
| Regression | **Assumption 2.B.SF.REG.1**: The variance $\text{Var}(g(X)) < \infty$. | **Assumption 2.B.SF.REG.3**: $h$ has bounded second derivative | **Assumption 2.B.SF.REG.2**: The design matrix $\Phi(\mathbf{X})$ is full rank.  | $\mathcal{O}(M^{-1})$ |
| Kernel Smoothing | **Assumption 2.B.SF.KS.1**: The variance $\mathbb{E} \left[ g^2(X) \right] < \infty$  <br /> **Assumption 2.V.SF.KS.2**: The discontinuity of $h$, $\mathcal{D}_h$, satisfies $\mathbb{P}(\{g(X) \in \mathcal{D}_h\}) = 0$ <br /> **Assumption 2.V.SF.KS.5**: $\sup_M \mathbb{E} \left[ Z^4_M(X) \right] < \infty$ <br /> **Assumption 2.V.SF.KS.4**: There exist some $C_f$ such that the pdf of $X$, $f_X(x) \leq C_f$ for all $x \in \Omega$. | **Assumption 2.SF.KS.3**: $h$ has bounded first and second order derivatives | **Assumption 2.B.SF.KS.2**: $w \to 0$, $Mw^d \to \infty$, and $Mw^{d+4} \to 0$ as $M \to \infty$. <br /> **Assumption 2.V.SF.KS.1**: $\hat{g}^{\text{KS}}_M(X)$ converge to $g(X)$ in probability as $M \to \infty$. | $\mathcal{O}(M^{-\min\{1, \frac{4}{d+2}\}})$  |
| Kernel Ridge Regression | **Assumption 1.SF.KRR.1**: sub-Gaussian noise terms <br /> **Assumption 1.SF.KRR.2**: $\Omega$ is bounded and convex |  **Assumption 1.SF.KRR.3**: $h$ has bounded first and second order derivatives | $g$ is in the RKHS associated with the Matern kernel | N/A |