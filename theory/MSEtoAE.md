# General Convergence Results

## Definitions and Terminologies

### Convergence in MSE

$$
    \mathbb{E} \left[ X_M^2 \right] = \mathcal{O} \left( M^k \right)
$$

implies

$$
    \exist C \limsup_{M} \frac{\mathbb{E} \left[ X_M^2 \right]}{M^k} \leq C
$$

### Convergence in Probabilistic Order

$$
    \left| X_M \right| = \mathcal{O}_\mathbb{P} \left( M^k \right)
$$

implies that for large enough $M$,

$$
    \forall \epsilon > 0 \exist C \left( \mathbb{P} \left( \left| X_M \right| > CM^k \right) \leq \epsilon \right)
$$

## Showing $\mathcal{O}_\mathbb{P}$ from $\mathcal{O}$

Let $\mathbb{E} \left[ X_M^2 \right] = \mathcal{O} \left( M^k \right)$.

From the definition of convergence in MSE, there exists a constant $C$ such that

$$
    \limsup_{M} \frac{\mathbb{E} \left[ X_M^2 \right]}{M^k} \leq C
$$

Hence, there exists some $m$ such that for all $m \geq M$, 

$$
    \mathbb{E} \left[ X_m^2 \right] \leq C m^k
$$

The convergence can be shown by separating the expectation into two parts: tail and nontail.

$$
    \mathbb{E} \left[ X_m^2 \cdot \mathbb{I}\{|X_m| \leq dm^s\} \right] + \mathbb{E} \left[ X_m^2 \cdot \mathbb{I}\{|X_m| > dm^s\} \right]  \leq C m^k
$$

We don't have a nice bound for the first term $\mathbb{E} \left[ X_m^2 \cdot \mathbb{I}\{|X_m| \leq dm^s\} \right]$, but it is positive.

For the second term, we have

$$
\begin{align*}
    \mathbb{E} \left[ X_m^2 \cdot \mathbb{I}\{|X_m| > dm^s\} \right] 
    & \geq \mathbb{E} \left[ d^2 m^{2s} \cdot \mathbb{I}\{|X_m| > dm^s\} \right] \\
    & = d^2 m^{2s} \cdot \mathbb{E} \left[ \mathbb{I}\{|X_m| > dm^s\} \right] \\
    & = d^2 m^{2s} \cdot \mathbb{P} \left( |X_m| > dm^s\right)
\end{align*}
$$

Therefore, we are able to get a loose bound in the form of 

$$
    d^2 m^{2s} \cdot \mathbb{P} \left( |X_m| > dm^s\right) \leq C m^k
$$

Let $s = \frac{k}{2}$. Arranging the terms, we have

$$
    \mathbb{P} \left( |X_m| > dm^{\frac{k}{2}} \right) \leq \frac{C}{d^2}
$$

What's left is to plug in the terms. For all $\epsilon >0$, there exist $C^* = \sqrt{\frac{C}{\epsilon}}$ such that

$$
    \mathbb{P} \left( |X_m| > C^*m^{\frac{k}{2}} \right) \leq \epsilon
$$

Hence, this is the same as

$$
    \left| X_M \right| = \mathcal{O}_\mathbb{P} \left( M^{\frac{k}{2}} \right)
$$


## Showing $\mathcal{O}$ from $\mathcal{O}_\mathbb{P}$

Let $\left| X_M \right| = \mathcal{O}_\mathbb{P} \left( M^k \right)$.

From the definition of convergence in probabilistic order, for $M$ large enough, for all positive $\epsilon$, there exists a constant $C$ such that

$$
    \left( \mathbb{P} \left( \left| X_M \right| > CM^k \right) \leq \epsilon \right)
$$

In order to show the convergence in MSE, we separate the expectation into two parts:

$$
    \mathbb{E} \left[ X_M^2 \right] = \mathbb{E} \left[ X_M^2 \cdot \mathbb{I}\{|X_M| \leq dM^s\} \right] + \mathbb{E} \left[ X_M^2 \cdot \mathbb{I}\{|X_M| > dM^s\} \right]
$$

where the first part is bounded by $d^2 M^{2s}$:

$$ 
\begin{align*}
    \mathbb{E} \left[ X_M^2 \cdot \mathbb{I}\{|X_M| \leq dM^s\} \right] 
    & \leq \mathbb{E} \left[ d^2 M^{2s} \cdot \mathbb{I}\{|X_M| \leq dM^s\} \right] \\
    & = d^2 M^{2s} \cdot \mathbb{E} \left[ \mathbb{I}\{|X_M| \leq dM^s\} \right] \\
    & = d^2 M^{2s} \cdot \mathbb{P} \left( |X_M| \leq dM^s \right) \leq d^2 M^{2s}
\end{align*}
$$

The second part can not be bounded easily. If $X_M$ admits a density function $f_{X_M}$,

$$
\begin{align*}
    \mathbb{E} \left[ X_M^2 \cdot \mathbb{I}\{|X_M| > dM^s\} \right] 
    & = \int_{dM^s}^\infty x^2 f_{X_M}(x) dx + \int_{-\infty}^{-dM^s} x^2 f_{X_M}(x) dx
\end{align*}
$$

Intuitively, in the tail parts of $X_M$, if the square of $X_M$ is larger than the density function evaluated at $X_M$, or equivalently, $X_M^{-2}$ grows faster than $f_{X_M}$, both integrals will not converge to a finite number.

Hence, we are able to concluded that convergence in MSE is stronger than convergence in probabilistic order.

## Theorem 1

Let $X_M$ be a sequence of random variables. If $\mathbb{E} \left[ X_M^2 \right] = \mathcal{O} \left( M^k \right)$, then $\left| X_M \right| = \mathcal{O}_\mathbb{P} \left( M^k \right)$.

