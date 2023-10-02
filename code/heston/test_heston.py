import numpy as np
import matplotlib.pyplot as plt

# Heston model parameters
r = 0.05           # Risk-free rate
kappa = 2.0        # Mean-reversion speed
theta = 0.04       # Long-term variance
xi = 0.1           # Volatility of volatility
rho = -0.7         # Correlation between asset price and variance

# Initial values
S0 = 100           # Initial asset price
V0 = 0.04          # Initial variance

# Time parameters
T = 1              # Total simulation time
dt = 0.001         # Time step
N = int(T/dt)      # Number of time steps

# Arrays to store simulations
S = np.zeros(N)
V = np.zeros(N)
S[0] = S0
V[0] = V0

# Simulation of the Heston model
for i in range(1, N):
    dW1 = np.sqrt(dt) * np.random.randn()
    dW2 = np.sqrt(dt) * np.random.randn()

    # Creating the correlated Wiener process
    dW2_corr = rho * dW1 + np.sqrt(1 - rho**2) * dW2

    # Update variance
    V[i] = V[i-1] + kappa*(theta - V[i-1])*dt + xi*np.sqrt(V[i-1])*dW2_corr
    V[i] = max(V[i], 0)  # Ensure non-negativity

    # Update asset price
    S[i] = S[i-1] + r*S[i-1]*dt + np.sqrt(V[i-1])*S[i-1]*dW1

# Plotting
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(np.linspace(0, T, N), S)
plt.title('Heston Model Simulation')
plt.ylabel('Asset Price')

plt.subplot(2, 1, 2)
plt.plot(np.linspace(0, T, N), V)
plt.ylabel('Variance')
plt.xlabel('Time')

plt.tight_layout()
plt.show()