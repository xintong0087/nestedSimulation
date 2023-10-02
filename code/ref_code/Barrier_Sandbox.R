rm(list = ls())
library(mvtnorm)

##### Black-Scholes model parameters and option parameters
S0 <- 100       # initial asset price
mu <- 0.05      # annual expected return (real world drift)
rf <- 0.02      # annual risk-free rate (risk neutral drift)
vol <- 0.3      # annual volatility
T <- 2          # time-to-maturity (at time zero)
tau <- 4 / 52     # risk horizon
h <- 1 / 52       # time step size

K_barrier <- c(80, 90, 100, 110, 120, 130)      # barrier option strikes
H_barrier <- c(70, 80, 90, 100, 110, 120)       # lower barriers
U_barrier <- c(Inf, Inf, Inf, Inf, Inf, Inf, Inf)   # upper barriers
type_barrier <- c('pdo', 'pdo', 'pdo', 'pdo', 'pdo', 'pdo')
# pdo: down-and-out put
# cdo: down-and-out call
# pdi: down-and-in put
# cdi: down-and-in call
# puo: up-and-out put
# cuo: up-and-out call
# pui: up-and-in put
# cui: up-and-in call
position_barrier <- c(1, 1, 1, 1, 1, 1)
num_barrier <- length(K_barrier)

##### Derived parameters
T_inner <- T - tau  # time to maturity (at time tau)
num_steps_outer <- round(tau / h)      # number of steps in outer loop
num_steps_inner <- round(T_inner / h)  # number of steps in inner loop
num_steps_total <- num_steps_outer + num_steps_inner

meanlog_outer <- (mu - vol^2 / 2) * h   # per-step normal mean in outer loops
meanlog_inner <- (rf - vol^2 / 2) * h   # per-step normal mean in inner loops
sdlog_outer <- vol * sqrt(h)  # per-step normal sd in outer loops
sdlog_inner <- vol * sqrt(h)  # per-step normal sd in inner loops

#####  outer simulation (real world projection from time 0 to tau)
num_outer <- 1e2 # number of outer scenarios
num_inner <- 1e3 # number of inner replications (N0 for ESS calculation)


BarrierOptionSim <- function(S_full_paths, T, r, K, H = -Inf, U = Inf, option_type, position){
  # S_full_paths: a matrix of stock price paths
  # T: time to maturity
  # r: risk-free rate
  # K: strike price
  # option_type:
  # pdo: down-and-out put
  # cdo: down-and-out call
  # pdi: down-and-in put
  # cdi: down-and-in call
  # puo: up-and-out put
  # cuo: up-and-out call
  # pui: up-and-in put
  # cui: up-and-in call
  # position: position of the option, positive for long, negative for short
  
  num_steps <- nrow(S_full_paths)
  num_paths <- ncol(S_full_paths)
  
  # S_T: stock price at maturity
  S_T <- S_full_paths[num_steps, ]
  
  # check if letter d is in each of option_type
  if (grepl("d", option_type)) {
    # the k-th barrier option requires a lower barrier
    # minimum price of the underlying asset
    s_min <- matrix(0, nrow = num_steps - 1, ncol = num_paths)
    for (s in 2:num_steps) {
      b1 <- log(S_full_paths[s, ])
      b2 <- log(S_full_paths[s - 1, ])
      U <- runif(num_paths)
      s_min[s - 1, ] <- exp(((b1 + b2) - sqrt((b1-b2)^2 - 2*vol^2*h*log(U)))/2)
    }
    s_min <- apply(s_min, 2, min)
  } else if (grepl("u", option_type)) {
    # the k-th barrier option requires an upper barrier
    # maximum price of the underlying asset
    s_max <- matrix(0, nrow = num_steps - 1, ncol = num_paths)
    for (s in 2:num_steps) {
      b1 <- log(S_full_paths[s, ])
      b2 <- log(S_full_paths[s - 1, ])
      U <- runif(num_paths)
      s_max[s - 1, ] <- exp(((b1 + b2) + sqrt((b1-b2)^2 - 2*vol^2*h*log(U)))/2)
    }
    s_max <- apply(s_max, 2, max)
  }
  
  if (option_type == 'pdo') {
    # down-and-out put
    payoff <- pmax(K - S_T , 0) * (s_min > H) * position
  } else if (option_type == 'cdo') {
    # down-and-out call
    payoff <- pmax(S_T - K, 0) * (s_min > H) * position
  } else if (option_type == 'pdi') {
    # down-and-in put
    payoff <- pmax(K - S_T, 0) * (s_min < H) * position
  } else if (option_type == 'cdi') {
    # down-and-in call
    payoff <- pmax(S_T - K, 0) * (s_min < H) * position
  } else if (option_type == 'puo') {
    # up-and-out put
    payoff <- pmax(K - S_T, 0) * (s_max < U) * position
  } else if (option_type == 'cuo') {
    # up-and-out call
    payoff <- pmax(S_T - K, 0) * (s_max < U) * position
  } else if (option_type == 'pui') {
    # up-and-in put
    payoff <- pmax(K - S_T, 0) * (s_max > U) * position
  } else if (option_type == 'cui') {
    # up-and-in call
    payoff <- pmax(S_T - K, 0) * (s_max > U) * position
  } else {
    # give a warning message that the type should be either "pdo", "cdo", "pdi", "cdi", "puo", "cuo", "pui", or "cui"
    warning("The type of the option should be either 'pdo', 'cdo', 'pdi', 'cdi', 'puo', 'cuo', 'pui', or 'cui'.")
  }
  
  return(exp(-r * T) * payoff)
}

PTSingleAssetBarrierOption =
    function(TypeFlag = c("cdoA", "cuoA", "pdoA", "puoA", "coB1", "poB1",
             "cdoB2", "cuoB2"), S, X, H, time1, Time2, r, b, sigma)
{   # A function implemented by Diethelm Wuertz

    # Description:
    #   Partial-time single asset barrier options

    # References:
    #   Haug, Chapter 2.10.3

    # FUNCTION:

    # Compute:
    TypeFlag = TypeFlag[1]
    PartialTimeBarrier = NA
    t1 = time1
    T2 = Time2
    if (TypeFlag == "cdoA") eta = 1
    if (TypeFlag == "cuoA") eta = -1

    # Continue:
    d1 = (log(S/X) + (b + sigma^2/2) * T2) / (sigma * sqrt(T2))
    d2 = d1 - sigma * sqrt (T2)
    f1 = (log(S/X) + 2 * log(H/S) + (b + sigma^2/2) * T2) / (sigma * sqrt(T2))
    f2 = f1 - sigma * sqrt (T2)
    e1 = (log(S / H) + (b + sigma ^ 2 / 2) * t1) / (sigma * sqrt(t1))
    e2 = e1 - sigma * sqrt (t1)
    e3 = e1 + 2 * log (H / S) / (sigma * sqrt(t1))
    e4 = e3 - sigma * sqrt (t1)
    mu = (b - sigma ^ 2 / 2) / sigma ^ 2
    rho = sqrt (t1 / T2)
    g1 = (log(S / H) + (b + sigma ^ 2 / 2) * T2) / (sigma * sqrt(T2))
    g2 = g1 - sigma * sqrt (T2)
    g3 = g1 + 2 * log (H / S) / (sigma * sqrt(T2))
    g4 = g3 - sigma * sqrt (T2)
    z1 = CND(e2) - (H / S) ^ (2 * mu) * CND(e4)
    z2 = CND(-e2) - (H / S) ^ (2 * mu) * CND(-e4)
    z3 = CBND (g2, e2, rho) - (H / S) ^ (2 * mu) * CBND(g4, -e4, -rho)
    z4 = CBND (-g2, -e2, rho) - (H / S) ^ (2 * mu) * CBND(-g4, e4, -rho)
    z5 = CND (e1) - (H / S) ^ (2 * (mu + 1)) * CND(e3)
    z6 = CND (-e1) - (H / S) ^ (2 * (mu + 1)) * CND(-e3)
    z7 = CBND (g1, e1, rho) - (H / S) ^ (2 * (mu + 1)) * CBND(g3, -e3, -rho)
    z8 = CBND (-g1, -e1, rho) - (H / S) ^ (2 * (mu + 1)) * CBND(-g3, e3, -rho)

    if (TypeFlag == "cdoA" || TypeFlag == "cuoA") {
        # call down-and out and up-and-out type A
        PartialTimeBarrier =
            (S * exp ((b - r) * T2) * (CBND(d1, eta * e1, eta * rho) -
             (H / S) ^ (2 * (mu + 1)) * CBND(f1, eta * e3, eta * rho))
             - X * exp (-r * T2) * (CBND(d2, eta * e2, eta * rho) - (H
             / S) ^ (2 * mu) * CBND(f2, eta * e4, eta * rho))) }

    if (TypeFlag == "cdoB2" && X < H) {
        # call down-and-out type B2
        PartialTimeBarrier =
            (S * exp ((b - r) * T2) * (CBND(g1, e1, rho) - (H / S) ^
            (2 * (mu + 1)) * CBND(g3, -e3, -rho)) - X * exp (-r * T2)
            * (CBND(g2, e2, rho) - (H / S) ^ (2 * mu) * CBND(g4, -e4,
                                                             -rho))) }

    if (TypeFlag == "cdoB2" && X > H) {
        PartialTimeBarrier = (PTSingleAssetBarrierOption("coB1", S, X, H, t1,
                                                         T2, r, b, sigma)) }

    if (TypeFlag == "cuoB2" && X < H) {
        # call up-and-out type B2
        PartialTimeBarrier =
            (S * exp ((b - r) * T2) * (CBND(-g1, -e1, rho) - (H / S) ^
            (2 * (mu + 1)) * CBND(-g3, e3, -rho)) - X * exp (-r * T2)
            * (CBND(-g2, -e2, rho) - (H / S) ^ (2 * mu) * CBND(-g4,
            e4, -rho)) - S * exp ((b - r) * T2) * (CBND(-d1, -e1, rho)
            - (H / S) ^ (2 * (mu + 1)) * CBND(e3, -f1, -rho)) + X *
            exp (-r * T2) * (CBND(-d2, -e2, rho) - (H / S) ^ (2 * mu)
            * CBND(e4, -f2, -rho)))}

    if (TypeFlag == "coB1" && X > H) {
        # call out type B1
        PartialTimeBarrier =
            (S * exp ((b - r) * T2) * (CBND(d1, e1, rho) - (H / S) ^
            (2 * (mu + 1)) * CBND(f1, -e3, -rho)) - X * exp (-r * T2)
            * (CBND(d2, e2, rho) - (H / S) ^ (2 * mu) * CBND(f2, -e4,
            -rho))) }

    if (TypeFlag == "coB1" && X < H) {
        PartialTimeBarrier =
            (S * exp ((b - r) * T2) * (CBND(-g1, -e1, rho) - (H / S) ^
            (2 * (mu + 1)) * CBND(-g3, e3, -rho)) - X * exp (-r * T2)
            * (CBND(-g2, -e2, rho) - (H / S) ^ (2 * mu) * CBND(-g4,
            e4, -rho)) - S * exp ((b - r) * T2) * (CBND(-d1, -e1, rho)
            - (H / S) ^ (2 * (mu + 1)) * CBND(-f1, e3, -rho)) + X *
            exp (-r * T2) * (CBND(-d2, -e2, rho) - (H / S) ^ (2 * mu)
            * CBND(-f2, e4, -rho)) + S * exp ((b - r) * T2) *
            (CBND(g1, e1, rho) - (H / S) ^ (2 * (mu + 1)) * CBND(g3,
            -e3, -rho)) - X * exp (-r * T2) * (CBND(g2, e2, rho) - (H
            / S) ^ (2 * mu) * CBND(g4, -e4, -rho))) }

    if (TypeFlag == "pdoA") {
        # put down-and out and up-and-out type A
        PartialTimeBarrier = (PTSingleAssetBarrierOption("cdoA",
                              S, X, H, t1, T2, r, b, sigma) -
                              S * exp((b - r) * T2) * z5 + X * exp(-r * T2) * z1)}

    if (TypeFlag == "puoA") {
        PartialTimeBarrier = (PTSingleAssetBarrierOption("cuoA",
                              S, X, H, t1, T2, r, b, sigma) -
                              S * exp((b - r) * T2) * z6 + X * exp(-r * T2) * z2) }

    if (TypeFlag == "poB1") {
        # put out type B1
        PartialTimeBarrier = (PTSingleAssetBarrierOption("coB1",
                              S, X, H, t1, T2, r, b, sigma) -
                              S * exp((b - r) * T2) * z8 + X * exp(-r * T2) * z4 -
                              S * exp((b - r) * T2) * z7 + X * exp(-r * T2) * z3) }

    if (TypeFlag == "pdoB2") {
        # put down-and-out type B2
        PartialTimeBarrier = (PTSingleAssetBarrierOption("cdoB2",
                              S, X, H, t1, T2, r, b, sigma) -
                              S * exp((b - r) * T2) * z7 + X * exp(-r * T2) * z3) }

    if (TypeFlag == "puoB2") {
        # put up-and-out type B2
        PartialTimeBarrier = (PTSingleAssetBarrierOption("cuoB2",
                              S, X, H, t1, T2, r, b, sigma) -
                              S * exp((b - r) * T2) * z8 + X * exp(-r * T2) * z4) }

    # Return Value:
    return(PartialTimeBarrier)
}

NDF =
function(x)
{   # A function implemented by Diethelm Wuertz

    # Description:
    #   Calculate the normal distribution function.

    # FUNCTION:

    # Compute:
    result = exp(-x*x/2)/sqrt(8*atan(1))

    # Return Value:
    result
}

CND =
function(x)
{   # A function implemented by Diethelm Wuertz

    # Description:
    #   Calculate the cumulated normal distribution function.

    # References:
    #   Haug E.G., The Complete Guide to Option Pricing Formulas

    # FUNCTION:

    # Compute:
    k  = 1 / ( 1 + 0.2316419 * abs(x) )
    a1 =  0.319381530; a2 = -0.356563782; a3 = 1.781477937
    a4 = -1.821255978; a5 =  1.330274429
    result = NDF(x) * (a1*k + a2*k^2 + a3*k^3 + a4*k^4 + a5*k^5) - 0.5
    result = 0.5 - result*sign(x)

    # Return Value:
    result
}

CBND =
function(x1, x2, rho)
{   # A function implemented by Diethelm Wuertz

    # Description:
    #   Calculate the cumulative bivariate normal distribution function.

    # References:
    #   Haug E.G., The Complete Guide to Option Pricing Formulas

    # FUNCTION:

    # Compute:
    # Take care for the limit rho = +/- 1
    a = x1
    b = x2
    if (abs(rho) == 1) rho = rho - (1e-12)*sign(rho)
    # cat("\n a - b - rho :"); print(c(a,b,rho))
    X = c(0.24840615, 0.39233107, 0.21141819, 0.03324666, 0.00082485334)
    y = c(0.10024215, 0.48281397, 1.0609498, 1.7797294, 2.6697604)
    a1 = a / sqrt(2 * (1 - rho^2))
    b1 = b / sqrt(2 * (1 - rho^2))
    if (a <= 0 && b <= 0 && rho <= 0) {
       Sum1 = 0
       for (I in 1:5) {
            for (j in 1:5) {
            Sum1 = Sum1 + X[I] * X[j] *
              exp(a1*(2*y[I]-a1) + b1*(2*y[j]-b1) +
              2*rho*(y[I]-a1)*(y[j]-b1)) } }
       result = sqrt(1 - rho^2) / pi * Sum1
       return(result) }
    if (a <= 0 && b >= 0 && rho >= 0) {
        result = CND(a) - CBND(a, -b, -rho)
        return(result) }
    if (a >= 0 && b <= 0 && rho >= 0) {
        result = CND(b) - CBND(-a, b, -rho)
        return(result) }
    if (a >= 0 && b >= 0 && rho <= 0) {
        result = CND(a) + CND(b) - 1 + CBND(-a, -b, rho)
        return(result) }
    if (a * b * rho >= 0 ) {
        rho1 = (rho*a - b) * sign(a) / sqrt(a^2 - 2*rho*a*b + b^2)
        rho2 = (rho*b - a) * sign(b) / sqrt(a^2 - 2*rho*a*b + b^2)
        delta = (1 - sign(a) * sign(b)) / 4
        result = CBND(a, 0, rho1) + CBND(b, 0, rho2) - delta
        return(result) }

    # Return Value:
    invisible()
}

##---- deterministic outer scenarios
qts <- (1:num_outer)/(num_outer+1) # deterministic quantiles
S_tau <- S0*exp(meanlog_outer * num_steps_outer + qnorm(qts)*sdlog_outer*sqrt(num_steps_outer))

##### closed-form portfolio value
val_true <- rep(0, num_outer)
val_nested <- rep(0, num_outer)
for(i in 1:num_outer){
  print(sprintf("True Value: This is the %d-th outer loop.", i))
  S_tau_i <- S_tau[i]
  
  portfolio_value <- 0  # placeholder for portfolio value

  for (k in 1:num_barrier) {
    if(position_barrier[k] != 0){
      # true price of k-th barrier option
      barrier_price_k <- PTSingleAssetBarrierOption("pdoB2", S_tau_i, K_barrier[k], H_barrier[k], h, T, rf, 0, vol)
      portfolio_value <- portfolio_value + barrier_price_k
    }
  }
  val_true[i] <- portfolio_value
  #==========================
  # growth factor in inner simulation
  z_inner_full <- rnorm(num_steps_inner * num_inner, mean = meanlog_inner, sdlog_inner)
  z_inner_full <- matrix(z_inner_full, nrow = num_steps_inner)
  z_inner_full <- apply(z_inner_full, 2, cumsum)
  
  S_inner_full <- S_tau_i * exp(z_inner_full)   # full inner price paths
  
  portfolio_value <- 0  # placeholder for portfolio value
  
  for (k in 1:num_barrier) {
    if (position_barrier[k] != 0){
      # simulated price of k-th Asian option
      price_k <- BarrierOptionSim(S_full_paths = S_inner_full,
                                  T = T_inner, r = rf, K = K_barrier[k], H = H_barrier[k], U = U_barrier[k], option_type = type_barrier[k], position = position_barrier[k])
      portfolio_value <- portfolio_value + mean(price_k)
    }
  }
  
  val_nested[i] <- portfolio_value
}

plot(S_tau, val_true, xlab = "S_tau", ylab = "Portfolio Value", main = "True Value")
plot(val_true, val_nested, xlab = "True Value", ylab = "Standard Nested Sim", main = "QQ-plot for validation")