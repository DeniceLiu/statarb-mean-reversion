# Statistical Arbitrage: Optimal Entry and Exit for Mean Reversion Trading

This repository implements a statistical arbitrage strategy using the **Ornstein-Uhlenbeck (OU) process** to model and trade mean-reverting spreads between asset pairs. We derive optimal entry and liquidation thresholds using both **Maximum Likelihood Estimation (MLE)** and **regression-based methods**, and evaluate strategies with **transaction costs** and **stop-loss constraints**.

---

## Objective

To develop a robust pairs trading strategy by:
- Identifying asset pairs with mean-reverting price spreads.
- Calibrating OU process parameters ($\theta$, $\mu$, $\sigma$).
- Solving optimal stopping problems with entry, exit, and stop-loss conditions.
- Backtesting model performance across training, test, and retrained datasets.

---

## Model Overview

We assume the log spread $X_t$ between two assets follows an Ornstein-Uhlenbeck (OU) process:

$dX_t = \mu(\theta - X_t)dt + \sigma dB_t$

Where:
- $\theta$: long-term mean (the spread reverts to this level)
- $\mu$: speed of mean reversion (higher implies quicker mean reversion)
- $\sigma$: volatility of the spread
- $B_t$: standard Brownian motion

The discrete-time approximation used for regression is:

$\Delta X_t = \mu(\theta - X_{t-1})\Delta t + \epsilon_t$

With:

$\epsilon_t \sim \mathcal{N}(0, \sigma^2 \Delta t)$

And regression formulation for comparison:

$X_t = \beta_0 + \beta_1 X_{t-1} + \epsilon_t$

Solving:

$\mu = -\ln(\beta_1) / \Delta t$

$\theta = \frac{\beta_0}{1 - \beta_1}$

---

## Portfolio Construction

Given assets $S_1$ and $S_2$, the portfolio spread is:

$X_t = \log S_1 - \beta \log S_2$

We fix $\alpha = 1$, and optimize $\beta$ to best fit the OU process.

---

## Parameter Estimation

We estimate the OU parameters $\theta$, $\mu$, and $\sigma$ using:

- **Ordinary Least Squares (OLS)** approximation, based on:

$\Delta X_t = \mu(\theta - X_{t-1}) \Delta t + \epsilon_t$, where 
$\epsilon_t \sim \mathcal{N}(0, \sigma^2 \Delta t)$

With regression:

$X_t = \beta_0 + \beta_1 X_{t-1} + \epsilon_t$

Then:

$\mu = -\ln(\beta_1)/\Delta t$, 

$\theta = \frac{\beta_0}{1 - \beta_1}$

---

- **Maximum Likelihood Estimation (MLE)** using the full likelihood under the OU process:

The conditional density:

$$
f^{OU}(x_i | x_{i-1}; \theta, \mu, \sigma) = \frac{1}{\sqrt{2\pi \tilde{\sigma}^2}} \exp\left( -\frac{(x_i - x_{i-1} e^{-\mu \Delta t} - \theta(1 - e^{-\mu \Delta t}))^2}{2\tilde{\sigma}^2} \right)
$$

with variance:

$$
\tilde{\sigma}^2 = \sigma^2 \frac{1 - e^{-2\mu \Delta t}}{2\mu}
$$

We maximize the average log-likelihood:

$$
\ell(\theta, \mu, \sigma \mid x_0, \dots, x_n) = -\frac{1}{2} \ln(2\pi) - \ln(\tilde{\sigma}) - \frac{1}{2\tilde{\sigma}^2} \cdot \frac{1}{n} \sum_{i=1}^{n} \left( x_i - x_{i-1} e^{-\mu \Delta t} - \theta(1 - e^{-\mu \Delta t}) \right)^2
$$

Finally, we optimize:


$\beta^* = \arg\max_{\beta} \ell(\theta^, \mu^, \sigma^* \mid x_0, x_1, ..., x_n)$


---

## Optimal Stopping Problems

We define two key stopping problems with discount rate $r$, transaction cost $c$, and stop-loss level $L$.

### 1. Optimal Liquidation (Exit)

Given a position in the spread, the optimal exit time $\tau^*$ solves:

$V(x) = \sup_{\tau} \mathbb{E} \left[ e^{-r\tau}(X_{\tau} - c) \right]$

The optimal liquidation threshold $b^*$ satisfies:

- $V(x) = x - c$ if $x \geq b^*$
- Otherwise, continue holding the position

### 2. Optimal Entry

Before entering the position, minimize expected cost:

$J(x) = \inf_{\tau} \mathbb{E} \left[ e^{-r\tau}(V(X_\tau) + c) \right]$

The optimal entry level $d^*$ satisfies:

- $J(x) = V(x) + c$ if $x \leq d^*$
- Otherwise, continue waiting

### 3. Stop-Loss Constraint

If a stop-loss is applied at level $L$, the liquidation time becomes:

$\tau_L = \inf \{ t \geq 0 : X_t \leq L \text{ or } X_t \geq b_L^* \}$

The optimal entry becomes bounded between $a_L$ and $d_L$, solved via helper functions $C(x)$ and $D(x)$.

---

## Data & Experiments

### Primary Pair: GLD–SIL (Gold vs Silver ETFs)

- Data: 2022–2024 daily log prices
- Optimal hedge ratio: ~0.36
- Entry: ~1.23 $\sigma$ below mean, Exit: ~0.37 $\sigma$ above mean
- Half-life of mean reversion: ~25 trading days

### Other Pairs:
- **AAPL–MSFT**: Weak correlation (0.13), abandoned after poor out-of-sample performance
- **KO–PEP**: High correlation (0.80), consistent spread, retrained with no stop-loss

---

## References

- Avellaneda, M., & Lee, J. H. (2010). Statistical Arbitrage in the US Equities Market.
Quantitative Finance, 10(7), p.761-782.
- Leung, T. & Li, X. (2015). Optimal Mean Reversion Trading with Transaction Costs and Stop-Loss Exit. [arXiv:1411.5062](https://arxiv.org/abs/1411.5062)
- Benth, F. E., & Karlsen, K. H. (2005). A note on Merton's portfolio selection problem for the
Schwartz mean-reversion model. Stochastic Analysis and Applications, 23(4), 687-704.
- Gatev, E., Goetzmann, W., & Rouwenhorst, K. (2006). Pairs trading: Performance of a
relative-value arbitrage rule. Review of Financial Studies, 19(3), 797-827.
- Vidyamurthy, G. (2004). Pairs Trading: Quantitative Methods and Analysis. Wiley.

