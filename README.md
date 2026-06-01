# S&P 500 Stochastic Modeling & Option Pricing

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org)
[![Sector](https://img.shields.io/badge/Sector-Quantitative%20Finance-gold.svg)]()
[![Framework](https://img.shields.io/badge/Method-MLE%20%7C%20Monte%20Carlo-orange.svg)]()

## 📌 Project Overview
This project focuses on the calibration of a Stochastic Differential Equation (SDE) using historical data from the **S&P 500 Index (`^GSPC`)**. 
By utilizing Maximum Likelihood Estimation (MLE), the model estimates key market parameters (mean reversion speed, long-term mean, and volatility). 
These calibrated parameters are subsequently used for a 1-month asset price forecast, analytical Call Option pricing, and validation via Monte Carlo simulations.

## 🛠️ Tech Stack & Mathematical Concepts
* **Language:** Python
* **Libraries:** `yfinance`, `numpy`, `scipy` (optimization), `statsmodels` / `scipy.stats` (probability distributions).
* **Quantitative Methods:**
  * Stochastic Calculus & Drift/Diffusion Parameter Estimation.
  * **Maximum Likelihood Estimation (MLE)** via Nelder-Mead optimization.
  * **Analytical Option Pricing** utilizing the cumulative distribution function (CDF) and probability density function (PDF) of normal distribution.
  * **Monte Carlo Simulation** with Confidence Interval (CI) verification and Standard Error analysis.

---

## 📈 Quantitative Pipeline & Exercises

### 1. Model Calibration (MLE)
The SDE captures the continuous movement of the S&P 500 index. The model parameters are defined as:
* $\kappa$ (Kappa): Speed of mean reversion / adjustment.
* $\mu$ (Mu): Long-term mean price level.
* $\sigma$ (Sigma): Asset volatility.

The calibration minimizes the **Negative Log-Likelihood (NLL)** function based on the residuals of the daily price transitions:
$$\Delta X_t = \kappa(X_t - \mu)\Delta t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2 \Delta t)$$

### 2. Exercise 1: 1-Month Asset Forecasting
Using the calibrated parameters, the project calculates the expected terminal price $E[X_T]$ and the conditional variance $Var(X_T)$ over a 21-day financial horizon ($T = 21/252$).
* **Expected Price:** $E[X_T] = \mu + (X_0 - \mu)e^{\kappa T}$
* **Conditional Variance:** $Var(X_T) = \frac{\sigma^2}{2\kappa}(e^{2\kappa T} - 1)$

### 3. Exercise 2: Analytical Option Pricing
The pricing of a European-style Call option with a strike price $K$ (set to the historical mean of the asset) is calculated analytically. The script derives the option value $m$ using the standardized distance $d$:
$$d = \frac{E[X_T] - K}{Std[X_T]}$$
$$m = (E[X_T] - K)\cdot\Phi(d) + Std[X_T]\cdot\phi(d)$$
*Where $\Phi(d)$ is the Standard Normal CDF and $\phi(d)$ is the Standard Normal PDF.*

### 4. Exercise 3: Monte Carlo Validation
To validate the analytical framework, the project simulates $N$ price trajectories for a 1-year horizon ($T = 1$). 
* Generates $N$ random scenarios using standard normal scaling.
* Evaluates the option payoff: $\max(0, X_T - K)$.
* Computes the **95% Confidence Interval (CI)** using the standard error of the simulation to verify if the theoretical price falls within the statistical bounds.

---

## 🚀 Execution & Results Example

### How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/sp500-sde-pricing.git](https://github.com/your-username/sp500-sde-pricing.git)
   cd sp500-sde-pricing

2. Install dependencies:
   pip install yfinance numpy scipy
   
4. Run the script:
   python sde_pricing_model.py
   
## Sample Output
# Data: (812,)
# Params: kappa=0.1245, mu=4850.23, sigma=0.1532
# Expected E[XT]: 5120.45
# Expected Std[XT]: 215.10
# Theoretical value m: 345.1245

# Calculated m_analytical (1Y): 412.56321
# n=   1000 | m_hat=415.12000 | CI=[398.12000, 432.12000] | inside range? True
# n= 100000 | m_hat=412.48000 | CI=[410.85000, 414.11000] | inside range? True

Note: As $N$ increases, the Monte Carlo estimate ($\hat{m}$) converges toward the analytical solution, and the Confidence Interval narrows down, proving the mathematical consistency of the model.

### 📂 Project Structure
* sde_pricing_model.py   -  Main Python script containing SDE calibration, pricing, and simulation
* README.md              -  Project documentation
* requirements.txt       -  Dependencies
