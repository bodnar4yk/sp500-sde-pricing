import yfinance as yf
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm #for ex2.ex3
import time #fot ex.3

 
# 1. Download data S&P 500
ticker = "^GSPC"
start_date="2023-01-01"
data = yf.download(ticker, start=start_date)['Close']
close_values = data.to_numpy().flatten()
dt = 1/252  # Step time - 1 work day of year

print(f"Data{close_values.shape}") #812 rows


# 2. Function evaluate params (Maximum Likelihood Estimation)
def nll_sde(params, x_t, dt):

    kappa, mu, sigma = params
    if sigma <= 0 or kappa <= 0: return 1e10
    
    # Model: x[t+1] = x[t] + kappa*(x[t] - mu)*dt + error
    expected_diff = kappa * (x_t[:-1] - mu) * dt
    actual_diff = np.diff(x_t)

    # Errors need to have distribution as N(0, sigma^2 * dt)
    residuals = actual_diff - expected_diff
    variance = (sigma**2) * dt
    
    # Negative log likelihood
    ll = -0.5 * len(residuals) * np.log(2 * np.pi * variance) - 0.5 * np.sum(residuals**2) / variance
    return -ll

# Search the Best params
initial_params = [0.1, close_values.mean(), close_values.std()]
res = minimize(nll_sde, initial_params, args=(close_values, dt), method='Nelder-Mead') 
# chose 'Nelder-Mead' because we don`t need to calculate slace. We just compare function values for each x

kappa_hat, mu_hat, sigma_hat = res.x # result params for a describe function 

print(f"Params: kappa={kappa_hat:.4f}, mu={mu_hat:.2f}, sigma={sigma_hat:.2f}")

## END: We got function with parameters that our data describes ##

##################################################
### Exercise 1 forcast for T = 1 month (21 days) ###
T_horizon = 21 * dt
x0 = float(close_values[-1])
print (x0)

# Expected E[XT]
E_XT = mu_hat + (x0 - mu_hat) * np.exp(kappa_hat * T_horizon)

# Expected Var(XT)
Var_XT = (sigma_hat**2 / (2 * kappa_hat)) * (np.exp(2 * kappa_hat * T_horizon) - 1)
Std_XT = np.sqrt(Var_XT)

print(f"Expected E[XT]: {E_XT.item():.2f}")
print(f"Expected Std[XT]: {Std_XT.item():.2f}")

##################################################
### Exercise 2 option price ###

#K=close_values[-1] # last price
K=close_values.mean()

#print(close_values[-1],K,E_XT) ## can choose the one of them

# 1. We define d as the standardized distance from the expected price to the strike.
d = (E_XT - K) / Std_XT

# 2. Calculate m
# norm.cdf(d) - це Φ(d) інтегральна функція розподілу стандартного нормального розподілу (CDF).
# norm.pdf(d) - це φ(d) щільність стандартного нормального розподілу (PDF).
m_analytical = (E_XT - K) * norm.cdf(d) + Std_XT * norm.pdf(d)

print(f"Theoretical value m: {m_analytical:.4f}")


##################################################
### Exercise 3 ###

# Params from our function
kappa, mu, sigma = res.x
x0 = float(close_values[-1])
T_horizon = 252 * dt

# # 2. Calucate m (price option) for new T
# Evaluate for new T_horizon 1 year
E_XT = mu_hat + (x0 - mu_hat) * np.exp(kappa_hat * T_horizon)
Var_XT = (sigma_hat**2 / (2 * kappa_hat)) * (np.exp(2 * kappa_hat * T_horizon) - 1)
Std_XT = np.sqrt(Var_XT)
d_test = (E_XT - K) / Std_XT
m_analytical_consistent = (E_XT - K) * norm.cdf(d_test) + Std_XT * norm.pdf(d_test)


def run_monte_carlo_fixed(n, m_true, k, m, s, x_init, strike, time_t):
    start_time = time.time()
    
    # Simulation
    epsilon = np.random.normal(0, 1, n)
    std_sim = s * np.sqrt((np.exp(2 * k * time_t)-1) / (2 * k))
    exp_sim = m + (x_init - m) * np.exp(k * time_t)
    XT = exp_sim + std_sim * epsilon
    
    m_i = np.maximum(0, XT - strike)
    m_hat = np.mean(m_i)
    
    # Standard deviation
    std_error = np.sqrt(np.sum((m_i - m_hat)**2) / (n * (n - 1)))
    
    ci = [m_hat - 1.96 * std_error, m_hat + 1.96 * std_error]
    is_inside = ci[0] <= m_true <= ci[1]
    
    print(f"n={n:7} | m_hat={m_hat:.5f} | CI=[{ci[0]:.5f}, {ci[1]:.5f}] | inside range? {is_inside}")

# Launch
print(f"Calculated m_analytical: {m_analytical_consistent:.5f}")
for n in [1000, 100000]:
    run_monte_carlo_fixed(n, m_analytical_consistent, kappa_hat, mu_hat, sigma_hat, x0, K, T_horizon)


######################################################


