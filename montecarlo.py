import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

# Function to get data
def get_data(etf_list, start, end):
    data = yf.download(etf_list, start=start, end=end)['Close']
    returns = data.pct_change()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()*2
    return mean_returns, cov_matrix

# List of stocks
stock_list = ['ACWD.L', 'DFNS.L', 'IITU.L', 'WLDS.L', 'AAPL', 'LMT', 'NATO.L', 'JEDI.L']

# Define the date range
end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=365*10)

# Get the mean returns and covariance matrix
meanReturns, covMatrix = get_data(stock_list, start_date, end_date)

weights = [0.6, 0.07, 0.1, 0.1, 0.01, 0.02, 0.05, 0.05]

# Monte Carlo method
mc_sims = 100
time_range = 365*5

meanMatrix = np.full(shape=(time_range, len(weights)), fill_value=meanReturns).T
portfolio_sims = np.full(shape=(time_range, mc_sims), fill_value=0.0)

initial_value = 5000

for m in range(0, mc_sims):
    Z = np.random.normal(size=(time_range, len(weights)))
    L = np.linalg.cholesky(covMatrix)
    dailyReturns = meanMatrix + np.inner(L, Z)
    portfolio_sims[:, m] = np.cumprod(np.inner(weights, dailyReturns.T) + 1) * initial_value

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(portfolio_sims)
plt.title('Monte Carlo Simulation of Portfolio Returns')
plt.xlabel('Days')
plt.ylabel('Portfolio Value')
plt.show()
plt.savefig('output.png')

# Calculate the mean and percentiles for the simulations
mean_simulation = portfolio_sims.mean(axis=1)
percentile_5th = np.percentile(portfolio_sims, 5, axis=1)
percentile_95th = np.percentile(portfolio_sims, 95, axis=1)

# Get the final values for the three cases
final_mean_value = mean_simulation[-1]
final_5th_percentile_value = percentile_5th[-1]
final_95th_percentile_value = percentile_95th[-1]

plt.figure(figsize=(10, 6))
plt.plot(portfolio_sims, color='gray', alpha=0.1)
plt.plot(mean_simulation, color='blue', label='Mean Simulation')
plt.fill_between(range(time_range), percentile_5th, percentile_95th, color='blue', alpha=0.2)
plt.title('Monte Carlo Simulation of Portfolio Returns')
plt.xlabel('Days')
plt.ylabel('Portfolio Value')
plt.legend()
# Annotate the final values on the plot
plt.annotate(f'${final_mean_value:.2f}', xy=(time_range-1, final_mean_value), xytext=(time_range-1+10, final_mean_value),
             arrowprops=dict(facecolor='blue', shrink=0.05), fontsize=12)
plt.annotate(f'${final_5th_percentile_value:.2f}', xy=(time_range-1, final_5th_percentile_value), xytext=(time_range-1+10, final_5th_percentile_value),
             arrowprops=dict(facecolor='blue', shrink=0.05), fontsize=12)
plt.annotate(f'${final_95th_percentile_value:.2f}', xy=(time_range-1, final_95th_percentile_value), xytext=(time_range-1+10, final_95th_percentile_value),
             arrowprops=dict(facecolor='blue', shrink=0.05), fontsize=12)
plt.show()
plt.savefig('output2.png')
