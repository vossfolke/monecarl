import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
from arch import arch_model

# Function to get data
def get_data(etf_list, start, end):
    data = yf.download(etf_list, start=start, end=end)['Close']
    returns = data.pct_change().dropna()
    return returns

# List of stocks
stock_list = ['ACWD.L', 'DFNS.L', 'IITU.L', 'WLDS.L', 'AAPL', 'LMT', 'NATO.L', 'JEDI.L']

# Define the date range
end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=365*10)

# Get the returns
returns = get_data(stock_list, start_date, end_date)

# Fit a GARCH model to each stock's returns
garch_models = {}
forecasted_vols = {}
for stock in stock_list:
    model = arch_model(returns[stock], vol='Garch', p=1, q=1)
    garch_fit = model.fit(disp='off')
    garch_models[stock] = garch_fit
    forecasted_vols[stock] = np.sqrt(garch_fit.forecast(horizon=365*5).variance.values[-1, :])

# Monte Carlo method
mc_sims = 1000
time_range = 365*5

mean_returns = returns.mean()
cov_matrix = returns.cov()*2

meanMatrix = np.full(shape=(time_range, len(stock_list)), fill_value=mean_returns).T
portfolio_sims = np.full(shape=(time_range, mc_sims), fill_value=0.0)

initial_value = 5000
weights = [0.6, 0.07, 0.1, 0.1, 0.01, 0.02, 0.05, 0.05]

for m in range(mc_sims):
    Z = np.random.normal(size=(time_range, len(stock_list)))
    L = np.linalg.cholesky(cov_matrix)
    daily_vols = np.array([forecasted_vols[stock] for stock in stock_list]).T
    daily_returns = meanMatrix + (daily_vols * Z).T
    portfolio_sims[:, m] = np.cumprod(np.inner(weights, daily_returns.T) + 1) * initial_value

# Calculate the mean and percentiles for the simulations
mean_simulation = portfolio_sims.mean(axis=1)
percentile_5th = np.percentile(portfolio_sims, 5, axis=1)
percentile_95th = np.percentile(portfolio_sims, 95, axis=1)

# Get the final values for the three cases
final_mean_value = mean_simulation[-1]
final_5th_percentile_value = percentile_5th[-1]
final_95th_percentile_value = percentile_95th[-1]

# Plot the results
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
plt.savefig('output_garch.png')

# Print the final values for the three cases
print(f"Final Mean Value: ${final_mean_value:.2f}")
print(f"Final 5th Percentile Value: ${final_5th_percentile_value:.2f}")
print(f"Final 95th Percentile Value: ${final_95th_percentile_value:.2f}")