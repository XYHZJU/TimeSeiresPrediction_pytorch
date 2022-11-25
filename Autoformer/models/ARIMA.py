import warnings
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load the data
data = pd.read_csv('dataset/datagen/BTP.csv', engine='python')
# A bit of pre-processing to make it nicer
# data['Date']=pd.to_datetime(data['Date'], format='%Y-%m-%d')
data = data[['date','OT']]
data.set_index(['date'], inplace=True)

# Plot the data
data.plot()
plt.ylabel('Monthly airline passengers (x1000)')
plt.xlabel('Date')
plt.show()

# Define the d and q parameters to take any value between 0 and 1
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

train = 0.7
train_data = data[:int(0.7*len(data))]
test_data = data[int(0.7*len(data)):]

# warnings.filterwarnings("ignore") # specify to ignore warning messages

# for param in pdq:
#     for param_seasonal in seasonal_pdq:
#         try:
#             mod = sm.tsa.statespace.SARIMAX(data,
#                                             order=param,
#                                             seasonal_order=param_seasonal,
#                                             enforce_stationarity=False,
#                                             enforce_invertibility=False)

#             results = mod.fit()

#             print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
#         except:
#             continue
# print(data.index)
mod = sm.tsa.statespace.SARIMAX(data,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])

results.plot_diagnostics(figsize=(15, 12))
plt.show()


# pred2 = results.get_forecast('2021/3/25  11:33:00')
pred2 = results.get_prediction(start='2021/3/25 11:33', dynamic=False)
pred2_ci = pred2.conf_int()
# print(pred2.predicted_mean[start:]])

ax = data.plot(figsize=(20, 16))
# ax.plot(pred2.predicted_mean,label='Dynamic Forecast (get_forecast)')
# pred2.predicted_mean.plot(ax=ax, label='Dynamic Forecast (get_forecast)')
# data.plot()
ax.fill_between(pred2_ci.index, pred2_ci.iloc[:, 0], pred2_ci.iloc[:, 1], color='k', alpha=.1)
plt.ylabel('Monthly airline passengers (x1000)')
plt.xlabel('Date')
plt.legend()
plt.show()

prediction = pred2.predicted_mean['2021/3/25 11:33':'2021/3/25 11:40'].values
# flatten nested list
truth = list(itertools.chain.from_iterable(data['2021/3/25 11:33':'2021/3/25 11:40'].values))
# Mean Absolute Percentage Error
MAPE = np.mean(np.abs((truth - prediction) / truth)) * 100
sse = np.sum((prediction - truth) ** 2)
sst = np.sum((truth - np.mean(truth)) ** 2)
r2 = 1-sse/sst

print('The Mean Absolute Percentage Error for the forecast  is {:.2f}%,r2 is {}'.format(MAPE,r2))