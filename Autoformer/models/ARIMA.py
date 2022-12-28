
import warnings
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Load the data
data = pd.read_csv('dataset/datagen/BTP.csv', engine='python')
# A bit of pre-processing to make it nicer
# data['Date']=pd.to_datetime(data['Date'], format='%Y-%m-%d')
data = data[['date','OT']]
data.set_index(['date'], inplace=True)
print(data[:5])



scaler = StandardScaler()
# scaler.fit(data['OT'])
# print(data['OT']).shape
mu = np.mean(data['OT'])
sigma = np.std(data['OT'])
print(mu,sigma)
            # print('standard: ',mu.shape,sigma.shape)
            # print('scale shape:',train_data.values.shape)
# data['OT'] = scaler.transform(data['OT'])
data['OT'] = (data['OT']-mu)/sigma
print(data[:5])

# data.plot()
# plt.ylabel('Monthly airline passengers (x1000)')
# plt.xlabel('Date')
# plt.show()

# Plot the data


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

train = 0.9
train_data = data[:int(train*len(data))]
test_data = data[int(train*len(data)):]

warnings.filterwarnings("ignore") # specify to ignore warning messages
AIC = []
SARIMAX_model = []

train_len = 12
pred_len = 5

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(train_data[-train_len:],
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            AIC.append(results.aic)
            SARIMAX_model.append([param, param_seasonal])
        except:
            continue
print(data.index)
print('最小 AIC 值为: {} 对应模型参数: SARIMAX{}x{}'.format(min(AIC), SARIMAX_model[AIC.index(min(AIC))][0],SARIMAX_model[AIC.index(min(AIC))][1]))

X = []
Y = []

for i in range(1,50):
    # mod = sm.tsa.statespace.SARIMAX(data[(i-1)*(train_len+pred_len):(i-1)*(train_len+pred_len)+train_len],
    #                                 order=(1, 1, 1),
    #                                 seasonal_order=(1, 1, 1, 12),
    #                                 enforce_stationarity=False,
    #                                 enforce_invertibility=False)
    mod = sm.tsa.statespace.SARIMAX(train_data[(i-1)*(train_len+pred_len):(i-1)*(train_len+pred_len)+train_len],
                                    order=SARIMAX_model[AIC.index(min(AIC))][0],
                                    seasonal_order=SARIMAX_model[AIC.index(min(AIC))][1],
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

    results = mod.fit()
    prediction = list(results.forecast(5))
    truth = list(data['OT'].values[(i-1)*(train_len+pred_len)+train_len:(i-1)*(train_len+pred_len)+(train_len+pred_len)])
    Y.append(prediction)
    X.append(truth)

print(np.array(X).shape,np.array(Y).shape)

def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def R2(pred, true):
    sse = np.sum((true - pred) ** 2)
    sst = np.sum((true - np.mean(true)) ** 2)
    r2 = 1 - sse / sst
    return r2

X = np.array(X)
Y = np.array(Y)

X = X*sigma + mu
Y = Y*sigma + mu

print('X shape',X.shape)
for i in range(0,5):
    X_ = X[:,i]
    Y_ = Y[:,i]
    print('r2: ',i,' ',R2(Y_,X_), 'rmse: ',RMSE(Y_,X_))


# print(results.summary().tables[1])

# results.plot_diagnostics(figsize=(15, 12))
# plt.show()


# # pred2 = results.get_forecast('2021/3/25  11:33:00')
# pred2 = results.get_prediction(start='2021/3/25 11:33', dynamic=False)
# pred2_ci = pred2.conf_int()
# # print(pred2.predicted_mean[start:]])

# ax = data.plot(figsize=(20, 16))
# # ax.plot(pred2.predicted_mean,label='Dynamic Forecast (get_forecast)')
# # pred2.predicted_mean.plot(ax=ax, label='Dynamic Forecast (get_forecast)')
# # data.plot()
# ax.fill_between(pred2_ci.index, pred2_ci.iloc[:, 0], pred2_ci.iloc[:, 1], color='k', alpha=.1)
# plt.ylabel('Monthly airline passengers (x1000)')
# plt.xlabel('Date')
# plt.legend()
# plt.show()

# prediction = list(results.forecast(5))
# # print(prediction)
# truth = list(test_data['OT'].values[:5])
# # print(truth)

# # prediction = pred2.predicted_mean['2021/3/25 11:33':'2021/3/25 11:37'].values
# # # flatten nested list
# # truth = list(itertools.chain.from_iterable(data['2021/3/25 11:33':'2021/3/25 11:37'].values))
# # data*sigma+mu
# prediction = [i*sigma+mu for i in prediction]
# truth = [i*sigma+mu for i in truth]
# print(prediction)
# print(truth)

# plt.plot(range(0,len(prediction)),prediction)
# plt.plot(range(0,len(prediction)),truth)
# plt.show()

# Mean Absolute Percentage Error
# for i in range(0,len(prediction)):
#     print("predict ",i+1,"th")
#     truth_ = truth[i]
#     prediction_ = prediction[i]
#     MAPE = np.mean(np.abs((truth_ - prediction_) / truth_)) * 100
#     RMSE = np.sqrt(np.mean((prediction_ - truth_) ** 2))
#     sse = np.sum((prediction_ - truth_) ** 2)
#     sst = np.sum((truth_ - np.mean(truth_)) ** 2)
#     r2 = 1-sse/sst

#     print('The Mean Absolute Percentage Error for the forecast  is {:.2f}%,r2 is {},rmse is {}'.format(MAPE,r2,RMSE))



# truth_ = np.array(truth)
# prediction_ = np.array(prediction)

# MAPE = np.mean(np.abs((truth_ - prediction_) / truth_)) * 100
# RMSE = np.sqrt(np.mean((prediction_ - truth_) ** 2))
# sse = np.sum((prediction_ - truth_) ** 2)
# sst = np.sum((truth_ - np.mean(truth_)) ** 2)
# r2 = 1-sse/sst

# print(r2)
# print('r2: ',r2_score(truth_,prediction_))


# # testpred = np.load('results/BTP_100_1_AutoCoTransformer_custom_ftMS_sl10_ll5_pl5_dm64_nh8_el2_dl1_df256_fc3_ebtimeF_dtTrue_Exp_0/pred.npy')
# # testtrue = np.load('results/BTP_100_1_AutoCoTransformer_custom_ftMS_sl10_ll5_pl5_dm64_nh8_el2_dl1_df256_fc3_ebtimeF_dtTrue_Exp_0/true.npy')

# # print(testpred)
# # print(testtrue)
# prediction_ = np.array([[85.27284790493039, 85.27608445695743, 85.26906393505051, 85.25260829227255, 85.25517089146382]])
# truth_ = np.array([[85.44, 85.58, 85.72, 85.86, 85.86]])

# def R2(pred, true):
#     sse = np.sum((true - pred) ** 2)
#     sst = np.sum((true - np.mean(true)) ** 2)
#     r2 = 1 - sse / sst
#     return r2

# print('r2: ',R2(prediction_,truth_))
# print('The Mean Absolute Percentage Error for the forecast  is {:.2f}%,r2 is {},rmse is {}'.format(MAPE,r2,RMSE))