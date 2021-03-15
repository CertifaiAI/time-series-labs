# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# > **Copyright &copy; 2020 CertifAI Sdn. Bhd.**<br>
#  **Copyright &copy; 2021 CertifAI Sdn. Bhd.**<br>
#  <br>
# This program and the accompanying materials are made available under the
# terms of the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0). \
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License. <br>
# <br>**SPDX-License-Identifier: Apache-2.0**
# %% [markdown]
# # 02 - ARIMA

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# # Stationarity

# %%
airpassengers = pd.read_csv('D:/CertifAI/time-series-labs/datasets/decomposition/AirPassengers.csv')

airpassengers_series = pd.Series(airpassengers['#Passengers'].values, 
                            index = pd.date_range('1949-01', periods = len(airpassengers), freq='M'))


# %%
plt.plot(airpassengers_series)
plt.title('airpassengers')
axes = plt.gca()

# %%
expected_Y = airpassengers['#Passengers'].values
expected_X = pd.date_range('1949-01', periods = len(airpassengers), freq='M')

from plotchecker import LinePlotChecker, ScatterPlotChecker, BarPlotChecker
pc = LinePlotChecker(axes)
pc.assert_num_lines(1)
pc.assert_x_data_equal(expected_X)
pc.assert_y_data_equal(expected_Y)
print('Success!!!')
# %% [markdown]
# There are many techniques (logarithm, exponential, de-trending, differencing) can be use to transform non-stationary series into stationary.
# 
# Which technique to use depend on the pattern of our time series
# 
# Since we has a series with increment variance, logarithm transformation can be use to smooth out the variance.

# %%
airpassengers_log = np.log(airpassengers_series)


# %%
%reset_selective -f "^ax1$"
%reset_selective -f "^ax2$"
%reset_selective -f "^ax$"
%reset_selective -f "^fig$"
# %%
# fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16, 4))
# ax = plt.gca()
ax1.plot(airpassengers_series)
ax = plt.gca()
ax1.set_title('original series')
# ax = plt.gca()
#%%
ax2.plot(airpassengers_log)
ax2.set_title('log transformation')

# %%


# %%
x_test = np.arange(0,144)
y_test = airpassengers_log.tolist()
plt.plot(x_test,y_test)
axes = plt.gca()
plt.show()

# %%
expected_x = x_test
expected_y = y_test
pc = LinePlotChecker(axes)
pc.assert_y_data_equal(expected_y)

print('Success!')
# %% [markdown]
# differencing

# %%
airpassengers_diff = airpassengers_log.diff()


# %%
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16, 4))

ax1.plot(airpassengers_log)
ax1.set_title('log transformation')

ax2.plot(airpassengers_diff)
ax2.set_title('differencing')

# %% [markdown]
# ## Stationarity Check

# %%
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16, 4))

ax1.plot(airpassengers_series, label = 'raw data')
ax1.plot(airpassengers_series.rolling(window=12).mean(), label="rolling mean");
ax1.plot(airpassengers_series.rolling(window=12).std(), label="rolling var");
ax1.set_title('original series')
ax1.legend()

ax2.plot(airpassengers_diff, label = 'transformed data')
ax2.plot(airpassengers_diff.rolling(window=12).mean(), label="rolling mean");
ax2.plot(airpassengers_diff.rolling(window=12).std(), label="rolling var");
ax2.set_title('stationary series')
ax2.legend()

# %% [markdown]
# ### Augmented Dickey-Fuller Test (ADF)
# 
# ADF is a type of unit root test. Unit roots are a cause for non-stationarity, the ADF test will test if unit root is present.
# 
# A time series is stationary if a single shift in time doesn’t change the time series statistical properties, in which case unit root does not exist.
# 
# The Null and Alternate hypothesis of the Augmented Dickey-Fuller test is defined as follows:
# - Null Hypothesis states there is the presence of a unit root.
# - Alternate Hypothesis states there is no unit root. In other words, Stationarity exists.
# 
# https://machinelearningmastery.com/time-series-data-stationary-python/
# 
# http://www.insightsbot.com/augmented-dickey-fuller-test-in-python/

# %%
def print_adf_result(adf_result):
    df_results = pd.Series(adf_result[0:4], index=['ADF Test Statistic','P-Value','# Lags Used','# Observations Used'])
    
    for key, value in adf_result[4].items():
        df_results['Critical Value (%s)'% key] = value
    print('Augmented Dickey-Fuller Test Results:')
    print(df_results)


# %%
from statsmodels.tsa.stattools import adfuller

result = adfuller(airpassengers_series, maxlag=12)
print_adf_result(result)


# %%
result = adfuller(airpassengers_diff[1:], maxlag=12)
print_adf_result(result)

# %% [markdown]
# ### Test Stationary using Autocorrelation Function (ACF)
# 
# http://rstudio-pubs-static.s3.amazonaws.com/311446_08b00d63cc794e158b1f4763eb70d43a.html

# %%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from statsmodels.graphics.gofplots import qqplot
# from scipy.stats import probplot


# %%
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16, 4))

plot_acf(airpassengers_series, ax1)
ax1.set_title('ACF of original series')

plot_acf(airpassengers_diff[1:], ax2)
ax2.set_title('ACF of differenced series')

plt.show()


# %%
fig, (ax3, ax4) = plt.subplots(1,2, figsize=(16, 4))

plot_pacf(airpassengers_series, ax3)
ax3.set_title('PACF of original series')

plot_pacf(airpassengers_diff[1:], ax4)
ax4.set_title('PACF of differenced series')

plt.show()

# %% [markdown]
# # Simple Forecasting

# %%
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

airpassengers_train = airpassengers_series[:-24]
airpassengers_test = airpassengers_series[-24:]

airpassengers_log_train = airpassengers_log[:-24]
airpassengers_log_test = airpassengers_log[-24:]

airpassengers_diff_train = airpassengers_diff[:-24]
airpassengers_diff_test = airpassengers_diff[-24:]

ses = SimpleExpSmoothing(airpassengers_diff_train[1:])
ses = ses.fit()

ses_forecast = ses.forecast(24)


# %%
plt.plot(airpassengers_diff_train)
plt.plot(ses_forecast)
plt.title('forecast for next 24 month')

# %% [markdown]
# Inverse differencing

# %%
ses_forecast[0] = ses_forecast[0] + airpassengers_log_train[-1]
ses_forecast_inv_diff = ses_forecast.cumsum()

# %% [markdown]
# Inverse log transformation

# %%
ses_forecast_inv_log = np.exp(ses_forecast_inv_diff)


# %%
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16, 4))

ax1.plot(airpassengers_log_train)
ax1.plot(airpassengers_log_test)
ax1.plot(ses_forecast_inv_diff)
ax1.set_title('inverse differencing')

ax2.plot(airpassengers_train)
ax2.plot(airpassengers_test)
ax2.plot(ses_forecast_inv_log)
ax2.set_title('inverse log transformation')

# %% [markdown]
# # ARIMA
# 
# ARIMA is stand for Autoregressive Integrated Moving Average. The integrated refers to differencing hence it allow the model to support series with trend.
# 
# ARIMA expects data that is either not seasonal or has the seasonal component removed, thus we can perform seasonal differencing to eliminate the seasonality in the data.

# %%
from statsmodels.tsa.arima_model import ARIMA


# %%
airpassengers_season_diff_train = airpassengers_train.diff(12)

plt.plot(airpassengers_season_diff_train)


# %%
fig, (ax5, ax6) = plt.subplots(1,2, figsize=(16, 4))

plot_acf(airpassengers_season_diff_train.dropna(), ax5)
ax3.set_title('ACF of differenced season seriess')

plot_pacf(airpassengers_season_diff_train.dropna(), ax6)
ax4.set_title('PACF of differenced season series')

plt.show()


# %%
#  Find d parameter for ARIMA
find_d = ARIMA(airpassengers_season_diff_train.dropna(), order=(0,0,0)).fit()
find_d.summary()


# %%
arima = ARIMA(airpassengers_season_diff_train.dropna(), order=(1,0,1)).fit()
arima.summary()

# %% [markdown]
# The values under *coef* are the weights of the respective terms. 
# 
# AIC and BIC is to tell how good is the model and can be use to compare with other models. The lower the AIC the better the model
# 
# 
# %% [markdown]
# ## Residuals
# 
# Residuals are useful in checking whether a model has adequately captured the information in the data. A good forecasting method will yield residuals with the following properties:
# - The residuals are uncorrelated. If there are correlations between residuals, then there is information left in the residuals which should be used in computing forecasts.
# - The residuals have zero mean. If the residuals have a mean other than zero, then the forecasts are biased.
# - The residuals have constant variance.
# - The residuals are normally distributed.
# 
# 
# Any forecasting method that does not satisfy these properties can be improved. However, that does not mean that forecasting methods that satisfy these properties cannot be improved. It is possible to have several different forecasting methods for the same data set, all of which satisfy these properties. Checking these properties is important in order to see whether a method is using all of the available information, but it is not a good way to select a forecasting method.

# %%
residuals = pd.Series(arima.resid)


# %%
import seaborn as sns

def check_residuals(series):
    fig = plt.figure(figsize=(16, 8))    
    gs = fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(series)
    ax1.set_title('residuals')
    
    ax2 = fig.add_subplot(gs[1,0])
    plot_acf(series, ax=ax2, title='ACF')
    
    ax3 = fig.add_subplot(gs[1,1])
    sns.kdeplot(series, ax=ax3)
    ax3.set_title('density')
    
    plt.show()


# %%
check_residuals(residuals)


# %%
arima_forecast, se, conf = arima.forecast(24)

arima_forecast = pd.Series(arima_forecast, index=airpassengers_test.index)
lower_series = pd.Series(conf[:, 0], index=airpassengers_test.index)
upper_series = pd.Series(conf[:, 1], index=airpassengers_test.index)


# %%
plt.plot(airpassengers_season_diff_train, label='train')
plt.plot(arima_forecast, label='forecast')

plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
plt.legend()


# %%
def inverse_differencing(orig_data, diff_data, interval):
    output = orig_data[:interval].tolist()
    for i in range(interval, len(diff_data)):
        output.append(output[i-interval] + diff_data[i])
    return output

def inverse_differencing_forecast(orig_series, diff_series, forecast_series, interval):
    series_merge = diff_series.append(forecast_series)
    inverse_diff_series = pd.Series(inverse_differencing(orig_series, series_merge, interval), 
                                    index=series_merge.index)
    return inverse_diff_series[-len(forecast_series):]

def train_test_forecast_plot(train_series, test_series, forecast_series, lower_upper=None):
    plt.plot(train_series, label = 'train')
    plt.plot(test_series, label = 'test')
    plt.plot(forecast_series, label = 'forecast')

    if lower_upper is not None:
        plt.fill_between(lower_upper[0].index, lower_upper[0], 
                     lower_upper[1], color='k', alpha=.15)
    plt.legend()


# %%
# inverse differenced series back to original series
airpassengers_forecast_series = inverse_differencing_forecast(airpassengers_train, airpassengers_season_diff_train, arima_forecast, 12)
airpassengers_lower_series = inverse_differencing_forecast(airpassengers_train, airpassengers_season_diff_train, lower_series, 12)
airpassengers_upper_series = inverse_differencing_forecast(airpassengers_train, airpassengers_season_diff_train, upper_series, 12)


# %%
type(airpassengers_train)


# %%
train_test_forecast_plot(airpassengers_train, airpassengers_test, airpassengers_forecast_series, 
                         [airpassengers_lower_series, airpassengers_upper_series])


# %%
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(airpassengers_test, airpassengers_forecast_series)
print('Test MSE: ', mse)

# %% [markdown]
# # SARIMA
# 
# Seasonal Autoregressive Integrated Moving Average (SARIMA) ia a method to forecast univariate time series with trend and seasonality.

# %%
from statsmodels.tsa.statespace.sarimax import SARIMAX


# %%
fig, (ax7, ax8) = plt.subplots(1,2, figsize=(16, 4))

plot_acf(airpassengers_train, ax7)
ax3.set_title('ACF of seasonal series')

plot_pacf(airpassengers_train, ax8)
ax4.set_title('PACF of seasonal series')

plt.show()


# %%
sarimax = SARIMAX(airpassengers_train, order=(3,1,1), seasonal_order=(0,1,0,12)).fit()
sarimax.summary()


# %%
sarimax.plot_diagnostics(figsize=(16, 8))
plt.show()


# %%
sarimax_forecast = sarimax.get_forecast(24)
sarimax_forecast_conf_int = sarimax_forecast.conf_int()


# %%
plt.plot(airpassengers_train, label='train')
plt.plot(airpassengers_test, label='test')
plt.plot(sarimax_forecast.predicted_mean, label='forecast')


plt.fill_between(sarimax_forecast_conf_int.index,
                 sarimax_forecast_conf_int.iloc[:, 0],
                 sarimax_forecast_conf_int.iloc[:, 1], color='k', alpha=.2)

plt.legend()

# %% [markdown]
# # Grid Search
# 
# Grid search is the process of performing exhaustive searching throught a manually specified parameters in order to determine the optimal values for a given model.
# 
# For example, ARIMA has the parameters p, d, and q. we can manually specify range of values for parameters p, d, q, and build models based on the all the combination of parameters in p, d, and q. The measurement for the models can be in-sample error (AIC, BIC), or out-sample error (MSE). Finally, the model with the lowest error will be selected.

# %%
param_p = [0,1,2,3,4,5]
param_d = [0,1] # ARIMA only support two times of differencing
param_q = [0,1,2]


# %%
best_error, best_params, best_model = None, None, None

for p in param_p:
    for d in param_d:
        for q in param_q:
            try:
                arima = ARIMA(airpassengers_season_diff_train.dropna(), order=(p,d,q)).fit()
                if best_error is None or arima.aic < best_error:
                    best_error = arima.aic
                    best_params = (p,d,q)
                    best_model = arima
                print('ARIMA({},{},{}), AIC={}'.format(p,d,q, arima.aic))
            except:
                pass
print('Best Error={}, Best Params={}'.format(best_error, best_params))


# %%
arima_forecast, se, conf = best_model.forecast(24)

arima_forecast = pd.Series(arima_forecast, index=airpassengers_test.index)
lower_series = pd.Series(conf[:, 0], index=airpassengers_test.index)
upper_series = pd.Series(conf[:, 1], index=airpassengers_test.index)


# %%
# inverse differenced series back to original series
airpassengers_forecast_series = inverse_differencing_forecast(airpassengers_train, airpassengers_season_diff_train, arima_forecast, 12)
airpassengers_lower_series = inverse_differencing_forecast(airpassengers_train, airpassengers_season_diff_train, lower_series, 12)
airpassengers_upper_series = inverse_differencing_forecast(airpassengers_train, airpassengers_season_diff_train, upper_series, 12)


# %%
train_test_forecast_plot(airpassengers_train, airpassengers_test, airpassengers_forecast_series, 
                         [airpassengers_lower_series, airpassengers_upper_series])


# %%
mse = mean_squared_error(airpassengers_test, airpassengers_forecast_series)
print('Test MSE: ', mse)

# %% [markdown]
# # Auto Arima
# 
# Pyramid brings R’s beloved `auto.arima` to Python

# %%
import pmdarima as pm
#scikit-learn version = 0.23.2


auto_arima = pm.arima.auto_arima(airpassengers_train, m=12,
                            trace=True, seasonal=True,
                            error_action='ignore',  
                            suppress_warnings=True)


# %%
auto_arima.summary()


# %%
auto_arima_forecast = auto_arima.predict(n_periods=24)
auto_arima_forecast_series = pd.Series(auto_arima_forecast, index=airpassengers_test.index)


# %%
train_test_forecast_plot(airpassengers_train, airpassengers_test, auto_arima_forecast_series, 
                         [airpassengers_lower_series, airpassengers_upper_series])

# %% [markdown]
# ## Exercise
# %% [markdown]
# We will run through a simple exercise of building ARIMA model for time series data analysis. 
# %% [markdown]
# Tasks that you are required to perform are list down as comment. Please insert your codes below the comment. An approximation of number of lines *n* is provided as a guideline to help you.
# %% [markdown]
# ### 1. Library and dataset loading

# %%
# import necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import seaborn as sns
from sklearn.metrics import mean_squared_error

get_ipython().run_line_magic('matplotlib', 'inline')

# load dataset
df = pd.read_csv('../../datasets/others/furniture-sales.csv')

# %% [markdown]
# ### 2. Data Pre-processing

# %%
# initial data 
df.head()

# setting DateTimeIndex since there is a datetime object conveniently 
df.DATE = pd.to_datetime(df.DATE, infer_datetime_format=True)
df = df.set_index(df.DATE)
df = df.asfreq("MS")

# remove date column
del df['DATE']

# rename column
df = df.rename(columns={'MRTSSM442USN': 'Furniture_Sales'})

# convert to Series
df = pd.Series(df.Furniture_Sales.values, index=df.index)

# display current Series
print(df.head())

# just extract and analyse data until the year 2006
df_subset = df.loc[:'2006']

# please perform a splitting to convert into train and test dataset
split_ratio = round(df_subset.shape[0]*0.8)
df_train = df_subset.iloc[:split_ratio]
df_test = df_subset.iloc[split_ratio:]

# %% [markdown]
# ### 3. EDA

# %%
# plot a time plot
df_train.plot()


# %%
# plot a lag plot for lag of 12
pd.plotting.lag_plot(df_train, lag=12)
plt.show()


# %%
# Stationarity check of mean, variance and autocorrelation
plt.plot(df_train, label='Furniture Sales')
plt.plot(df_train.rolling(window=12).mean(), label='mean')
plt.plot(df_train.rolling(window=12).std(), label='std')
plt.legend()
plt.show()


# %%
# perform ADF test
def print_adf_result(adf_result):
    df_results = pd.Series(adf_result[0:4], index=['ADF Test Statistic','P-Value','# Lags Used','# Observations Used'])
    
    for key, value in adf_result[4].items():
        df_results['Critical Value (%s)'% key] = value
    print('Augmented Dickey-Fuller Test Results:')
    print(df_results)
    

result = adfuller(df_train)
print_adf_result(result)

# %% [markdown]
# Obviously, the time series is not stationary. We will perform some transformations for it to be stationary.

# %%
# seasonal differencing 
df_train_seasonal_diff = df_train.diff(12)


# display result after transformation
df_train_seasonal_diff.dropna().plot()


# %%
# inspect stationarity using ADF after differecing
result = adfuller(df_train_seasonal_diff.dropna())
print_adf_result(result)


# %%
# inspect ACF and PACF to determine the hyperparameter to be used for ARIMA
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16, 4))

plot_acf(df_train_seasonal_diff.dropna(), ax1)
ax1.set_title('ACF of differenced season series')

plot_pacf(df_train_seasonal_diff.dropna(), ax2)
ax2.set_title('PACF of differenced season series')

plt.show()

# %% [markdown]
# ### 4. Model Building

# %%
# building ARIMA model
arima = ARIMA(df_train_seasonal_diff.dropna(), order=(3, 0, 3)).fit()

# display summary of model
arima.summary()


# %%
# residual analysis
residuals = pd.Series(arima.resid)

def check_residuals(series):
    fig = plt.figure(figsize=(16, 8))    
    gs = fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(series)
    ax1.set_title('residuals')
    
    ax2 = fig.add_subplot(gs[1,0])
    plot_acf(series, ax=ax2, title='ACF')
    
    ax3 = fig.add_subplot(gs[1,1])
    sns.kdeplot(series, ax=ax3)
    ax3.set_title('density')
    
    plt.show()
    
check_residuals(residuals)


# %%
# perform forecast using the newly built ARIMA model
arima_forecast, se, conf = arima.forecast(len(df_subset)-split_ratio)

arima_forecast = pd.Series(arima_forecast, index=df_test.index)
lower_series = pd.Series(conf[:, 0], index=df_test.index)
upper_series = pd.Series(conf[:, 1], index=df_test.index)

plt.plot(df_train_seasonal_diff, label='train')
plt.plot(arima_forecast, label='forecast')

plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
plt.legend()


# %%
# perform inverse transformation
def inverse_differencing(orig_data, diff_data, interval):
    output = orig_data[:interval].tolist()
    for i in range(interval, len(diff_data)):
        output.append(output[i-interval] + diff_data[i])
    return output

def inverse_differencing_forecast(orig_series, diff_series, forecast_series, interval):
    series_merge = diff_series.append(forecast_series)
    inverse_diff_series = pd.Series(inverse_differencing(orig_series, series_merge, interval), 
                                    index=series_merge.index)
    return inverse_diff_series[-len(forecast_series):]

def train_test_forecast_plot(train_series, test_series, forecast_series, lower_upper=None):
    plt.plot(train_series, label = 'train')
    plt.plot(test_series, label = 'test')
    plt.plot(forecast_series, label = 'forecast')

    if lower_upper is not None:
        plt.fill_between(lower_upper[0].index, lower_upper[0], 
                     lower_upper[1], color='k', alpha=.15)
    plt.legend()
    
df_subset_forecast_series = inverse_differencing_forecast(df_train, df_train_seasonal_diff, arima_forecast, 12)
df_subset_lower_series = inverse_differencing_forecast(df_train, df_train_seasonal_diff, lower_series, 12)
df_subset_upper_series = inverse_differencing_forecast(df_train, df_train_seasonal_diff, upper_series, 12)

train_test_forecast_plot(df_train, df_test, df_subset_forecast_series, 
                         [df_subset_lower_series, df_subset_upper_series])

# %% [markdown]
# ### 5. Model Evaluation

# %%
# Model evaluation
mse = mean_squared_error(df_test, df_subset_forecast_series)
print('Test MSE: ', mse)

# %% [markdown]
# ## References
# 
# 1. https://towardsdatascience.com/time-series-in-python-exponential-smoothing-and-arima-processes-2c67f2a52788

