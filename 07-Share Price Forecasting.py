#Share Price Forecasting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fbprophet as fbp
from sklearn.metrics import mean_squared_error, mean_absolute_error
plt.style.use('fivethirtyeight') 

df = pd.read_csv('sample_data / AMZN.csv')
df.head()
df[['ds', 'y']] = df[['Date', 'Adj Close']]
df = df[['ds', 'y']]
 
df.head()

split_date = "2019-07-21"
df_train = df.loc[df.ds <= split_date].copy()
df_test = df.loc[df.ds > split_date].copy()
model = fbp.Prophet()
model.fit(df_train)
forecast = model.predict(df_test)
forecast.tail()
model.plot(forecast)
model.plot_components(forecast)
print("Mean Squared Error (MSE):", mean_squared_error(y_true = df_test["y"], y_pred = forecast['yhat']))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_true = df_test["y"], y_pred = forecast['yhat']))
def mean_abs_perc_err(y_true, y_pred): 
	return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print("Mean Absolute % Error (MAPE): ", mean_abs_perc_err(y_true = np.asarray(df_test["y"]), y_pred = np.asarray(forecast['yhat'])))
