import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error as mse
from numpy import sqrt

#%%
FGC2=pd.read_csv(r"C:\Users\dbda.STUDENTSDC\Desktop\Project\project dummy\FGC2.csv")
FGC2.plot()
plt.show()
series = FGC2['Deaths']

FGC2.head(20)

y = FGC2['Deaths']
y_train =FGC2['Deaths'][:-5000]
y_test = FGC2['Deaths'][-5000:]

#%%
##############AutoRegressive Models SARIMA ########################

from pmdarima.arima import auto_arima
#train MA
model = auto_arima(y_train, trace = True,
                   error_action = 'ignore',
                   suppress_warnings = True,
                   seasonal = True, m = 12) #m = Period interval                  

#%%
#Make predictions
forecast = model.predict(n_periods = len(y_test))
forecast = pd.DataFrame(forecast, index = y_test.index,
                        columns = ['Prediction'])

#%%

plt.plot(y_train, color = 'blue', label = 'Train')
plt.plot(y_test, color = 'pink', label = 'Test')
plt.plot(forecast, color = 'purple', label = 'Forecast')

#%%
error = round(sqrt(mse(y_test, forecast)),2)

#plt.text(0.5, 0.5, 'Axis', transform = y_test.transAxes)

plt.text(100,302, 'RMSE = ' + str(error))
plt.legend(loc = 'best')
plt.show()




















