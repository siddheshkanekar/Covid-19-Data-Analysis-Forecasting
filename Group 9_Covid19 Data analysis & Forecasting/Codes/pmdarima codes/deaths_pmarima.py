# -*- coding: utf-8 -*-
"""Deaths pmarima.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14A5HohK0K7NSDddXJOolH6fEqATTjmtx
"""

pip install pmdarima

import pmdarima as pm
from pmdarima.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA

df =  pd.read_excel("FCG1.xlsx")
df['Date']=pd.to_datetime(df['Date'])
result_Death = df.groupby('Date')['Deaths'].sum().reset_index()
result_Death.set_index("Date",inplace=True)

result_Death

result_Death.plot()

"""No seasonality"""

y = result_Death
train, test = train_test_split(y, train_size=150)

model = pm.auto_arima(train, seasonal=False, m=12)

forecasts = model.predict(test.shape[0])

x = np.arange(y.shape[0])
plt.plot(x[:150], train, c='blue')
plt.plot(x[150:], forecasts, c='black')
plt.show()
