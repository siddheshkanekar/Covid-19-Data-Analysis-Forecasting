import pandas as pd
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss, r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression

#%%
FGC2=pd.read_csv(r"C:\Users\dbda.STUDENTSDC\Desktop\Project\project dummy\FGC2.csv")
FGC2= FGC2[(FGC2['Confirmed'] != ' ') & (FGC2['Deaths'] != ' ') & (FGC2['Recovered'] != " ") & (FGC2['Active'] != " ")& (FGC2['New_cases'] != " ") & (FGC2['New_recovered'] != " ") & (FGC2['New_deaths'] != " ")]
y=FGC2['Deaths']
#x=FGC2[['Confirmed','Recovered','Active','New_cases','New_deaths','New_recovered']]
x = FGC2.drop('Deaths', axis = 1)

#%%
kfold = KFold(n_splits = 5, shuffle = True, random_state = 23)
bag = BaggingRegressor(random_state = 23)
dtr = DecisionTreeRegressor(random_state = 23)
knn = KNeighborsRegressor()
elastic = ElasticNet()
lr = LinearRegression()

#%%

params = {'estimator': [lr, elastic, knn, dtr], 
          'n_estimators': [10,25,50,75]}
gcv_bg = GridSearchCV(bag, param_grid = params, cv = kfold, n_jobs = -1)

gcv_bg.fit(x,y)

#%%
print(gcv_bg.best_params_)
print(gcv_bg.best_score_)

#%%
bg = BaggingRegressor(random_state = 23, estimator = dtr)

params = {'estimator__max_depth': [None, 3,5],
          'estimator__min_samples_leaf': [2,5,10],
          'estimator__min_samples_split': [1,5,10],
          'n_estimators': [10, 50, 75]}

gcv_bg = GridSearchCV(bg, param_grid = params, cv = kfold, n_jobs = -1)

gcv_bg.fit(x,y)

#%%
print(gcv_bg.best_params_)
print(gcv_bg.best_score_)



















