from sklearn.metrics import accuracy_score, log_loss

import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


#%%
FGC2=pd.read_csv(r"C:\Users\dbda.STUDENTSDC\Desktop\Project\project dummy\FGC2.csv")
FGC2= FGC2[(FGC2['Confirmed'] != ' ') & (FGC2['Deaths'] != ' ') & (FGC2['Recovered'] != " ") & (FGC2['Active'] != " ")& (FGC2['New_cases'] != " ") & (FGC2['New_recovered'] != " ") & (FGC2['New_deaths'] != " ")]
y=FGC2['Deaths']
#x=FGC2[['Confirmed','Recovered','Active','New_cases','New_deaths','New_recovered']]
x = FGC2.drop('Deaths', axis = 1)

dtc = DecisionTreeClassifier(random_state = 23, max_depth = 1)
lr = LogisticRegression()
svm = SVC(probability = True, random_state = 23)

kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 23)

#%%
#AdaBoost Classifier
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(random_state = 23)

params = {'estimator': [dtc, lr, svm],
          'n_estimators': [25,50,100]}


gcv_ada = GridSearchCV(ada, param_grid = params, cv = kfold, scoring = 'neg_log_loss')

gcv_ada.fit(x,y)

#%%
print(gcv_ada.best_score_)
print(gcv_ada.best_params_)

#%%
#Gradient Boosting Classifier (GBM)
from sklearn.ensemble import GradientBoostingClassifier

gbm = GradientBoostingClassifier(random_state = 23)

params = {'learning_rate': np.linspace(0.001, 1, 10),
          'max_depth': [1,2,None],
          'n_estimators': [10,20,50]}

gcv_gbm = GridSearchCV(gbm, param_grid = params, cv = kfold, scoring = 'neg_log_loss')

gcv_gbm.fit(x,y)

#%%
print(gcv_gbm.best_score_)
print(gcv_gbm.best_params_)
#%%
#X G Boost

x_gbm = XGBClassifier(random_state = 23)

params = {'learning_rate': np.linspace(0.001,1,10),
          'max_depth': [1,3,5,None],
          'n_estimators':[10,20,50,100],
          'tree_method': ['hist', 'approx', 'exact']}
gcv_x_gbm = GridSearchCV(x_gbm, param_grid = params, verbose = 3, cv = kfold, scoring = 'neg_log_loss')

gcv_x_gbm.fit(x,y)

#%%
print(gcv_x_gbm.best_score_)
print(gcv_x_gbm.best_params_)
#%%
#Light GBM

l_gbm = LGBMClassifier(random_state = 23)

params = {'learning_rate': np.linspace(0.001,1,10),
          'max_depth': [1,3,5,None],
          'n_estimators':[10,20,50,100]}
gcv_l_gbm = GridSearchCV(l_gbm, param_grid = params, verbose = 3, cv = kfold, scoring = 'neg_log_loss')

gcv_l_gbm.fit(x,y)
#%%
print(gcv_l_gbm.best_score_)
print(gcv_l_gbm.best_params_)
#%%
#Cat Boost 
c_gbm = CatBoostClassifier(random_state = 23)

params = {'learning_rate': np.linspace(0.001,1,10),
          'max_depth': [1,3,5, None],
          'n_estimators': [10,20,50]}

gcv_c_gbm = GridSearchCV(c_gbm, param_grid =params, verbose = 3, cv = kfold, scoring = 'neg_log_loss')
gcv_c_gbm.fit(x,y)
#Cat Boost 
#command - pip install catboost


c_gbm = CatBoostClassifier(random_state = 23)

params = {'learning_rate': np.linspace(0.001,1,10),
          'max_depth': [1,3,5, None],
          'n_estimators': [10,20,50]}

gcv_c_gbm = GridSearchCV(c_gbm, param_grid =params, verbose = 3, cv = kfold, scoring = 'neg_log_loss')
gcv_c_gbm.fit(x,y)
#%%
print(gcv_c_gbm.best_score_)
print(gcv_c_gbm.best_params_)






