from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score
from sklearn.model_selection import KFold,cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

#%%
FGC2=pd.read_csv(r"C:\Users\dbda.STUDENTSDC\Desktop\Project\project dummy\FGC2.csv")
FGC2= FGC2[(FGC2['Confirmed'] != ' ') & (FGC2['Deaths'] != ' ') & (FGC2['Recovered'] != " ") & (FGC2['Active'] != " ")& (FGC2['New_cases'] != " ") & (FGC2['New_recovered'] != " ") & (FGC2['New_deaths'] != " ")]
y=FGC2['Deaths']
#x=FGC2[['Confirmed','Recovered','Active','New_cases','New_deaths','New_recovered']]
X= FGC2.drop('Deaths', axis = 1)
#%%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=23)

#%%
from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor(n_neighbors=6)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print(r2_score(y_test,y_pred))

#%%
params={'n_neighbors':[1,2,3,4,5,6,7,8,9,10]}
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=23)
gcv=GridSearchCV(knn,param_grid=params,cv=kfold,scoring='neg_log_loss')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

#%%
print(knn.get_params())

#%%
scaler = StandardScaler()
scaler.fit(X_train)
X_trn_scl = scaler.transform(X_train)
X_tst_scl = scaler.transform(X_test)
scores_ss = dict()
for k in [3,5,7]:
    knn = KNeighborsRegressor(n_neighbors = k)
    knn.fit(X_train, y_train)










