
import numpy as np
import pandas as pd

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler



#%%
FGC2=pd.read_csv(r"C:\Users\dbda.STUDENTSDC\Desktop\Project\project dummy\FGC2.csv")
FGC2= FGC2[(FGC2['Confirmed'] != ' ') & (FGC2['Deaths'] != ' ') & (FGC2['Recovered'] != " ") & (FGC2['Active'] != " ")& (FGC2['New_cases'] != " ") & (FGC2['New_recovered'] != " ") & (FGC2['New_deaths'] != " ")]
y=FGC2['Deaths']
#x=FGC2[['Confirmed','Recovered','Active','New_cases','New_deaths','New_recovered']]
x = FGC2.drop('Deaths', axis = 1)

#%%
std_scl = StandardScaler()
mm_scl = MinMaxScaler()
kfold = KFold(n_splits = 5, shuffle = True, random_state = 23)

#%%
#Linear
svm_lin = SVC(kernel = 'linear', probability = True, random_state = 23)

pipe_lin = Pipeline([('SCL', None), ('SVM', svm_lin)])

linparams = {'SVM__C': np.linspace(0.001, 6, 20),
          'SCL': [None, std_scl, mm_scl]}


#%%

gcv_lin = GridSearchCV(pipe_lin, param_grid = linparams, cv = kfold,
                       scoring = 'neg_log_loss', verbose = 3)

gcv_lin.fit(x,y)


#%%
#Polynomial
svm_poly = SVC(kernel = 'poly', probability = True, random_state = 23)
pipe_poly = Pipeline([('SCL', None), ('SVM', svm_poly)])

polyparams = {'SVM__C': np.linspace(0.001, 6, 20),
          'SVM__degree': [2,3,4,5,6],
          'SVM__coef0': np.linspace(-1,2,10),
          'SCL': [None, std_scl, mm_scl]}

#%%
gcv_poly = GridSearchCV(pipe_poly, param_grid = polyparams, cv = kfold, 
                        scoring = 'neg_log_loss', verbose = 3)
gcv_poly.fit(x,y)

#%%
print(gcv_poly.best_params_)
print(gcv_poly.best_score_)


#%%
svm_rbf = SVC(kernel = 'rbf', probability = True, random_state = 23)
pipe_rbf = Pipeline([('SCL', None), ('SVM', svm_rbf)])
rbfparams = {'SVM__C': np.linspace(0.001,6,20),
          'SVM__gamma': np.linspace(0.001, 5, 10),
          'SCL': [None, std_scl, mm_scl]}
#%%
gcv_rbf = GridSearchCV(pipe_rbf, param_grid = rbfparams, cv = kfold,
                       scoring = 'neg_log_loss', verbose = 3)
gcv_rbf.fit(x,y)
#%%
print(gcv_rbf.best_params_)
print(gcv_rbf.best_score_)





































