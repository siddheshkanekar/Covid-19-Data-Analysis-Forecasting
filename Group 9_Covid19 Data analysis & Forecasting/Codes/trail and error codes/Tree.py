import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import r2_score

#%%
FGC2=pd.read_csv(r"C:\Users\dbda.STUDENTSDC\Desktop\Project\project dummy\FGC2.csv")
FGC2= FGC2[(FGC2['Confirmed'] != ' ') & (FGC2['Deaths'] != ' ') & (FGC2['Recovered'] != " ") & (FGC2['Active'] != " ")& (FGC2['New_cases'] != " ") & (FGC2['New_recovered'] != " ") & (FGC2['New_deaths'] != " ")]
y=FGC2['Deaths']
#x=FGC2[['Confirmed','Recovered','Active','New_cases','New_deaths','New_recovered']]
x = FGC2.drop('Deaths', axis = 1)

#%%
dtr = DecisionTreeRegressor(random_state = 23)
kfold = KFold(n_splits = 5, shuffle = True, random_state = 23)

params = {'max_depth':[2,3,4,5,6,7,None],
          'min_samples_split':[2,5,10,20],
          'min_samples_leaf': [1,5,7,10,20]}

gcv_tree = GridSearchCV(dtr, param_grid = params, cv = kfold, verbose = 3)
gcv_tree.fit(x,y)

print(gcv_tree.best_params_)
print(gcv_tree.best_score_)
pd_cv = pd.DataFrame(gcv_tree.cv_results_)