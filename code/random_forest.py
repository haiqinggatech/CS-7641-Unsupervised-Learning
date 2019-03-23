import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import load_data, nn_layers, nn_reg, nn_iter, cluster_acc, myGMM, clusters, dims, dims_big, run_clustering, pairwiseDistCorr, reconstructionError, ImportanceSelect
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

out = '../results/random_forest/'

perm_x, perm_y, housing_x, housing_y = load_data() # perm, housing
# raise Exception('Remove this line to run code')

#2

rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=5,n_jobs=7)
fs_perm = rfc.fit(perm_x,perm_y).feature_importances_ 
fs_housing = rfc.fit(housing_x,housing_y).feature_importances_ 

tmp = pd.Series(np.sort(fs_perm)[::-1])
tmp.to_csv(out+'perm scree.csv')

tmp = pd.Series(np.sort(fs_housing)[::-1])
tmp.to_csv(out+'housing scree.csv')

#4
filtr = ImportanceSelect(rfc)
grid ={'filter__n':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_layers}
mlp = MLPClassifier(activation='relu',max_iter=nn_iter,early_stopping=True,random_state=5)
pipe = Pipeline([('filter',filtr),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(perm_x,perm_y)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'perm dim red.csv')


grid ={'filter__n':dims_big,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_layers}  
mlp = MLPClassifier(activation='relu',max_iter=nn_iter,early_stopping=True,random_state=5)
pipe = Pipeline([('filter',filtr),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(housing_x,housing_y)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'housing dim red.csv')


#3
dim = 5
filtr = ImportanceSelect(rfc,dim)
perm_x2 = filtr.fit_transform(perm_x,perm_y)


dim = 9
filtr = ImportanceSelect(rfc,dim)
housing_x2 = filtr.fit_transform(housing_x,housing_y)

run_clustering(out, perm_x2, perm_y, housing_x2, housing_y)
