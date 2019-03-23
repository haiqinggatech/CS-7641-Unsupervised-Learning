import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import load_data, nn_layers, nn_reg, nn_iter, cluster_acc, myGMM, clusters, dims, dims_big, run_clustering
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import FastICA

out = '../results/ica/'

perm_x, perm_y, housing_x, housing_y = load_data() # perm, housing


#2 

ica = FastICA(random_state=5)
kurt = {}
for dim in dims:
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(perm_x)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt[dim] = tmp.abs().mean()

kurt = pd.Series(kurt) 
kurt.to_csv(out+'perm scree.csv')


ica = FastICA(random_state=5)
kurt = {}
for dim in dims_big:
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(housing_x)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt[dim] = tmp.abs().mean()

kurt = pd.Series(kurt) 
kurt.to_csv(out+'housing scree.csv')
# raise


#4
grid ={'ica__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_layers}
ica = FastICA(random_state=5)       
mlp = MLPClassifier(activation='relu',max_iter=nn_iter,early_stopping=True,random_state=5)
pipe = Pipeline([('ica',ica),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(perm_x,perm_y)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'perm dim red.csv')


grid ={'ica__n_components':dims_big,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_layers}
ica = FastICA(random_state=5)       
mlp = MLPClassifier(activation='relu',max_iter=nn_iter,early_stopping=True,random_state=5)
pipe = Pipeline([('ica',ica),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(housing_x,housing_y)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'housing dim red.csv')

#3
dim = 5
ica = FastICA(n_components=dim,random_state=10)
perm_x2 = ica.fit_transform(perm_x)


dim = 9
ica = FastICA(n_components=dim,random_state=10)
housing_x2 = ica.fit_transform(housing_x)

run_clustering(out, perm_x2, perm_y, housing_x2, housing_y)

