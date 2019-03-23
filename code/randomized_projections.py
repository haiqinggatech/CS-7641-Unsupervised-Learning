
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from collections import defaultdict
from helpers import load_data, nn_layers, nn_reg, nn_iter, cluster_acc, myGMM, clusters, dims, dims_big, run_clustering, pairwiseDistCorr, reconstructionError
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from itertools import product

out = '../results/randomized_projections/'

perm_x, perm_y, housing_x, housing_y = load_data() # perm, housing
#2
print(1)
tmp = defaultdict(dict)
for i,dim in product(range(10),dims):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(perm_x), perm_x)
tmp =pd.DataFrame(tmp).T
tmp.to_csv(out+'perm scree1.csv')

print(2)
tmp = defaultdict(dict)
for i,dim in product(range(10),dims_big):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(housing_x), housing_x)
tmp =pd.DataFrame(tmp).T
tmp.to_csv(out+'housing scree1.csv')

print(3)
tmp = defaultdict(dict)
for i,dim in product(range(10),dims):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    rp.fit(perm_x)    
    tmp[dim][i] = reconstructionError(rp, perm_x)
tmp =pd.DataFrame(tmp).T
tmp.to_csv(out+'perm scree2.csv')

print(4)
tmp = defaultdict(dict)
for i,dim in product(range(10),dims_big):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    rp.fit(housing_x)  
    tmp[dim][i] = reconstructionError(rp, housing_x)
tmp =pd.DataFrame(tmp).T
tmp.to_csv(out+'housing scree2.csv')

#4
grid ={'rp__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_layers}
rp = SparseRandomProjection(random_state=5)       
mlp = MLPClassifier(activation='relu',max_iter=nn_iter,early_stopping=True,random_state=5)
pipe = Pipeline([('rp',rp),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(perm_x,perm_y)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'perm dim red.csv')


grid ={'rp__n_components':dims_big,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_layers}
rp = SparseRandomProjection(random_state=5)           
mlp = MLPClassifier(activation='relu',max_iter=nn_iter,early_stopping=True,random_state=5)
pipe = Pipeline([('rp',rp),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(housing_x,housing_y)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'housing dim red.csv')



#3
dim = 5
rp = SparseRandomProjection(n_components=dim,random_state=5)
perm_x2 = rp.fit_transform(perm_x)


dim = 9
rp = SparseRandomProjection(n_components=dim,random_state=5)
housing_x2 = rp.fit_transform(housing_x)

run_clustering(out, perm_x2, perm_y, housing_x2, housing_y)


