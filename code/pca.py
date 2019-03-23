
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import load_data, nn_layers, nn_reg, nn_iter, cluster_acc, myGMM, clusters, dims, dims_big, run_clustering
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

import pandas as pd

out = '../results/pca/'

perm_x, perm_y, housing_x, housing_y = load_data() # perm, housing


# 2

pca = PCA(random_state=5)
pca.fit(perm_x)
tmp = pd.Series(data = pca.explained_variance_,index = range(1,25))
tmp1 = pd.Series(data = pca.explained_variance_ratio_,index = range(1,25))
tmp2 = pd.Series(data = pca.explained_variance_ratio_.cumsum(),index = range(1,25))
tmp3 = pd.Series(data = pca.singular_values_,index = range(1,25))
pd.concat([tmp, tmp1, tmp2, tmp3], axis=1).to_csv(out+'perm scree.csv')


pca = PCA(random_state=5)
pca.fit(housing_x)
tmp = pd.Series(data = pca.explained_variance_,index = range(1,12))
tmp1 = pd.Series(data = pca.explained_variance_ratio_,index = range(1,12))
tmp2 = pd.Series(data = pca.explained_variance_ratio_.cumsum(),index = range(1,12))
tmp3 = pd.Series(data = pca.singular_values_,index = range(1,12))
pd.concat([tmp, tmp1, tmp2, tmp3], axis=1).to_csv(out+'housing scree.csv')

# raise Exception('wtf')
#4

grid ={'pca__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_layers}
pca = PCA(random_state=5)       
mlp = MLPClassifier(activation='relu',max_iter=nn_iter,early_stopping=True,random_state=5)
pipe = Pipeline([('pca',pca),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(perm_x,perm_y)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'perm dim red.csv')


grid ={'pca__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_layers}
pca = PCA(random_state=5)       
mlp = MLPClassifier(activation='relu',max_iter=nn_iter,early_stopping=True,random_state=5)
pipe = Pipeline([('pca',pca),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(housing_x,housing_y)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'housing dim red.csv')

# 3
dim = 5
pca = PCA(n_components=dim,random_state=10)
perm_x2 = pca.fit_transform(perm_x)

dim = 9
pca = PCA(n_components=dim,random_state=10)
housing_x2 = pca.fit_transform(housing_x)

run_clustering(out, perm_x2, perm_y, housing_x2, housing_y)
