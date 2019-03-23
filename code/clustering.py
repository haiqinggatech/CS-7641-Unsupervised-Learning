import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from time import clock
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from collections import defaultdict
from helpers import load_data, nn_layers, nn_reg, nn_iter, cluster_acc, myGMM, clusters
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import sys
import time
out = '../results/clustering/'
perm_x, perm_y, housing_x, housing_y = load_data() # perm, housing



# np.reshape(perm_y, 30000, order='F')

# raise Exception('Remove this line to run code')

SSE = defaultdict(dict) # some of squared errors
ll = defaultdict(dict) # log likelihood
acc = defaultdict(lambda: defaultdict(dict))
adjMI = defaultdict(lambda: defaultdict(dict))
km = kmeans(random_state=5)
gmm = GMM(random_state=5)

st = time.time()
print (len(clusters))
for k in clusters:
    km.set_params(n_clusters=k)
    gmm.set_params(n_components=k)
    km.fit(perm_x)
    gmm.fit(perm_x)
    SSE[k]['perm'] = km.score(perm_x)
    ll[k]['perm'] = gmm.score(perm_x) 

    acc[k]['perm']['Kmeans'] = cluster_acc(perm_y,km.predict(perm_x))
    acc[k]['perm']['GMM'] = cluster_acc(perm_y,gmm.predict(perm_x))
    adjMI[k]['perm']['Kmeans'] = ami(perm_y,km.predict(perm_x))
    adjMI[k]['perm']['GMM'] = ami(perm_y,gmm.predict(perm_x))

for k in clusters:
    km.set_params(n_clusters=k)
    gmm.set_params(n_components=k)

    km.fit(housing_x)
    gmm.fit(housing_x)
    SSE[k]['housing'] = km.score(housing_x)
    ll[k]['housing'] = gmm.score(housing_x)



    acc[k]['housing']['Kmeans'] = cluster_acc(housing_y,km.predict(housing_x))
    acc[k]['housing']['GMM'] = cluster_acc(housing_y,gmm.predict(housing_x))
    adjMI[k]['housing']['Kmeans'] = ami(housing_y,km.predict(housing_x))
    adjMI[k]['housing']['GMM'] = ami(housing_y,gmm.predict(housing_x))
    print(k, time.time()-st)
    
    
SSE = (-pd.DataFrame(SSE)).T
SSE.rename(columns = lambda x: x+' SSE (left)',inplace=True)
ll = pd.DataFrame(ll).T
ll.rename(columns = lambda x: x+' log-likelihood',inplace=True)
acc = pd.Panel(acc)
adjMI = pd.Panel(adjMI)


SSE.to_csv(out+'SSE.csv')
ll.to_csv(out+'logliklihood.csv')

print (acc.ix[:,:,'housing'])
acc.ix[:,:,'housing'].to_csv(out+'Housing acc.csv')
acc.ix[:,:,'perm'].to_csv(out+'Perm acc.csv')
adjMI.ix[:,:,'housing'].to_csv(out+'Housing adjMI.csv')
adjMI.ix[:,:,'perm'].to_csv(out+'Perm adjMI.csv')



#  

#5 in assingment is below

print(1)

grid ={'km__n_clusters':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_layers}
mlp = MLPClassifier(activation='relu',max_iter=nn_iter,early_stopping=True,random_state=5)
km = kmeans(random_state=5)
pipe = Pipeline([('km',km),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10)

gs.fit(perm_x,perm_y)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'Perm cluster Kmeans.csv')


print(2)

grid ={'gmm__n_components':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_layers}
mlp = MLPClassifier(activation='relu',max_iter=nn_iter,early_stopping=True,random_state=5)
gmm = myGMM(random_state=5)
pipe = Pipeline([('gmm',gmm),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(perm_x,perm_y)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'Perm cluster GMM.csv')




print(3)

grid ={'km__n_clusters':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_layers}
mlp = MLPClassifier(activation='relu',max_iter=nn_iter,early_stopping=True,random_state=5)
km = kmeans(random_state=5)
pipe = Pipeline([('km',km),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(housing_x,housing_y)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'Housing cluster Kmeans.csv')


print(4)

grid ={'gmm__n_components':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_layers}
mlp = MLPClassifier(activation='relu',max_iter=nn_iter,early_stopping=True,random_state=5)
gmm = myGMM(random_state=5)
pipe = Pipeline([('gmm',gmm),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(housing_x,housing_y)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'Housing cluster GMM.csv')

print(5)

# %% For chart 4/5
perm_x2D = TSNE(verbose=10,random_state=5).fit_transform(perm_x)
housing_x2D = TSNE(verbose=10,random_state=5).fit_transform(housing_x)

Perm2D = pd.DataFrame(np.hstack((perm_x2D,np.atleast_2d(perm_y).T)),columns=['x','y','target'])
Housing2D = pd.DataFrame(np.hstack((housing_x2D,np.atleast_2d(housing_y).T)),columns=['x','y','target'])

Perm2D.to_csv(out+'Perm2D.csv')
Housing2D.to_csv(out+'Housing2D.csv')


