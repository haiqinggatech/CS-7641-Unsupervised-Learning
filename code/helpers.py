import numpy as np
import pandas as pd
from numpy import array
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import sklearn.model_selection as ms
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


from collections import Counter
from sklearn.metrics import accuracy_score as acc
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.base import TransformerMixin,BaseEstimator
import scipy.sparse as sps
from scipy.linalg import pinv
from collections import defaultdict
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from time import clock
from sklearn.metrics import adjusted_mutual_info_score as ami

path = '../data/'


nn_layers = [(100,), (50,), (50, 50)]
nn_reg = [10**-x for x in range(1,5)]
nn_iter = 1500

clusters =  [2,5,10,15,20,25,30,35,40,50, 60, 70]
dims = [2,3, 4, 5, 6, 7,] # 8, 9, 10,15,20,25,30,35,40,45,50,55,60]
dims_big = [2,3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def load_data():
  np.random.seed(0)


  data2 = pd.read_excel(open(path+'default of credit card clients.xls','rb'), sheetname='Data', skiprows=1)

  data1 = pd.read_csv(path+'winequality-data.csv')

 
  data2_x = data2.iloc[:,:24]
  data2_y = data2.iloc[:,24]

#  print (adult.head())
  X = data1.iloc[:,:11]
  y = data1.iloc[:,11]


  data2_x = StandardScaler().fit_transform(data2_x)
  X = StandardScaler().fit_transform(X)

  return (data2_x, data2_y, X, y)


def cluster_acc(Y,clusterLabels):
    assert (Y.shape == clusterLabels.shape)
    pred = np.empty_like(Y)
    for label in set(clusterLabels):
        mask = clusterLabels == label
        sub = Y[mask]
        target = Counter(sub).most_common(1)[0][0]
        pred[mask] = target
#    assert max(pred) == max(Y)
#    assert min(pred) == min(Y)    
    return acc(Y,pred)


class myGMM(GMM):
    def transform(self,X):
        return self.predict_proba(X)


def run_clustering(out, data2_x, perm_y, adultX, adultY):
  SSE = defaultdict(dict)
  ll = defaultdict(dict)
  acc = defaultdict(lambda: defaultdict(dict))
  adjMI = defaultdict(lambda: defaultdict(dict))
  km = kmeans(random_state=5)
  gmm = GMM(random_state=5)

  st = clock()
  for k in clusters:
      km.set_params(n_clusters=k)
      gmm.set_params(n_components=k)
      km.fit(data2_x)
      gmm.fit(data2_x)
      SSE[k]['perm'] = km.score(data2_x)
      ll[k]['perm'] = gmm.score(data2_x)    
      acc[k]['perm']['Kmeans'] = cluster_acc(perm_y,km.predict(data2_x))
      acc[k]['perm']['GMM'] = cluster_acc(perm_y,gmm.predict(data2_x))
      adjMI[k]['perm']['Kmeans'] = ami(perm_y,km.predict(data2_x))
      adjMI[k]['perm']['GMM'] = ami(perm_y,gmm.predict(data2_x))
      
      km.fit(adultX)
      gmm.fit(adultX)
      SSE[k]['housing'] = km.score(adultX)
      ll[k]['housing'] = gmm.score(adultX)
      acc[k]['housing']['Kmeans'] = cluster_acc(adultY,km.predict(adultX))
      acc[k]['housing']['GMM'] = cluster_acc(adultY,gmm.predict(adultX))
      adjMI[k]['housing']['Kmeans'] = ami(adultY,km.predict(adultX))
      adjMI[k]['housing']['GMM'] = ami(adultY,gmm.predict(adultX))
      print(k, clock()-st)
      
      
  SSE = (-pd.DataFrame(SSE)).T
  SSE.rename(columns = lambda x: x+' SSE (left)',inplace=True)
  ll = pd.DataFrame(ll).T
  ll.rename(columns = lambda x: x+' log-likelihood',inplace=True)
  acc = pd.Panel(acc)
  adjMI = pd.Panel(adjMI)


  SSE.to_csv(out+'SSE.csv')
  ll.to_csv(out+'logliklihood.csv')
  acc.ix[:,:,'housing'].to_csv(out+'Housing acc.csv')
  acc.ix[:,:,'perm'].to_csv(out+'Perm acc.csv')
  adjMI.ix[:,:,'housing'].to_csv(out+'Housing adjMI.csv')
  adjMI.ix[:,:,'perm'].to_csv(out+'Perm adjMI.csv')

def pairwiseDistCorr(X1,X2):
    assert X1.shape[0] == X2.shape[0]
    
    d1 = pairwise_distances(X1)
    d2 = pairwise_distances(X2)
    return np.corrcoef(d1.ravel(),d2.ravel())[0,1]

    
def aveMI(X,Y):    
    MI = MIC(X,Y) 
    return np.nanmean(MI)
    
  
def reconstructionError(projections,X):
    W = projections.components_
    if sps.issparse(W):
        W = W.todense()
    p = pinv(W)
    reconstructed = ((p@W)@(X.T)).T # Unproject projected data
    errors = np.square(X-reconstructed)
    return np.nanmean(errors)
    
class ImportanceSelect(BaseEstimator, TransformerMixin):
    def __init__(self, model, n=1):
         self.model = model
         self.n = n
    def fit(self, *args, **kwargs):
         self.model.fit(*args, **kwargs)
         return self
    def transform(self, X):
         return X[:,self.model.feature_importances_.argsort()[::-1][:self.n]]
