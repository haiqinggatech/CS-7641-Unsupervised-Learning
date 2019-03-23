import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

from helpers import load_data, nn_layers, nn_reg, nn_iter


out = '../results/base/'
df1_x, df1_y, df2_x, df2_y = load_data()

grid ={'NN__alpha': nn_reg,'NN__hidden_layer_sizes': nn_layers}

mlp = MLPClassifier(activation='relu',max_iter=nn_iter,early_stopping=True,random_state=5)
pipe = Pipeline([('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(df1_x,df1_y)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'creditcard.csv',index=False)


mlp = MLPClassifier(activation='relu',max_iter=nn_iter,early_stopping=True,random_state=5)
pipe = Pipeline([('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(df2_x,df2_y)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'income.csv',index=False)