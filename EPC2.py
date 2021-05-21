import pandas as pd

from sklearn.neural_network import MLPClassifier

from sklearn import preprocessing

import numpy as np

df=pd.read_csv('treina2.txt', names=["x1", "x2", "x3", "x4", "y1", "y2", "y3"])
df2=pd.read_csv('teste2.txt', names=["x1", "x2", "x3", "x4", "y1", "y2", "y3"])

x1,x1t=df['x1'],df2['x1']

x2,x2t=df['x2'],df2['x2']

x3,x3t=df['x3'],df2['x3']

x4,x4t=df['x4'],df2['x4']

y1,y1t=df['y1'],df2['y1']

y2,y2t=df['y2'],df2['y2']

y3,y3t=df['y3'],df2['y3']

X=[]
Xt=[]
Y=[]
auxX=[1,2,3,4]
auxY=[1,2,3]

rng = np.random.RandomState(46)

for i in range(len(x1)):
    auxX=[x1[i],x2[i],x3[i],x4[i]]
    auxY=[y1[i],y2[i],y3[i]]
    X.append(auxX)
    Y.append(auxY)

for i in range(len(x1t)):
    auxXt=[x1t[i],x2t[i],x3t[i],x4t[i]]
    Xt.append(auxXt)

#### Feature scaling #######

scaler = preprocessing.StandardScaler().fit(X)
x1_scaled = scaler.transform(X)

scaler_t = preprocessing.StandardScaler().fit(Xt)
x1t_scaled = scaler.transform(Xt)

################################

clf = MLPClassifier(solver="adam",activation='logistic', hidden_layer_sizes=(15),tol=1e-6,learning_rate_init=0.1,max_iter=1800,random_state=rng)

T=clf.fit(x1_scaled,Y)

Yteste=clf.predict(x1t_scaled)

Yteste_prob=clf.predict_proba(x1t_scaled)

print(Yteste_prob)

print(Yteste)


















