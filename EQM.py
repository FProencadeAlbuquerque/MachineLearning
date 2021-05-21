import pandas as pd

from sklearn.neural_network import MLPClassifier

from sklearn import preprocessing

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

df=pd.read_csv('treina2.txt', names=["x1", "x2", "x3", "x4", "y1", "y2", "y3"])
df2=pd.read_csv('teste.txt', names=["x1", "x2", "x3", "x4", "y1", "y2", "y3"])

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


scaler = preprocessing.StandardScaler().fit(X)
x1_scaled = scaler.transform(X)

scaler_t = preprocessing.StandardScaler().fit(Xt)
x1t_scaled = scaler.transform(Xt)

clf = MLPClassifier(solver="adam",activation='logistic', hidden_layer_sizes=(15),tol=1e-6,learning_rate_init=0.1,max_iter=1800,random_state=rng)

scores_train = []

epocas=[]

Predito=[]

EQM=[]

N_EPOCHS = 450
N_CLASSES = Y

epoch = 0
while epoch < N_EPOCHS:
    clf.partial_fit(x1_scaled,Y,classes=N_CLASSES)
    Predito=clf.predict(x1_scaled)
    EQM.append(mean_squared_error(Y, Predito))
    epocas.append(epoch)
    epoch += 1


ax = plt.gca()
ax.tick_params(axis = 'both', which = 'major', labelsize = 20)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 20)

plt.plot(epocas, EQM, color='green', linewidth = 3)

csfont = {'fontname':'Times New Roman','size': 22}

plt.xlabel('Número de épocas',**csfont)

plt.ylabel('Erro quadrático médio',**csfont)

plt.title('Avaliação do EQM em função do número de épocas',**csfont)

plt.show()















