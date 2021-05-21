import numpy as np

import pandas as pd

from sklearn import svm

df=pd.read_csv('svm2_treinamento.txt', names=["x1", "x2", "y"])

df2=pd.read_csv('svm2_teste.txt', names=["x1", "x2", "y"])

x1=np.array(df['x1'])

x2=np.array(df['x2'])

y=np.array(df['y'])

x1t=np.array(df2['x1'])

x2t=np.array(df2['x2'])

yt=np.array(df2['y'])


X=[]

Y=[]

Xt=[]

Yt=[]

for i in range(len(x1)):
    X.append([x1[i],x2[i]])
    Y.append(y[i])

for j in range(len(x1t)):
    Xt.append([x1t[j],x2t[j]])
    Yt.append(yt[j])

C=100

gamma=10

clf = svm.SVC(kernel='rbf', C=C,gamma=gamma)

clf.fit(X, Y)

predito=clf.predict(Xt)

acertos=0

for k in range(len(Yt)):
    if predito[k]==Yt[k]:
        acertos+=1

Porcentagem=(acertos/len(Yt))*100

aux='%'

print('A porcentagem de acerto para gamma = %.3f foi de %.3f %s' %(gamma,Porcentagem,aux))

