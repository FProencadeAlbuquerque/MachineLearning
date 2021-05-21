import numpy as np

import pandas as pd

df=pd.read_csv('treinamento.txt', names=["x1", "x2", "x3", "d"])

x1=np.array(df['x1'])

x2=np.array(df['x2'])

x3=np.array(df['x3'])

d=np.array(df['d'])

np.random.seed(42)

W=np.random.normal(0,0.1,4)

taxa_apr=0.01

epocas=0

Erro=1

print("O vetor inicial W é \n ")
print(W, "\n")

while Erro==1:
    Erro=0
    Y=np.zeros(len(x1))
    for i in range(len(x1)):
        X=[-1, x1[i],x2[i],x3[i]]
        vecX=np.array(X)
        u=np.dot(np.transpose(W),vecX)
        if u<0:
            Y[i]=-1
        else:
            Y[i]=1
        if Y[i] != d[i]:
            W=W+np.dot(taxa_apr*(d[i]-Y[i]),X)
            Erro=1
    epocas+=1

print("A quantidade de epocas utilizadas foi %d \n" %(epocas))
print("O vetor final W é \n ")
print(W, "\n")
print("A saida para o modelo foi")
print(Y,"\n" )
print("As saidas para treinamento eram \n")
print(d)
