import numpy as np

import pandas as pd

df=pd.read_csv('teste.txt', names=["x1", "x2", "x3"])

x1=np.array(df['x1'])

x2=np.array(df['x2'])

x3=np.array(df['x3'])

W1L=[0.02493853, -0.24167227, -0.241828, 0.49681843]

W2L=[ 0.00573999, -0.15488214, -0.15975031,  0.31740044]

W3L=[ 0.02493853, -0.24167227, -0.241828, 0.49681843]

W4L=[ 0.00263748, -0.13479583, -0.13170255,  0.26631591]

W5L=[0.05848758, -0.32604843, -0.33804397,  0.68670916]

W1=np.asarray(W1L).astype(float)
W2=np.asarray(W2L).astype(float)
W3=np.asarray(W3L).astype(float)
W4=np.asarray(W4L).astype(float)
W5=np.asarray(W5L).astype(float)

Y = np.zeros(len(x1))

for i in range(len(x1)):
    X = [-1,x1[i],x2[i],x3[i]]
    vecX = np.array(X).astype(float)
    u = np.dot(np.transpose(W1), vecX)
    if u < 0:
        Y[i] = -1
    else:
        Y[i] = 1

Classes=[]

for i in range(len(x1)):
    if Y[i]==1:
        Classes.append("Classe C1")
    else:
        Classes.append("Classe C2")

df.insert(3,"Y",Y)
df.insert(4,"Classification",Classes)
print(df)


