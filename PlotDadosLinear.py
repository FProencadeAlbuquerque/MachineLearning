import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('svm1.txt', names=["x1", "x2", "y"])

x1=np.array(df['x1'])

x2=np.array(df['x2'])

y=np.array(df['y'])

X=[]
Y=[]
X1=[]
Y1=[]

for i in range(len(y)):
    if y[i]==1:
        X.append(x1[i])
        Y.append(x2[i])
    else:
        X1.append(x1[i])
        Y1.append(x2[i])

plt.scatter(X,Y,s=100,c='orange',label='Valores +1')
plt.scatter(X1,Y1,s=100,c='blue',label='Valores -1')

ax = plt.gca()
ax.tick_params(axis = 'both', which = 'major', labelsize = 22)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 22)

csfont = {'fontname':'Times New Roman','size': 28}

plt.title('Divisão entre as classes',**csfont)
plt.xlabel('x',**csfont)
plt.ylabel('y',**csfont)

plt.legend()

plt.show()




