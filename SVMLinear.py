import numpy as np

import pandas as pd

from sklearn import svm

df=pd.read_csv('svm1.txt', names=["x1", "x2", "y"])

x1=np.array(df['x1'])

x2=np.array(df['x2'])

y=np.array(df['y'])

X=[]

Y=[]

for i in range(len(x1)):
    X.append([x1[i],x2[i]])
    Y.append(y[i])

C=1

clf = svm.SVC(kernel='linear', C=C)

clf.fit(X, Y)

print(clf.coef_)
print(clf.intercept_)