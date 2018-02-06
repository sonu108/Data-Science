from sklearn.datasets import load_iris
import numpy as np
data = np.zeros(shape=[151,6],dtype='S15')
X = np.zeros(shape=[150,4],dtype=float)
y = np.zeros(shape=[150,1])
target = {'Iris-virginica':2,'Iris-versicolor':1,'Iris-setosa':0}
print(y.size)
import csv
with open('Iris.csv','rb') as csvfile:
    data1 = csv.reader(csvfile)
    next(data1)
    for i in data1:
        for j in range(len(i)):
            data[int(i[0])][j] = i[j]
X = data[:,1:5]
y = 
print(data)
