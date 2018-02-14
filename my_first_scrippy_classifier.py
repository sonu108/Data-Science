from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import random as rd
from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)


class scrippyclf():
    def __init__(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train
    def fit(self,x_train,y_train):
        pass
    def predict(self,x_test):
        score = []
        for i in (x_test):
            label = self.closest(i)
            score.append(label)
        print(score)
        return score
    def closest(self,row):
        dizt = euc(row,self.x_train[0])
        idx = self.y_train[0]
        for i in range(1,len(x_test)-1):
            d = euc(row,self.x_train[i])
            if(d<dizt):
                dizt = d
                idx = self.y_train[i]
        return idx
        
iris = load_iris()
x = iris.data
y = iris.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.4)
#knn = KNeighborsClassifier(5)
knn = scrippyclf(x_train,y_train)
knn.fit(x_train,y_train)
predict = knn.predict(x_test)
acc = accuracy_score(y_test,predict)
print(acc)
