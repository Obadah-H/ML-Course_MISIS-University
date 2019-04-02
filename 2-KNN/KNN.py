import math
import sys
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from scipy.spatial import distance
np.set_printoptions(suppress=True)
data = np.genfromtxt(sys.argv[1], delimiter=',')
new_data=np.zeros(len(data[0]))
classes = int(np.amax(data[:,-1]) + 1)
_=np.zeros(classes)
for i in range(len(data)):
    if _[int(data[i][-1])]<100:
        new_data=np.c_[new_data,data[i]]
    _[int(data[i][-1])]+=1
new_data=new_data.transpose()
data=np.delete(new_data,0,axis=0)
X = data[:,0:-1]
y = data[:,-1:].transpose()[0].astype(int)
def normalize(_x):
    return (_x-_x.min(axis=0))/(_x.max(axis=0)-_x.min(axis=0))
X=normalize(X)
Point = [1.5,3.0,1.4,2.5,3,3.1,-1.1,0,1.2,2.4,3.1,1,3,5,3.0,1.4,2.6,2.3,2.1,-3.1,2]
classes = np.amax(y) + 1
class KNN:
    def __init__(self,X,y):
        self.X=X
        self.y=y
        self.max_k=200
        self.dynamic=sys.argv[3]
        self.h=0.9
        if len(sys.argv)==5:
            self.h=sys.argv[4]
    def calc_destances(self,combined,point):
        for d in range(len(combined)):
            distance = 0
            for j in range(len(X[0])):
                distance = distance + (combined[d][j] - point[j])**2
            distance = math.sqrt(distance)
            combined[d,-1] = distance
        return combined[combined[:,-1].argsort()]
        
    def calc_prep(self,point=Point):
        X=self.X
        y=self.y
        combined = np.c_[ X, y , np.zeros(len(y))]
        combined = combined[combined[:,-2].argsort()] #sort by class
        combined_distances = self.calc_destances(combined,point)
        return combined_distances
        
    def exclude(self):
        errors=np.zeros(self.max_k , dtype=int)
        for k in range(self.max_k):
            for i in range(len(self.X)):
                X=np.delete(self.X, i,axis=0)
                y=np.delete(self.y, i,axis=0)
                point=self.X[i]
                target_class=self.y[i]
                combined = np.c_[ X, y , np.zeros(len(y))]
                combined_distances = self.calc_destances(combined,point)
                point_class=self.calc_knn(k,combined_distances)
                if point_class!=target_class:
                    errors[k]=errors[k]+1
        print(errors[1:])
        plt.plot(np.arange(self.max_k-1)+1,errors[1:])
        plt.savefig("exclude.png")
        s = "The smallest value of k is: "+str(errors.argmin(axis=0))+ " and number of errors is: "+str(errors.min())
        print(s)
        return s
    
    def Epanechnikov(self,u,h=0.9):
        return 3*(1-(u/h)**2)/4
    def Quartic(self,u,h=0.9):
        return 15*(1-(u/h)**2)**2/16

    def calc_knn(self,k,combined_distances,kernel=True):
        number_of_points=np.zeros(classes)
        step=1
        _d=combined_distances[k][-1] #distance for k+1
        for i in range(k):
            if kernel:
                if self.dynamic=='True':
                    step=self.Epanechnikov(combined_distances[i][-1],_d)
                else:
                    step=self.Epanechnikov(combined_distances[i][-1],self.h)
            number_of_points[int(combined_distances[i][-2])]+=step #Epanechnicov(d) or Epanechnicov(d)*d
        point_class=number_of_points.argmax(axis=0)
        return point_class
_knn = KNN(X,y)
info=_knn.exclude()
html_str = """
<center><h3>KNN</h3></center>
<img src='exclude.png'>
<br/>"""+ str(info)+"""
"""

Html_file= open(sys.argv[2]+".html","w")
Html_file.write(html_str)
Html_file.close()
