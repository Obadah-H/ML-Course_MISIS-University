import matplotlib.pyplot as plt
import math
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
np.set_printoptions(suppress=True)
data = np.genfromtxt(sys.argv[1], dtype=float,delimiter=',')
X = data[:,0:-1]
X_std = StandardScaler().fit_transform(X)
n=len(X[0])
mean_vec = np.mean(X_std, axis=0)
#print(X_std.shape[0]-1)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
#print('Covariance matrix is \n%s' %cov_mat)
comment=""
methods=[0,0,0]
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_vals.sort()
eig_vals[:]=eig_vals[::-1]
eig_vals=np.abs(eig_vals)
#eig_pairs.sort()
#eig_pairs.reverse()

#print('Eigenvalues:')
values = np.zeros(len(X[0]))
counter=0
for i in eig_vals:
    #print(i[0])
    values[counter]=i
    counter+=1
#yy=eig_pairs
xx = np.empty(len(X[0]), dtype='object')
for i in range(len(xx)):
    xx[i]=str(i+1)

#Kaiser
selected=[]
for i in range (len(values)):
    if values[i]>1:
        selected=np.append(selected,i)
#print("Kaiser: Selected components:",selected.astype(int)+1)
a="Kaiser: Selected components:"+str(selected.astype(int)+1)
comment+="Kaiser method: " + str(int(selected[-1])+1)+" selected components\n"
methods[0]=int(selected[-1])+1
#broken cane
l = np.zeros(n)
#print(values)
for i in range(n):
    k=i+1
    tmp=0
    for j in range(k,n+1):
        tmp=tmp+1/j
    l[i] = (1/n) * tmp
#print(l)
v=np.copy(values)
for i in range(len(v)):
    v[i]=v[i]/n
#print(v)
selected=[]
for i in range(len(v)):
    if(v[i]>l[i]):
        selected=np.append(selected,i)
    else:
        break
#print("Broken cane: Selected components:",selected.astype(int)+1)
b="Broken cane: Selected components:"+str(selected.astype(int)+1)
comment+="Broken cane method: " + str(i)+" selected components\n"
methods[1]=i
#scree
angle = 30
step=2
if n> 100:
    step=110
selected=[]
math.sin(np.deg2rad(angle))
for i in range(len(values)-1):
    h=values[i]-values[min(i+step , len(values)-1)]
    if h/np.sqrt(1+h**2) < math.sin(np.deg2rad(angle)):
        #print("Scree: Selected components:",list(range(1,i+2)))
        break
c="Scree: Selected components:"+str(list(range(1,i+2)))
comment+="Scree method: " + str(i+1 )+" selected components\n"
methods[2]=i+1
plt.xlabel('PCA')
plt.ylabel('%')
#plt.ylim(0,100)
plt.plot(xx, (values/len(X[0]))*100,marker='o', color='g')
plt.text(n/3, 0.8*((values[0]/len(X[0]))*100),comment)
plt.savefig("plot.png")
plt.close()

xx=["Kaiser: "+str(methods[0]) , "Broken cane: "+str(methods[1]) , "Scree: "+str(methods[2])]
yy=np.zeros(3)
for i in range(3):
    yy[i]=(sum(values[:methods[i]])/len(X[0]))*100
plt.ylim(0,100)
plt.bar(xx, yy)
plt.xlabel('Method')
plt.ylabel('%')
plt.title("Percentage")
plt.savefig("percentage.png")
html_str = """
<style>
table, th, td {
  border: 1px solid black;
}</style>
<center><h3>PCA</h3></center>
<img src='plot.png'></br>
This plot shows the percentage covered by each method:<br/>
<img src='percentage.png'><br />
Kaiser: """+str(yy[0])+"""%<br/>
Broken cane: """+str(yy[1])+"""%<br/>
Scree: """+str(yy[2])+"""%"""

Html_file= open(sys.argv[2]+".html","w")
Html_file.write(html_str)
Html_file.close()
