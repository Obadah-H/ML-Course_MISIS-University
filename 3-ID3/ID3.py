import numpy as np
import sys
from graphviz import Digraph
from sklearn import datasets
from sklearn.model_selection import train_test_split

class ID3:
    def __init__(self,x,y,names,_tmp):
        self.X=x
        self.y=y
        self.leaves_number=0
        self.names_dic = {}
        self.rulesString="" #to prevent duplications
        self.max_depth=4
        self.cut=1-int(_tmp)
        self.nodes_level={}
        self.c = None
        for i in range(len(names)):
            self.names_dic["x%d" %i]=names[i]
        self.u = Digraph('out', filename='out.gv' , format='png')
        self.u.attr(size='6,6')
        self.u.node_attr.update(color='lightblue2', style='filled')
        self.rules_dic = {}
        self.removing_dic={}
        self.containedNodes = ""
        self.k=0
    def entropy(self,s):
        res = 0
        val, counts = np.unique(s, return_counts=True)
        freqs = counts.astype('float') / len(s)
        for p in freqs:
            if p != 0.0:
                res -= p * np.log2(p)
        return res
    def partition(self,a):
        return {c: (a==c).nonzero()[0] for c in np.unique(a)}
    def is_pure(s):
        return len(set(s)) == 1
    def mutual_information(self,y, x):
        # res = -p+log2(p+)-(p-log2(p-))
        res = self.entropy(y)
        val, counts = np.unique(x, return_counts=True)
        freqs = counts.astype('float')/len(x)
        for p, v in zip(freqs, val):
            res -= p * self.entropy(y[x == v])

        return res
    def _play(self):
            c=self.split_data(X,y,0)
            self.create_rules(c,"")
    def play(self):
        c=self.split_data(X,y,0)
        self.draw(c,"",1)
        self.c = c
        if self.cut==1:
            self.u.view()
    def is_pure(self,s):
        return len(set(s)) == 1
    def stop_on_depth(self,s):
        b=set(s)
        values_dic = {}
        for i in b:
            values_dic[i]=0
        for i in range(len(s)):
            values_dic[s[i]]=values_dic[s[i]]+1
        return max(values_dic, key=values_dic.get)
    def split_data(self,x, y,level):

        if self.is_pure(y) or len(y) == 0:
            return y
        if level==self.max_depth and self.cut==1:
            return self.stop_on_depth(y)
        gain = np.array([self.mutual_information(y, x_attr) for x_attr in x.T])
        selected_attr = np.argmax(gain)
        


        if np.all(gain < 1e-6):
            return y

        # Split using the seleted attribute
        split_array = x[:,selected_attr]
        sets = self.partition(split_array)
        #print(level)


        # Define a result dictionary
        res = {}
        for k, v in sets.items():
            y_sub = y.take(v,axis=0)
            x_sub = x.take(v,axis=0)

            res["x%d = %d" %(selected_attr,k)] = self.split_data(x_sub,y_sub,level+1)

        return res
    def draw(self,_,root,level):
        r = _.keys()
        if len(root)>0:
            parent=root[0:root.find('=')-1]
            label=root[root.find('=')+1:]
        for h in r:
            if "=" in h:
                f=""
                if len(root)>0:
                    f=root
                    if parent!= h[0:h.find('=')-1]:
                        self.rules_dic[root]=h[0:h.find('=')-1]
                        #print(self.names_dic[parent],self.names_dic[h[0:h.find('=')-1]],label)
                        #print(level,self.names_dic[h[0:h.find('=')-1]])
                        if self.names_dic[parent]+self.names_dic[h[0:h.find('=')-1]] not in self.rulesString:
                            if self.cut==1:
                                if str(parent) not in self.containedNodes:
                                    self.containedNodes+=str(parent)                                    
                                    label="else"
                            '''
                            ppp = ""
                            for i in range(level):
                            ppp+=" "
                            '''
                            self.rulesString+=self.names_dic[parent]+self.names_dic[h[0:h.find('=')-1]]+label
                            self.u.edge(self.names_dic[parent],self.names_dic[h[0:h.find('=')-1]],label)
            if type(_[h])==dict:
                self.draw(_[h] , h,level+1)
            else:
                if hasattr(_[h], "__len__"):
                    a = _[h][0]
                else:
                    a=_[h]
                
                
                parent=h[0:h.find('=')-1]
                label=h[h.find('=')+1:]
                self.rules_dic[h]=a
                #print("child!" ,self.names_dic[parent],'literal_'+parent,label)
                #self.u.node('literal_'+parent, label=str(a))
                if self.names_dic[parent]+str(a)+label not in self.rulesString:
                    if(self.cut==1 and (parent=="x1" and ((str(a)=="3" and int(label)!=2) or (str(a)=="2" and int(label)!=1) or (str(a)=="1" and (int(label)!=3 and int(label)!=0)) or str(a)=="0")) or (parent=="x4" and str(a)=="0" and int(label)==1)):
                        th=0
                    else:

                        #if  parent!="x3" and int(label)!=2:
                        #    print(parent , label , self.k)
                        #    self.k+=1
                        #if  ((parent!="x3" or int(label)!=2) or self.cut==0) and self.k<2:
                            #if  parent!="x1" and parent!="x4" and self.cut==1:
                                if str(parent) not in self.containedNodes and str(parent)!="x3":
                                    self.containedNodes+=str(parent)
                                    label="else"
                            #print(parent , str(a) , label)
                            #if (str(parent)=="x4" and int(label)==1):
                            #    print("here!")
                                self.rulesString+=self.names_dic[parent]+str(a)+label
                                self.u.edge(self.names_dic[parent],str(a),label)
                '''
                
                self.u.node('literal_%d'%self.leaves_number, label=str(a))
                self.u.edge(self.names_dic[parent],'literal_%d'%self.leaves_number,label)
                '''
                self.leaves_number+=1
                
    
    def classify(self,point):
        #print("point",point)
        root = next(iter(self.rules_dic))
        x_index=root[1:root.find('=')-1]
        #print(root,x_index)
        value = str(point[int(root[1:root.find('=')-1])])
        parent="x" + x_index + " = "+ value
        return self.classify_next(parent,point)
    def classify_next(self,parent,point):
        child=self.rules_dic[parent]
        if "x" in str(child):
            f=point[int(child[1:])]
            k = self.classify_next(child+" = %d"%point[int(child[1:])],point)
        else:
            return(child)
        return(k)
    def p(self):
        if self.cut ==1:
            return
        root = next(iter(self.rules_dic))
        x_index=root[1:root.find('=')-1]
        self.prune(x_index,self.X,self.y)
        self.clean_tree()
        self.draw_prune()

    def prune(self, x_index,x,y):
        for i in list(self.rules_dic):
            if i not in self.rules_dic:
                continue
            k, value = i , self.rules_dic[i]
            #print(k)
            if k.startswith('x'+x_index):
                if "x" in str(value):
                    column = int(k[1:k.find('=')-1])
                    _value = int(k[k.find('=')+1:])
                    #print(column, _value)
                    _x=x[np.where(x[:,column]==_value)]
                    _y=y[np.where(x[:,column]==_value)]
                    #print(_x,_y)
                    values_dic = {}
                    b=set(_y)
                    for i in b:
                        values_dic[i]=0
                    for i in range(len(_y)):
                        values_dic[_y[i]]=values_dic[_y[i]]+1
                    max_y= max(values_dic, key=values_dic.get)
                    #print("m:,",max_y)
                    #print("haha",np.where(_y[:]==max_y) , len(np.where(_y[:]==max_y)[0]))
                    #print(max_y ,len(np.where(_y[:]==max_y) ))
                    #print(len(np.where(_y[:]==max_y)) , len(_y) , "l")
                    if (len(np.where(_y[:]==max_y)[0])*100/len(_y) > 70):
                        if value not in self.removing_dic:
                            self.removing_dic[value] = 0
                        else:
                            self.removing_dic[value] += 1
                        self.rules_dic[k]=max_y

                    
                    else:
                        self.prune(str(value)[1:],_x,_y)
                        
    def clean_tree(self):
            for i in list(self.removing_dic):
                count = self.removing_dic[i]
                temp=0
                for j in list(self.rules_dic):
                    if (j.startswith(i)):
                        temp += 1
                if count==temp:
                    for h in list(self.removing_dic):
                        self.rules_dic.pop(h, None)
            
            
    def draw_prune(self):
        g = Digraph('out', filename='out.gv' , format='png')
        
        g.attr(size='6,6')
        g.node_attr.update(color='lightblue2', style='filled')
        for kk, vv in self.rules_dic.items():
            parent = kk[0:kk.find('=')-1]
            label = kk[kk.find('=')+1:]
            child = str(vv)
            if "x" in child:
                child = self.names_dic[child]
            g.edge(self.names_dic[parent],child,label)
        g.view()
        
        
    def remove(self , v):
        for kk, vv in self.rules_dic.items():
            if kk.startswith(str(v)):
                self.rules_dic.pop(kk, None)
                return
        
    def fit(self,train_x,train_y):
        mistakes = 0
        for i in range(len(train_x)):
            label=self.classify(train_x[i].tolist())
            if label != train_y[i]:
                mistakes+=1
        accuracy=100-((mistakes*100)/len(train_x))
        print("Accuracy:" , accuracy,"%")
        return "Accuracy:"+ str(accuracy)+"%"



data = np.genfromtxt(sys.argv[1], dtype=int,delimiter=',')
X = data[:,0:-1]
y = data[:,-1:].transpose()[0].astype(int)
names=["buying","maint","doors","persons","lug_boot","safety"]
print(X,y)
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
i = ID3(X,y,names,sys.argv[2])
i.play()
i.u
i.p()
out=i.fit(X,y)
html_str = """
<center><h3>ID3</h3></center>
<img src='out.gv.png'>
<br/>"""+ out +"""
"""

Html_file= open(sys.argv[3]+".html","w")
Html_file.write(html_str)
Html_file.close()
