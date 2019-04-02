## This project implements ID3 tree
* For cutting method, it’s using two criterias: 
To stop if the information gain is less than a value
to stop if we specify the maximum depth of our tree
* For pruning it removes a node if one of its branches has majority of possibilities (I’m using 70% as a limit).
This is the output of our run on the car dataset with our parameters for cutting method:

<img src="https://github.com/Obadah-H/ML-Course_MISIS-University/blob/master/3-ID3/Cut.png?raw=true">

the accuracy is: 78.76%
This is the output of our run on the car dataset with our parameters for pruning method:

<img src="https://github.com/Obadah-H/ML-Course_MISIS-University/blob/master/3-ID3/Prune.png?raw=true">

the accuracy is: 74.71%
The dataset is about rating cars according to their specs. Link: http://archive.ics.uci.edu/ml/datasets/Car+Evaluation
How to run it:
```
python3 (dataset) (Method) (HTML page)
```
method: for cut: 0 , for pruning: 1
for example:

```
    python3 id3.py car.csv 1 out
```

*Note: if we don't specify 'h', the default value is '0.9'*
