# KNN

## For normal KNN method:

is achieved by running method ‘calc_knn’. It will return the class of a point according to the standard KNN method. (the attached notebook has an example to find predict a point’s class using normal KNN).

## Kernels:

Two kernel have been applied programmatically to achieve kernels: Epanechnikov and Quartic. The default one is  Epanechnikov, but it can be changed in ‘calc_knn’ method.
In general, for kernels we have Parzen window (parameter ‘h’), which can be a static value or a dynamic value.
Running the code using the static configuration gives the output in folder ‘static’.
Running the code using the dynamic configuration gives output in the folder ‘dynamic’.
(a report is generated automatically).

## How to run it:
```
python.exe (dataset) (HTML page) (Dynamic/static) [h]
```
(Dynamic/static): True for dynamic, False for static
h (parzen window parameter): is an additional parameter which can be used only if we are using the static KNN
for example, for dynamic knn, html name is 'out', dataset name is 'waveform':
```
    python3 knn.py waveform.csv out True
```
if we want a static window:
```
    python3 knn.py waveform.csv out False 0.9
```
An output example to find optimal k for KNN:

<img src="https://github.com/Obadah-H/ML-Course_MISIS-University/blob/master/2-KNN/dynamic/exclude.png?raw=true">

*Note: if we don't specify 'h', the default value is '0.9'*
