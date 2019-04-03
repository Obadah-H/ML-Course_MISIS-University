# PCA
PCA implemntation: Kaiser, broken cane and scree methods of choosing the number of components are considered.

Note: For scree method, it is supposed to be determined by a person after he checks the plot, and it’s determined when plot is not going down that much. Instead, to calculate it programmatically, I decided to go with the angle between two eignvalues percentage, when it’s less than a certain amount, stop. This figure explains my idea:

<img src="https://github.com/Obadah-H/ML-Course_MISIS-University/blob/master/5-PCA/pca.png?raw=true">
