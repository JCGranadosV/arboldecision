# Load libraries
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.preprocessing import LabelEncoder   

col_names = ['Id','SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']
# load dataset
irisDataset = pd.read_csv('Iris.csv')

d = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
td = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
irisDataset['Species'] = irisDataset['Species'].map(d)

features = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']

X = irisDataset[features]
y = irisDataset['Species']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X,y)

sepalLength = float(input("Ingresa la longitud del sepalo en centimetros: "))
sepalWidth = float(input("Ingresa el ancho del sepalo en centimetros: "))
petalLength = float(input("Ingresa la longitud del petalo en centimetros: "))
petalWidth = float(input("Ingresa el ancho del petalo en centimetros: "))

result = td[dtree.predict([[sepalLength, sepalWidth, petalLength,petalWidth]])[0]]

print("Result: ",result)

