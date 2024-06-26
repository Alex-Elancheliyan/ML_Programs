import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from matplotlib.colors import ListedColormap
style.use('fivethirtyeight')
import math
from sklearn import model_selection, neighbors
import pandas as pd

# Sample code number,Clump Thickness,Uniformity of Cell Size,Uniformity of Cell Shape,Marginal Adhesion,Single Epithelial Cell Size,Bare Nuclei,Bland Chromatin,Normal Nucleoli,Mitoses,Class
def read():
    df = pd.read_csv('breast-cancer-wisconsin.data')
    df.replace('?', -99999, inplace=True)
    df = df.drop(['number'], axis=1)
    X = np.array(df.drop(['Class'], axis=1))
    y = np.array(df['Class'])
    return X, y

#Train The Model:
def train(X, y):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print('Accuracy on Test Data :     ', accuracy * 100)

def one_sample():
    print('One Sample Measure')
    example_measures = np.array([1,1,1,2,2,1,3,1,1])
    example_measures = example_measures.reshape(1, -1)
    prediction = clf.predict(example_measures)
    print('Prediction :     ', prediction)
def two_sample():
    print('Two Sample Measure')
    example_measures = np.array([[4, 12, 31, 10, 1, 2, 13, 25, 1], [4, 2, 1, 1, 1, 2, 3, 2, 1]])
    example_measures = example_measures.reshape(2, -1)
    prediction = clf.predict(example_measures)
    print('Prediction :     ', prediction)

def generic():
        print('Sample Measure Unknown - Generic Representation')
        example_measures = np.array(
            [[4, 2, 1, 1, 1, 2, 3, 2, 1], [8, 9, 10, 7, 7, 10, 8, 7, 1], [2, 1, 2, 1, 2, 1, 3, 1, 1],
             [4, 1, 2, 1, 1, 2, 3, 1, 1]])
        example_measures = example_measures.reshape(len(example_measures), -1)
        prediction = clf.predict(example_measures)
        print('Prediction :     ', prediction)

def euclidean():
    # Euclidean Distance theory
    plot1 = [1, 3]
    plot2 = [2, 5]
    euclidean_distance = math.sqrt((plot1[0] - plot2[0]) ** 2 + (plot1[1] - plot2[1]) ** 2)
    print('Euclidean distance :     ', euclidean_distance)
def plotting():
    new_features = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [8, 9, 10, 7, 7, 10, 8, 7, 1], [2, 1, 2, 1, 2, 1, 3, 1, 1],
                             [4, 1, 2, 1, 1, 2, 3, 1, 1]])
    new_features = new_features.reshape(len(new_features), -1)
    result = clf.predict(new_features)
    col = lambda val: 0 if val == 2 else 1
    for i, j in enumerate(result):
        l = lambda j: 0 if (j == 2) else 1
        plt.scatter(i, j, color=ListedColormap(('black', 'red'))(l(j)), label=j)

euclidean()
clf = neighbors.KNeighborsClassifier()
X, y = read()
train(X, y)
print('Validation Results :    ')
one_sample()
two_sample()
generic()
plotting()

