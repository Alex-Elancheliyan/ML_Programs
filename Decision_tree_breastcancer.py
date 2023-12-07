import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load your dataset (replace 'your_dataset.csv' with the actual filename)
df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df = df.drop(['number'], axis=1)
X = np.array(df.drop(['Class'], axis=1))
y = np.array(df['Class'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree classifier
dt_classifier = DecisionTreeClassifier()

# Train the classifier on the training data
dt_classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = dt_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{classification_rep}")
new_features = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [8, 9, 10, 7, 7, 10, 8, 7, 1], [2, 1, 2, 1, 2, 1, 3, 1, 1],
                         [4, 1, 2, 1, 1, 2, 3, 1, 1]])
new_features = new_features.reshape(len(new_features), -1)
result = dt_classifier.predict(new_features)
col = lambda val: 0 if val == 2 else 1
for i, j in enumerate(result):
    l = lambda j: 0 if (j == 2) else 1
    plt.scatter(i, j, color=ListedColormap(('black', 'red'))(l(j)), label=j)
plt.show()