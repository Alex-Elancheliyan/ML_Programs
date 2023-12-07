import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Fashion-MNIST dataset
fashion_mnist = datasets.fetch_openml('Fashion-MNIST', version=1, cache=True)


# Convert data to NumPy array and ensure it's contiguous
X = np.array(fashion_mnist.data, order='C')
y = fashion_mnist.target

# Assuming X is the NumPy array
df = pd.DataFrame(X, columns=[f'pixel_{i}' for i in range(X.shape[1])])
print(df.head(10))

#Split data set into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a k-Nearest Neighbors (k-NN) classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Train the classifier on the training data
knn_classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = knn_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{classification_rep}")

# Visualize some predictions
fig, axes = plt.subplots(5, 20, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(28, 28), cmap='gray')
    ax.set_title(f'A : {y_test.iloc[i]}, \n P : {y_pred[i]}')
    ax.axis('off')
plt.show()