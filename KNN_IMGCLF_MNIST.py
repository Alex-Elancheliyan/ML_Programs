import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import model_selection, neighbors
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the digits dataset
digits = datasets.load_digits()

# Flatten the images and split the dataset
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.2, random_state=42)

# Create a Support Vector Machine (SVM) classifier
clf = neighbors.KNeighborsClassifier()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

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
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
    ax.set_title(f'A : {y_test[i]}, \n P : {y_pred[i]}')
    ax.axis('off')
plt.show()