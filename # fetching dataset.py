from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# Fetch dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
x, y = mnist['data'], mnist['target']

# Convert labels to integers
y = y.astype(np.int8)

# Display multiple sample digits
def plot_sample_digits(data, labels, sample_size=10):
    plt.figure(figsize=(10, 1))
    for index in range(sample_size):
        plt.subplot(1, sample_size, index + 1)
        plt.imshow(data[index].reshape(28, 28), cmap=plt.cm.binary, interpolation='nearest')
        plt.title(f"{labels[index]}")
        plt.axis("off")
    plt.show()

plot_sample_digits(x, y)

# Prepare the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train a logistic regression classifier
clf = LogisticRegression(max_iter=1000, solver='lbfgs', tol=0.1)
clf.fit(x_train, y_train)

# Evaluate the classifier
y_pred = clf.predict(x_test)
print("Classification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# Cross-validation
cross_val_scores = cross_val_score(clf, x_train, y_train, cv=3, scoring="accuracy")
print("Cross-validation accuracy:", cross_val_scores.mean())

# Display correctly classified images
correctly_classified_indices = np.where(y_pred == y_test)[0]

def plot_classified_images(data, labels_true, labels_pred, indices, title, sample_size=5):
    plt.figure(figsize=(10, 2))
    for index in range(sample_size):
        plt.subplot(1, sample_size, index + 1)
        plt.imshow(data[indices[index]].reshape(28, 28), cmap=plt.cm.binary, interpolation='nearest')
        plt.title(f"T:{labels_true[indices[index]]}\nP:{labels_pred[indices[index]]}")
        plt.axis("off")
    plt.suptitle(title)
    plt.show()

plot_classified_images(x_test, y_test, y_pred, correctly_classified_indices, "Correctly Classified Images")

# Display misclassified images
misclassified_indices = np.where(y_pred != y_test)[0]
plot_classified_images(x_test, y_test, y_pred, misclassified_indices, "Misclassified Images")

# Plot some specific samples
def plot_specific_samples(data, indices, title):
    plt.figure(figsize=(10, 1))
    for i, index in enumerate(indices):
        plt.subplot(1, len(indices), i + 1)
        plt.imshow(data[index].reshape(28, 28), cmap=plt.cm.binary, interpolation='nearest')
        plt.title(f"Index {index}")
        plt.axis("off")
    plt.suptitle(title)
    plt.show()

# Display specific sample digits
specific_indices = [36001, 100, 200, 300, 400]
plot_specific_samples(x, specific_indices, "Specific Sample Digits")


