import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Logistic Regression is a linear model for classification that predicts probabilities.
#Despite its name, it's used for binary classification (can be extended to multiclass).

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


#Predicting a new result
print(classifier.predict(sc.transform([[30,87000]])))


#Predicting the Test set results
y_pred = classifier.predict(X_test)
# Reshape is used to ensure column vectors, concatenate joins them horizontally
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


#Making the Confusion Matrix(tests accuracy)
"""
----------------
We'll evaluate using:
1. Confusion Matrix - shows correct/incorrect classifications
2. Accuracy Score - percentage of correct predictions
"""

# 3. Making the Confusion Matrix and calculating accuracy
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)  # Format: [[TN, FP], [FN, TP]]

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2%}")

"""

"""
DATA VISUALIZATION
------------------
We'll create visualizations to understand:
1. Decision boundary on training set
2. Decision boundary on test set
"""

# Visualizing the Training set results
plt.figure(figsize=(10, 6))

# Inverse transform to get original scale values for visualization
X_set, y_set = sc.inverse_transform(X_train), y_train

# Create a mesh grid for plotting decision boundary
# The grid extends slightly beyond the data range for better visualization
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
    np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25)
)

# Predict for each point on the grid (requires scaling the grid points)
Z = classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T))
Z = Z.reshape(X1.shape)  # Reshape to match grid dimensions

# Create filled contour plot for decision regions
plt.contourf(
    X1, X2, Z,
    alpha=0.75,  # Transparency
    cmap=ListedColormap(['#FA8072', '#1E90FF'])  # Colors for classes
)

# Plot limits based on grid
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# Plot the actual training points
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set == j, 0], X_set[y_set == j, 1],
        c=ListedColormap(['#FA8072', '#1E90FF'])(i),
        label=j
    )

# Add chart elements
plt.title('Logistic Regression Decision Boundary (Training set)')
plt.xlabel('Age (original scale)')
plt.ylabel('Estimated Salary (original scale)')
plt.legend(title="Class")
plt.show()

# Visualizing the Test set results (similar to training visualization)
plt.figure(figsize=(10, 6))

# Get test set in original scale
X_set, y_set = sc.inverse_transform(X_test), y_test

# Create mesh grid (smaller extension for test set visualization)
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.25),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.25)
)

# Predict for grid points
Z = classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T))
Z = Z.reshape(X1.shape)

# Plot decision regions
plt.contourf(X1, X2, Z, alpha=0.75, cmap=ListedColormap(['#FA8072', '#1E90FF']))

# Set plot limits
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# Plot test points with simpler color specification
colors = ['#FA8072', '#1E90FF']  # Salmon and Dodger Blue
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set == j, 0], X_set[y_set == j, 1],
        color=colors[i], label=j
    )

# Add chart elements
plt.title('Logistic Regression Decision Boundary (Test set)')
plt.xlabel('Age (original scale)')
plt.ylabel('Estimated Salary (original scale)')
plt.legend(title="Class")
plt.show()