import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 #Explanation of K-NN:
# 1. K-NN is a supervised learning algorithm used for classification and regression.
# 2. When classifying a data point, the algorithm calculates the distance (e.g., Euclidean) to all points in the training set.
# 3. It selects the 'k' nearest neighbors and assigns the most common class label among them to the new data point.
# 4. Important parameters:
#    - `n_neighbors`: Number of neighbors to consider.
#    - `metric`: Distance metric used for calculation.
#    - `p`: Power parameter for Minkowski distance (p=2 corresponds to Euclidean distance).



# In K-NN, we calculate distances (e.g., Euclidean distance) to find the nearest neighbors.
# Euclidean distance formula:
# d(x, x_i) = sqrt(sum((x_j - x_{ij})^2 for all features j))
# Example:
# If a new point has features (26, 23000), and a training point has features (22, 20000),
# d = sqrt((26 - 22)^2 + (23000 - 20000)^2) = sqrt(4 + 9000000) ≈ 3000.01

# By finding 'k' nearest neighbors and their majority class, K-NN predicts the class of the new point.


#EXTRAAAAAAAAAAA
# The `p` parameter in Minkowski distance determines the type of distance metric used:
# The general formula for Minkowski distance is:
# d(x, x_i) = (sum(|x_j - x_{ij}|^p for all features j))^(1/p)
# Where:
# - x: The point being classified
# - x_i: A training point
# - n: The number of features
# - p: The power parameter (defines the type of distance metric)

# Common values for p:
# 1. p = 1: Manhattan Distance (L1 norm)
#    d(x, x_i) = sum(|x_j - x_{ij}| for all features j)
#    This is the sum of absolute differences and is used for grid-like navigation (e.g., city-block distance).

# 2. p = 2: Euclidean Distance (L2 norm)
#    d(x, x_i) = sqrt(sum((x_j - x_{ij})^2 for all features j))
#    This is the straight-line distance and the most commonly used metric.

# 3. Higher values of p (e.g., p = 3 or more):
#    d(x, x_i) gives more weight to larger feature differences, and distances grow non-linearly.

# 4. p → infinity: Chebyshev Distance
#    d(x, x_i) = max(|x_j - x_{ij}| for all features j)
#    This considers only the largest absolute difference among all features.


# Example of Minkowski distance with different values of p:

# Suppose we have two points:
# Point A: [26, 23000] (e.g., age and salary of a person)
# Point B: [22, 20000] (another person in the dataset)

# Minkowski distance formula:
# d(x, x_i) = (sum(|x_j - x_{ij}|^p for all features j))^(1/p)

# Example calculations:
# For p = 1 (Manhattan Distance):
# d(A, B) = |26 - 22| + |23000 - 20000| = 4 + 3000 = 3004

# For p = 2 (Euclidean Distance):
# d(A, B) = sqrt((26 - 22)^2 + (23000 - 20000)^2) = sqrt(4 + 9000000) ≈ 3000.01

# For p = 3:
# d(A, B) = ((|26 - 22|^3 + |23000 - 20000|^3))^(1/3) = ((4^3 + 3000^3))^(1/3)
#          = ((64 + 27000000000))^(1/3) ≈ 3000.00001

# As p increases, the larger feature differences dominate the calculation.


dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


from sklearn.model_selection import train_test_split
# This step divides the data into training data (75%) and test data (25%).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Training the K-NN model
# K-NN works by finding the 'k' nearest neighbors to classify a data point based on majority vote.
# Parameters:
# - n_neighbors: Number of neighbors (k)
# - metric: Distance metric (Minkowski allows Euclidean when p=2)a.

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)



#Predicting the Test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Evaluating the Model
# Confusion matrix: Summarizes prediction results by showing correct/incorrect classifications.
# Accuracy score: Percentage of correctly predicted points.
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)




#Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.5),
                     np.araange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.5))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(['#FA8072', '#1E90FF']))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(['#FA8072', '#1E90FF'])(i), label = j)
plt.title('K-NN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()




#Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_test), y_test
# Create a grid of points
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.5),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.5)
)
# Predict for each point on the grid
Z = classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape)
# Plot the decision boundary
plt.contourf(X1, X2, Z, alpha=0.75, cmap = ListedColormap(['#FA8072', '#1E90FF']) )
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
# Define colors for scatter plot
colors = ['#FA8072', '#1E90FF']
# Plot the test set points
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set == j, 0], X_set[y_set == j, 1],
        color=colors[i], label=j
    )
# Add titles and labels
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()