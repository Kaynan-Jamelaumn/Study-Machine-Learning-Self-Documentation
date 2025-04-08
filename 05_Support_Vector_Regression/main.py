import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Support Vector Regression (SVR) is a machine learning technique based on Support Vector Machines (SVM).
# Unlike traditional regression models, SVR tries to fit the data within a specified margin of tolerance (epsilon).
# The goal is to find a curve that captures the general trend of the data, while ignoring minor deviations or outliers.
# This is achieved by optimizing a loss function that allows for flexibility in defining the "fit" zone (epsilon-tube).
# Key Parameter - Kernel:
# The kernel function determines how the data is mapped into higher dimensions to find a more accurate fit.
# Common kernels include linear, polynomial, and RBF (Radial Basis Function). Here, RBF is used for non-linear regression.


dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values



#Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


#Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)



# Predicting a new result with the trained SVR model
# Predict the salary for a specific position level (e.g., 6.5) and inverse transform to get the original scale
predicted_salary = sc_y.inverse_transform(
    regressor.predict(sc_X.transform([[6.5]])).reshape(-1, 1)
)
# Note: Reshape ensures the predicted values are in 2D format for compatibility with inverse_transform.


# Visualising the SVR results
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')  # Scatter plot of the original data
plt.plot(
    sc_X.inverse_transform(X),
    sc_y.inverse_transform(regressor.predict(X).reshape(-1, 1)),
    color='blue'
)  # Line plot of the predicted results
plt.title('Truth or Bluff (SVR)')  # Title of the plot
plt.xlabel('Position level')  # X-axis label
plt.ylabel('Salary')  # Y-axis label
plt.show()

# Visualising the SVR results with higher resolution and smoother curve
# Create a grid of position levels for higher resolution
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))  # Reshape to 2D array for prediction

plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')  # Scatter plot of the original data
plt.plot(
    X_grid,
    sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1, 1)),
    color='blue'
)  # Line plot of the predicted results with a smoother curve
plt.title('Truth or Bluff (SVR)')  # Title of the plot
plt.xlabel('Position level')  # X-axis label
plt.ylabel('Salary')  # Y-axis label
plt.show()
