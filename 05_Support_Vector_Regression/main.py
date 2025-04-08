import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Support Vector Regression (SVR) is a machine learning technique based on Support Vector Machines (SVM).
# Unlike traditional regression models, SVR tries to fit the data within a specified margin of tolerance (epsilon).

# 1. Goal:
#    - SVR aims to find a hyperplane (or curve for non-linear problems) that predicts values within a tolerance level (epsilon).
#    - The model does not penalize predictions within the epsilon margin (soft margin).
#    - Predictions outside the epsilon margin incur a penalty proportional to the distance from the margin.

# 2. Key concepts:
#    - Epsilon-tube: Defines the margin of tolerance around the hyperplane. Predictions within this tube are considered "correct".
#    - Support Vectors: Points that lie on the margin or outside it, which influence the position and orientation of the hyperplane.
#    - Slack Variables: Allow the model to handle points outside the epsilon-tube.

# 3. Loss function:
#    - The SVR optimization minimizes a combination of the regularization term and the penalty for predictions outside the epsilon margin:
#      Minimize: (1/2) * ||w||^2 + C * Σ(max(0, |y_i - f(x_i)| - ε))
#        - ||w||^2: Regularization term to avoid overfitting.
#        - C: Trade-off parameter between model complexity and tolerance to errors.
#        - |y_i - f(x_i)|: Absolute error between the predicted value and actual value.
#        - ε (epsilon): Defines the tolerance for error.

# 4. Kernel trick:
#    - Maps the input features into a higher-dimensional space for non-linear relationships.
#    - Example kernels include linear, polynomial, and RBF (used here for complex, non-linear data).

# Example with calculations:
# Let's consider a simple dataset:
# X = [1, 2, 3, 4, 5]  # Feature values
# y = [1.5, 3.5, 2.0, 5.0, 4.5]  # Target values

# Assume:
#    - Epsilon (ε) = 0.5
#    - C (trade-off parameter) = 1.0
#    - Predicted values (f(x)) will be calculated as an example.

# Example 1: X = 3, prediction = 2.0
#    - Actual value: y = 2.0
#    - Absolute error = |y - f(x)| = |2.0 - 2.0| = 0.0
#    - Since the error is within epsilon (0.5), no penalty is applied.

# Example 2: X = 4, prediction = 4.2
#    - Actual value: y = 5.0
#    - Absolute error = |y - f(x)| = |5.0 - 4.2| = 0.8
#    - Error exceeds epsilon (0.5), so penalty = 0.8 - 0.5 = 0.3.
#    - This penalty contributes to the loss function.

# 5. Final predictions:
#    - After training, the model predicts values based on support vectors and the selected kernel.
#    - Predictions are then transformed back to the original scale if feature scaling was applied.

# Visualization:
# The epsilon-tube is visualized as a margin around the hyperplane (or curve), and only points outside this margin 
# affect the model's training process. This makes SVR robust to small deviations or noise in the data.

# Summary:
# - SVR provides flexibility by ignoring small deviations and focusing on the overall trend.
# - The key is balancing model complexity (regularization) and error tolerance (epsilon).

# Note:
# - For regression tasks with noisy data, SVR can effectively smooth out the data while capturing its general behavior.
# - Ensure proper scaling of features before training, as SVR is sensitive to feature magnitude.



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
