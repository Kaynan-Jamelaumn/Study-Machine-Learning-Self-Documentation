import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Decision Tree Regressor:
# The Decision Tree algorithm works by repeatedly splitting the data into subsets based on a feature's value.
# Each split is chosen to minimize the prediction error (e.g., mean squared error for regression).
# At the end of the process, we have "leaves" that represent the final predictions for each region of the data.


# How the Decision Tree algorithm works(simplified):
# 1. Start with the root node, containing all data.
# 2. Find the best feature and value to split the data by testing all possible splits.
#    - For regression, the best split minimizes the Mean Squared Error (MSE):
#      MSE = (Sum of Squared Errors) / (Number of Samples)

## example:
"""

"""

# 3. Recursively split the data into child nodes until a stopping criterion is met.
#    - Stopping criteria include:
#      - Maximum depth of the tree.
#      - Minimum number of samples in a node.
#      - All samples in a node belong to the same target value (pure leaf).
# 4. Each leaf contains the mean of the target values in its region, which is used as the prediction.

# Advantages:
# - Captures non-linear relationships in the data.
# - Easy to interpret and visualize.
# - Works well with both numerical and categorical data.

# Disadvantages:
# - Sensitive to small changes in data (can lead to different splits).
# - Prone to overfitting if the tree grows too deep.
# - Produces step-like predictions, which may not be ideal for smooth relationships.


dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


# Training the Decision Tree Regression model
from sklearn.tree import DecisionTreeRegressor  # Importing the model
regressor = DecisionTreeRegressor(random_state=0)  # Initializing the model with a fixed random state
regressor.fit(X, y)  # Training the model on the entire dataset


## Predicting a new result
regressor.predict([[6.5]])  # Input must be a 2D array

# Visualizing the Decision Tree Regression results (higher resolution)
# Create a high-resolution grid of position levels for better visualization
X_grid = np.arange(min(X), max(X), 0.01)  # Creates values from min(X) to max(X) with step size of 0.01
X_grid = X_grid.reshape((len(X_grid), 1))  # Reshape into a 2D array for the model

# Scatter plot of the original data
plt.scatter(X, y, color='red', label='Actual Data')  # Red dots for the actual salary values
# Line plot of the predictions over the high-resolution grid
plt.plot(X_grid, regressor.predict(X_grid), color='blue', label='Model Prediction')  # Blue line for predictions
plt.title('Truth or Bluff (Decision Tree Regression)')  # Title of the plot
plt.xlabel('Position Level')  # Label for the x-axis
plt.ylabel('Salary')  # Label for the y-axis
plt.legend()  # Display legend for better understanding
plt.show()  # Display the plot









# How the Decision Tree algorithm works(detailed):
# 1. Start with all the data in a single node (the root).
# 2. Test all possible splits of the feature values (e.g., position levels).
#    - For each split, divide the data into two groups: left and right.
#    - Calculate the MSE for each group:
#      - MSE = mean((y - mean_y)^2) for each group.
#    - Compute the weighted average of the MSEs for the two groups:
#      Weighted MSE = (N_left * MSE_left + N_right * MSE_right) / Total_samples
# 3. Choose the split that minimizes the weighted MSE and create child nodes.
# 4. Recursively repeat steps 2-3 for the child nodes until a stopping criterion is met:
#    - Maximum depth of the tree.
#    - Minimum number of samples in a node.
#    - Pure leaf (all y values in a node are the same).
# 5. Leaves contain the average of the y values in their region, which is used for prediction.

# Example:
# Suppose the data has 10 samples, and the split occurs at X = 6:
#    Left group: [1, 2, 3, 4, 5, 6], y = [150, 200, 250, 300, 400, 500]
#    Right group: [7, 8, 9, 10], y = [600, 800, 1000, 1200]
# The MSE for each group is calculated, and the weighted MSE is minimized.
# This split would result in two regions:
#    - Region 1 (X ≤ 6): Prediction = mean([150, 200, 250, 300, 400, 500])
#    - Region 2 (X > 6): Prediction = mean([600, 800, 1000, 1200])


# Detailed explanation of the example with calculations:

# Step 1: Splitting the data at X = 6
#    - Left group: [1, 2, 3, 4, 5, 6], y = [150, 200, 250, 300, 400, 500]
#    - Right group: [7, 8, 9, 10], y = [600, 800, 1000, 1200]

# Step 2: Calculate the mean of y for each group:
#    - Left group mean (mean_left_y) = (150 + 200 + 250 + 300 + 400 + 500) / 6 = 300.0
#    - Right group mean (mean_right_y) = (600 + 800 + 1000 + 1200) / 4 = 900.0

# Step 3: Calculate the Mean Squared Error (MSE) for each group:
#    - Left group MSE (mse_left) = mean((y - mean_y)^2)
#      = [(150 - 300)^2 + (200 - 300)^2 + (250 - 300)^2 + (300 - 300)^2 + (400 - 300)^2 + (500 - 300)^2] / 6
#      = [22500 + 10000 + 2500 + 0 + 10000 + 40000] / 6 = 14166.67
#    - Right group MSE (mse_right) = mean((y - mean_y)^2)
#      = [(600 - 900)^2 + (800 - 900)^2 + (1000 - 900)^2 + (1200 - 900)^2] / 4
#      = [90000 + 10000 + 10000 + 90000] / 4 = 50000.0

# Step 4: Calculate the weighted MSE for the split:
#    - Number of samples in the left group (N_left) = 6
#    - Number of samples in the right group (N_right) = 4
#    - Total number of samples (Total_samples) = 10
#    - Weighted MSE = (N_left * mse_left + N_right * mse_right) / Total_samples
#      = (6 * 14166.67 + 4 * 50000.0) / 10 = 28500.0

# Step 5: Determine predictions for each region:
#    - Region 1 (X ≤ 6): Prediction = mean_left_y = 300.0
#    - Region 2 (X > 6): Prediction = mean_right_y = 900.0

# Summary:
# - The split at X = 6 minimizes the weighted MSE (28500.0) compared to other possible splits.
# - Predictions are made based on the mean of y values in each region:
#    - Region 1 (X ≤ 6): Predicted salary = 300.0
#    - Region 2 (X > 6): Predicted salary = 900.0
