#Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


# Training a Simple Linear Regression model on the Training set
# Linear Regression is used here to predict salary based on years of experience.
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()  # Create a LinearRegression object
regressor.fit(X_train, y_train)  # Fit the model to the training data

# Making predictions on the Test set
# Using the trained model to predict the salaries based on the test set features (experience)
y_pred = regressor.predict(X_test)

# Visualizing the Training set results
# This graph will show the relationship between years of experience and salary for the training data
# The red points represent the actual data, and the blue line shows the model's predictions.
plt.scatter(X_train, y_train, color = 'red')  # Plot actual data points for training set
plt.plot(X_train, regressor.predict(X_train), color = 'blue')  # Plot the regression line for training set predictions
plt.title('Salary vs Experience (Training set)')  # Set the title of the plot
plt.xlabel('Years of Experience')  # Label for the x-axis
plt.ylabel('Salary')  # Label for the y-axis
plt.show()  # Display the plot

# Visualizing the Test set results
# Similar to the training set visualization, but using test set data
# The red points represent the actual test data, and the blue line still shows the model's predictions based on the training data
plt.scatter(X_test, y_test, color = 'red')  # Plot actual data points for test set
plt.plot(X_train, regressor.predict(X_train), color = 'blue')  # Plot the regression line for training set predictions
plt.title('Salary vs Experience (Test set)')  # Set the title of the plot
plt.xlabel('Years of Experience')  # Label for the x-axis
plt.ylabel('Salary')  # Label for the y-axis
plt.show()  # Display the plot







"""
Linear Regression Equation Documentation:

The Linear Regression equation used in this model is:

    y = β₀ + β₁ * x

Where:
- y (Salary) is the dependent variable (the value we want to predict, in this case, salary).
- x (Years of Experience) is the independent variable (the feature used to predict salary).
- β₀ (Intercept) is the point where the regression line crosses the y-axis, which represents the predicted salary when the years of experience is 0.
- β₁ (Slope) is the coefficient that represents how much the salary increases with each additional year of experience.

### Example:
For a given model, the equation might look like this:

    Salary = 30000 + 5000 * (Years of Experience)

Where:
- β₀ = 30000: The starting salary when someone has 0 years of experience.
- β₁ = 5000: The salary increase per additional year of experience.

For an individual with 3 years of experience, the predicted salary would be:

    Salary = 30000 + 5000 * 3 = 45000

Thus, the predicted salary for someone with 3 years of experience would be 45,000 units.

The model finds the best-fitting line that minimizes the error between the predicted and actual salary values based on the data. This is achieved by determining the values of β₀ and β₁ during the training process using the training data.
"""