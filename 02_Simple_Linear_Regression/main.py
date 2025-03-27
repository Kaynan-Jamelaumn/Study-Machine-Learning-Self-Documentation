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