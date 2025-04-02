import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Loading the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values  # Selecting the independent variable
y = dataset.iloc[:, -1].values    # Selecting the dependent variable (salary)

# Creating the polynomial transformation of the data
poly_reg = PolynomialFeatures(degree=4)  # Generating polynomial terms up to degree 4
X_poly = poly_reg.fit_transform(X)       # Transforming X to include new polynomial terms

# Creating and training the linear regression model with transformed data
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Polynomial Regression Formula Explanation:
# The polynomial regression equation is given by:
# y = θ₀ + θ₁X + θ₂X² + θ₃X³ + θ₄X⁴ + ... + θₙXⁿ
# 
# - y: The predicted output (salary in this case)
# - X: The independent variable (position level)
# - θ₀, θ₁, θ₂, ... , θₙ: Coefficients (parameters) determined by the model
# - X², X³, X⁴, ... , Xⁿ: Higher-degree terms introduced to capture nonlinear relationships
# - The model learns the values of θ coefficients to best fit the data

# Bayesian Information Criterion (BIC) Formula:
# The Bayesian Information Criterion (BIC) is a statistical metric used for model selection. It helps to determine the best model by balancing goodness of fit and model complexity.
# BIC = k * ln(n) + n * ln(RSS / n)
# 
# - k: Number of parameters in the model (including the intercept)
# - n: Number of data points
# - RSS: Residual Sum of Squares (sum of squared errors between predicted and actual values)
# - ln: Natural logarithm
# The BIC is used for model selection, penalizing excessive complexity.

# Computing BIC for the polynomial regression model
n = len(y)  # Number of data points
k = X_poly.shape[1]  # Number of parameters
RSS = mean_squared_error(y, lin_reg_2.predict(X_poly)) * n  # Residual sum of squares
BIC = k * np.log(n) + n * np.log(RSS / n)
print(f'Bayesian Information Criterion (BIC) for polynomial regression: {BIC}')

# Visualizing the Polynomial Regression results
plt.scatter(X, y, color='red')  # Plotting the actual data points
plt.plot(X, lin_reg_2.predict(X_poly), color='blue')  # Polynomial regression line
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualizing the Polynomial Regression with a smoother curve
X_grid = np.arange(min(X), max(X), 0.1)  # Generating more points for a smoother curve
X_grid = X_grid.reshape((len(X_grid), 1))  # Reshaping to be compatible with the model
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')  # Smooth polynomial regression line
plt.title('Truth or Bluff (Polynomial Regression - High Resolution)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Plotting BIC to visualize model complexity vs fit
plt.bar(['Polynomial Regression'], [BIC], color='purple')
plt.title('Bayesian Information Criterion (BIC)')
plt.ylabel('BIC Value')
plt.show()

# Making a prediction with the polynomial model
predicted_salary = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(f'Predicted salary for level 6.5: {predicted_salary[0]}')
