import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')  
# Loads the data from a CSV file named 'Data.csv'.
# The dataset is stored as a Pandas DataFrame.

X = dataset.iloc[:, :-1].values  
# Selects all columns except the last one as independent variables (features).
y = dataset.iloc[:, -1].values  
# Selects the last column as the dependent variable (target).

# Handling missing data
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')  
# Creates an object to replace missing values (NaN) with the mean of the corresponding column.

imputer.fit(X[:, 1:3])  
# Computes the mean for the specified columns (in this case, columns 1 and 2).

X[:, 1:3] = imputer.transform(X[:, 1:3])  
# Replaces the missing values with the calculated means.

# Encoding categorical variables (e.g., 'Country' or 'Movie Genre')
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Example of handling a categorical field like 'Country'
# Suppose the first column contains country names such as 'France', 'Germany', and 'Spain'.
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')  
# Transforms the first column (index 0) into a one-hot encoded format.

X = np.array(ct.fit_transform(X))  
# Applies the transformation and converts the result back to a NumPy array.

# Example of handling another categorical field like 'Movie Genre'
# Suppose there is an additional column representing movie genres such as 'Action', 'Drama', 'Comedy'.
dataset['Genre'] = ['Action', 'Drama', 'Comedy', 'Action', 'Drama']  # Example genre data
genre_ct = ColumnTransformer(transformers=[('genre_encoder', OneHotEncoder(), ['Genre'])], remainder='passthrough')  
# Transforms the 'Genre' column into a one-hot encoded format.

X_with_genre = np.array(genre_ct.fit_transform(dataset))  
# Applies the transformation and adds the encoded genre data.

# Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  
# Splits the dataset into training (80%) and test (20%) sets.
# random_state ensures reproducibility of the split.

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()  
# Creates an object for standard scaling (normalization with mean 0 and standard deviation 1).

X_train = sc.fit_transform(X_train)  
# Fits the scaler on the training data and transforms it.

X_test = sc.transform(X_test)  
# Transforms the test data using the scaler fitted on the training data.



#ADITIONAL DOCUMENTATION:


# --- Feature Scaling: Standardization ---
# Standardization transforms the data to have a mean of 0 and a standard deviation of 1.
# Formula: z = (x - mean) / std_dev
#   - Where:
#     - x = original value
#     - mean = average of the column
#     - std_dev = standard deviation of the column
#   - Standardization does not bound the data to a specific range.
#   - It is useful when the data follows a normal distribution or for algorithms sensitive to scale (e.g., SVM, Logistic Regression).

# Standard Deviation (std_dev) is a measure of how spread out the data is around the mean.
# It is calculated as the square root of the variance.

sc = StandardScaler()
X_train_standardized = sc.fit_transform(X_train)  # Fit to training data and transform
X_test_standardized = sc.transform(X_test)       # Transform test data using the same scaler

# --- Feature Scaling: Normalization ---
# Normalization transforms the data to fit within a specific range, typically [0, 1].
# Formula: x' = (x - min) / (max - min)
#   - Where:
#     - x = original value
#     - min = minimum value of the column
#     - max = maximum value of the column
#   - Normalization is useful for algorithms based on distance metrics (e.g., KNN, Neural Networks).



normalizer = MinMaxScaler()
X_train_normalized = normalizer.fit_transform(X_train)  # Fit to training data and transform
X_test_normalized = normalizer.transform(X_test)       # Transform test data using the same scaler



# --- Summary of Scaling Methods ---
# Standardization adjusts the data to have zero mean and unit variance, but does not restrict it to a specific range.
# Normalization adjusts the data to fit within a predefined range, typically [0, 1].






# Documentation: What is Standard Deviation? ---
# Standard Deviation (std_dev) is a measure of how spread out the data is around the mean.
# It is calculated as the square root of the variance.
# 
# Steps:
# 1. Calculate the Mean (μ):
#    μ = (Σx_i) / n
#    Where:
#      - x_i = each value in the dataset
#      - n = total number of values
#
# 2. Calculate the Variance (σ^2):
#    σ^2 = (Σ(x_i - μ)^2) / n
#    This is the average of the squared differences between each data point and the mean.
#
# 3. Calculate the Standard Deviation (σ):
#    σ = sqrt(σ^2)
#
# For a sample (not the entire population), divide by (n - 1) instead of n in the variance formula.
#
# Formula for scaling data (standardization):
#    z = (x - mean) / std_dev
#
# Example:
# If the mean = 10 and std_dev = 2, a value of x = 14 would be standardized as:
#    z = (14 - 10) / 2 = 2.0