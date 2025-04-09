# -*- coding: utf-8 -*-
"""
Random Forest Regression for Math Beginners
"""
import numpy as np

# =============================================================================
# SIMPLE EXAMPLE DATASET (5 employees)
# =============================================================================
"""
Position Level: [1, 2, 3, 4, 5] 
Salary:        [30, 50, 60, 80, 110] (in thousands)

Visual Representation:
Level 1: $30k
Level 2: $50k
Level 3: $60k
Level 4: $80k
Level 5: $110k
"""
X = np.array([[1], [2], [3], [4], [5]])  # Position levels
y = np.array([30, 50, 60, 80, 110])      # Salaries

# =============================================================================
# HOW ONE DECISION TREE WORKS (SIMPLIFIED)
# =============================================================================
"""
Example Split at Level 2.5:
--------------------------
Left Branch (Levels ≤ 2.5): [1, 2] → Salaries [30, 50]
Right Branch (Levels > 2.5): [3, 4, 5] → Salaries [60, 80, 110]

Calculations:
1. Left Node Prediction = Average(30, 50) = 40
2. Right Node Prediction = Average(60, 80, 110) = 83.33

Mean Squared Error (MSE) Calculation:
MSE = (MSE_left * n_left + MSE_right * n_right) / n_total

Where:
- MSE_left = average of [(30-40)² + (50-40)²]/2 = [100 + 100]/2 = 100
- MSE_right = [(60-83.33)² + (80-83.33)² + (110-83.33)²]/3 ≈ [544 + 11 + 711]/3 ≈ 422
- Weighted MSE = (2*100 + 3*422)/5 = (200 + 1266)/5 = 293.2

The algorithm tests ALL possible splits and chooses the one with lowest MSE
"""

# =============================================================================
# RANDOM FOREST WITH 3 TREES (TINY EXAMPLE)
# =============================================================================
"""
Tree 1: (Bootstrap sample [1,2,2,3,5] - notice 2 is duplicated, 4 is missing)
- Split at 2.5 → Predictions: left=40, right=83.33

Tree 2: (Sample [1,3,3,4,5])
- Split at 3.5 → 
  left=avg(30,60,60)=50, right=avg(80,110)=95

Tree 3: (Sample [2,3,4,4,5])
- Split at 4.5 →
  left=avg(50,60,80,80)=67.5, right=110

Final Prediction for Level 6.5:
Tree 1: 83.33 (falls in right branch)
Tree 2: 95    (right branch)
Tree 3: 110   (right branch)
Average = (83.33 + 95 + 110)/3 ≈ 96.11
"""

# =============================================================================
# IMPLEMENTATION WITH STEP-BY-STEP MATH
# =============================================================================
from sklearn.ensemble import RandomForestRegressor

# Create a forest with 3 trees (for demonstration)
forest = RandomForestRegressor(n_estimators=3, random_state=0)
forest.fit(X, y)

# Predict for level 6.5
prediction = forest.predict([[6.5]])
print(f"Predicted salary for level 6.5: ${prediction[0]:.2f}k")

# =============================================================================
# VISUALIZING TREE DECISIONS
# =============================================================================
# Generate test points
X_test = np.arange(1, 6, 0.1).reshape(-1, 1)

# Get individual tree predictions
tree_predictions = []
for tree in forest.estimators_:
    tree_predictions.append(tree.predict(X_test))

# Plotting
plt.figure(figsize=(10,6))
plt.scatter(X, y, color='red', s=100, label='Actual Salaries')

# Plot individual trees
for i, preds in enumerate(tree_predictions, 1):
    plt.plot(X_test, preds, '--', alpha=0.7, label=f'Tree {i} Prediction')

# Plot forest average
plt.plot(X_test, forest.predict(X_test), 
         color='blue', 
         linewidth=3, 
         label='Forest Average')

plt.title('How Random Forest Predictions Combine')
plt.xlabel('Position Level')
plt.ylabel('Salary (in $k)')
plt.legend()
plt.show()

"""
KEY OBSERVATIONS:
1. Each tree makes "stair-step" predictions
2. Different trees split at different points
3. The forest average (blue line) smooths out the predictions
4. At level 6.5 (off the chart), the prediction is ~96.11k as calculated
"""

# =============================================================================
# FORMULAS AS PYTHON FUNCTIONS
# =============================================================================
def calculate_mse(y_true, y_pred):
    """Mean Squared Error Calculation"""
    return np.mean((y_true - y_pred)**2)

def weighted_mse(left_y, right_y):
    """Calculate weighted MSE for a split"""
    left_pred = np.mean(left_y)
    right_pred = np.mean(right_y)
    
    mse_left = calculate_mse(left_y, left_pred)
    mse_right = calculate_mse(right_y, right_pred)
    
    n_left = len(left_y)
    n_right = len(right_y)
    
    return (mse_left * n_left + mse_right * n_right) / (n_left + n_right)

# Example calculation for split at 2.5
left = y[X.ravel() <= 2.5]  # [30, 50]
right = y[X.ravel() > 2.5]  # [60, 80, 110]

print(f"\nWeighted MSE for split at 2.5: {weighted_mse(left, right):.1f}")