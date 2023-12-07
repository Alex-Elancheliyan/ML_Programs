import numpy as np
import statsmodels.api as sm

# Sample data for X and Y
X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]])  # Independent variables or Features
Y = np.array([2, 3, 4, 4, 5])  # Dependant variable or Label

# Add a constant term for intercept
X = sm.add_constant(X)

# Fit the multiple linear regression model
model = sm.OLS(Y, X).fit()

# Get the model summary
model_summary = model.summary()
print(model_summary)

coefficients = model.params
print('coefficients:')
print(coefficients)
