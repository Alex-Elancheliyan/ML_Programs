import numpy as np
import statsmodels.api as sm

# sample data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Add a constant term for the intercept
X = sm.add_constant(x)
# print(' Constant:', X)

# FIt a line OLS model
model = sm.OLS(y, X).fit()

# Get the summarry of the regression model
summary = model.summary()
print(summary)
