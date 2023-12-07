import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels
import statsmodels.api as sm

# sample data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Add a constant term for the intercept
X = sm.add_constant(x)
# print(' Constant:', X)

# FIt a line OLS model
model = sm.OLS(y, X).fit()

# residuals
residuals = model.resid

# Residual plot
plt.figure(figsize=(8, 6))
plt.scatter(model.predict(), residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residual Plot')
plt.xlabel("Predicted Values:")
plt.ylabel("Residuals:")
plt.show()

# check fror NOrmality of residuals using a Q_Q plot
sm.qqplot(residuals, line='s')
plt.title('Q-Q Plot')
plt.show()

# Histogram Of Residuals
plt.hist(residuals, bins=15, edgecolor='k', alpha=0.7)
plt.title('Histogram of residuals')
plt.xlabel("Residuals")
plt.ylabel('Frequency')
plt.show()

# check for heteroscedacitiy (residuals vs Fitted values
plt.scatter(model.predict(), residuals)
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values (Check for Heteroscedasticity)')
plt.axhline(y=0, color='red', linestyle='--')
plt.show()

# Run a formal test for heteroscedasticity ( Breush - Pagan test)
_, p_value, _, _ = statsmodels.stats.diagnostic.het_breuschpagan(residuals, x)
print(f'P-Value for B-Pagan test:{p_value}')
if p_value < 0.5:
    print(" Heteroscedasticity Occurs")
else:
    print("Heteroscedasticity Not occurs")

# Check residuals are normally distributed using Shapiro-Wilk Test
_, p_value = stats.shapiro(residuals)
print(f'P-Value for Shapiro-Wilk test:{p_value}')
if p_value > 0.5:
    print(" Normal distribution Occurs")
else:
    print("Normal distribution Not occurs")
