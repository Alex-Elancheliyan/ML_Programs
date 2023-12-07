import numpy as np
from statistics import mean
from scipy.stats import f

def calculate_slope_intercept(xvalues, yvalues):
    m = (((mean(xvalues) * mean(yvalues)) - mean(xvalues * yvalues)) /
         ((mean(xvalues) * mean(xvalues)) - mean(xvalues * xvalues)))
    b = mean(yvalues) - m * mean(xvalues)
    return m, b


def linear_regression():
    regression_line = [(m * x) + b for x in xvalues]
    return regression_line


def determination_coeff(ys_orig, ys_line):
    y_mean = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean)
    print('SER:',squared_error_regr)
    print('SEYM:',squared_error_y_mean)
    err_value = squared_error_regr /squared_error_y_mean
    print('Error Value:', err_value)
    rsq = 1 - err_value
    return rsq


def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig) * (ys_line - ys_orig))


#DRIVER CODE
xvalues = np.array([1, 2, 3, 4, 5])
yvalues = np.array([1, 2, 3, 4, 5])

m, b = calculate_slope_intercept(xvalues, yvalues)

print('slope:', m, 'Intercept:', b)
regression_line = linear_regression()
print('REG LINE:',regression_line)
Rsq = determination_coeff(yvalues, regression_line)
print('RSQ:',Rsq)

threshold = 0.5
if (Rsq > threshold):
    print('Range is Acceptable')
else:
    print('Range is not Acceptable')

#PERFORM F TEST to test the significance of R squared
n = len(yvalues)
k = 1  #Number of independant variables including intercepts
dof_reg = k
dof_resid = n - k - 1 #degree of freedom residuals

#calculate the explained variance (numerator)
explained_variance = squared_error(yvalues,regression_line)

#CALCULATE THE RESIDUAL VARIANCE (DENOMINATOR)
residual_variance = squared_error(yvalues, mean(yvalues))

#calculate the F-Statstics
F = (explained_variance / dof_reg) / (residual_variance / dof_resid)

#calculate the P-value associated with the F-Statistics
p_value = 1 - f.cdf(F, dof_reg, dof_resid)

print("F-Statistics:", F)
print("P-Value:", p_value)

# Set the significance
alpha =  0.05

#Perform hypothesis test
if p_value < alpha:
    print("R Squared is stastically significant."
          "Reject the null Hypothesis.")
else:
    print("R Squared is not stastically significant."
          "Accept the null Hypothesis.")