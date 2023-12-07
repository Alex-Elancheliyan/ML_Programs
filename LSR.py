import numpy as np
from statistics import mean


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


# DRIVER CODE

xvalues = np.array([1, 2, 3, 4, 5])
yvalues = np.array([2, 4, 6, 8, 10])

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



