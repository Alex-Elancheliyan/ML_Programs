import numpy as np
import matplotlib.pyplot as plt


x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 6, 6, 9, 10])

#calculating mean of x and y
mean_x = np.mean(x)
mean_y = np.mean(y)

#Calculate slope m and y intercept B
numerator = np.sum(( x - mean_x)*( y - mean_y))
denominator = np.sum(( x- mean_x) ** 2)
m = numerator / denominator
b = mean_y - m * mean_x
print('slope:', m, 'Intercept:', b)


#simply finding m and b using polyfit function of numpy
m, b = np.polyfit (x, y, 1)

y_pred = m * x + b
print('Predicted value', y_pred)

#CAlculate R square values

residuals = y - y_pred
ss_residual = np.sum(residuals ** 2)
ss_total = np.sum((y - np.mean(y)) ** 2)
r_square = 1 - (ss_residual / ss_total)

print(f"Regression Equation : y = { m:.2f}x + {b:2f}")
print("Pred:",y_pred)
print(f"R Squared value Value: { r_square:.2f}")

threshold = 0.7
if (r_square >= threshold):
    print('R Square is  above Acceptable threshold')
else:
    print('R Square is not  above Acceptable threshold')

plt.title('BEST FIT LINE')
plt.scatter(x ,y ,color='#003F72',label='Original Values')
plt.scatter( x , y_pred ,color='#FF0000',label='Predicted Value')
plt.plot(x, y, label='Reg Line')
plt.legend(loc='best')
plt.show()
