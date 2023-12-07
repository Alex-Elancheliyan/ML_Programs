import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X = housing.data
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

print('Coefficients:\n', reg.coef_)
print('Variance score:{}'.format(reg.score(X_test, y_test)))

plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train, color='green', s=10, label='Train Data')
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test, color='blue', s=10, label='Test Data')
plt.hlines(y=0, xmin=0, xmax=10, linewidth=2)
plt.legend(loc="upper right")
plt.title("Residual Errors")
plt.show()
