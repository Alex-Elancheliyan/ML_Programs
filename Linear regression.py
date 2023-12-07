import numpy as np
from statistics import mean
import matplotlib.pyplot as plt


def calculate_slope_intercept(xvalues,yvalues):
    m = (((mean(xvalues) * mean(yvalues)) - mean(xvalues * yvalues)) /
        ((mean(xvalues) * mean(xvalues))-mean(xvalues * xvalues)))
    b = mean(yvalues)- m * mean(xvalues)
    return m,b

def linear_regression():
    regression_line= [(m * x) + b for x in xvalues]
    plt.style.use('ggplot')
    plt.title('Training Data & Regression Line')
    plt.scatter(xvalues, yvalues, color='#003f72', label='Training Data')
    plt.plot(xvalues, regression_line, label='Reg Line')
    plt.legend(loc='best')
    plt.show()



def test_data():
    predict_xvalue = 7
    predict_yvalue = (m * predict_xvalue)+ b
    print('Test Data for x :    ', predict_xvalue, '   ',
          'Test Data for y :    ',predict_yvalue)

    plt.title('train, Reg Line & Test Value')
    plt.scatter(xvalues,yvalues,color='#003F72',label='Data')
    plt.scatter(predict_xvalue,predict_yvalue,color='#FF0000',label='Predicted Value')
    plt.legend(loc='best')
    plt.show()



def validate_results():
    predict_xvalues= np.array([2.5,3.5,4.5,5.5,6.5],dtype=np.float64)
    predict_yvalues=[(m*x) +b for x in predict_xvalues]
    print('Validation Data set')
    print('Xvalues:', predict_xvalues)
    print('Y values:', predict_yvalues)
    plt.style.use('ggplot')
    plt.title('Actual Prediction')
    plt.scatter(predict_xvalues, predict_yvalues, color='#003f72', label='Prediction Data')
    plt.legend(loc='best')
    plt.show()


xvalues = np.array([1,2,3,4,5], dtype=np.float_)
yvalues = np.array([2,4,6,8,10], dtype=np.float_)


m, b = calculate_slope_intercept(xvalues,yvalues)
 
print('slope:',m,'Intercept:',b)

linear_regression()
test_data()
validate_results()