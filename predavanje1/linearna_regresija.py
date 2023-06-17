import sklearn.linear_model as lm
import numpy as np
import matplotlib.pyplot as plt

x_train=np.array([[0],[1],[2]])
y_train=np.array([0,1,2])

plt.scatter(x_train,y_train)
plt.show()

linear_model=lm.LinearRegression()
linear_model.fit(x_train,y_train)

print(linear_model.intercept_)
print(linear_model.coef_)

x_test=np.array([[0.5],[3]])
y_predict=linear_model.predict(x_test)
print(y_predict)
























