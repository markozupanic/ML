#feature selection proc
#explorativna analiza podataka
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

NUM_OF_POLYNOM_DEGREES = 10

diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True, as_frame=True)

diabetes_X = diabetes_X["s5"].to_frame()

nrows, ncols = 2, NUM_OF_POLYNOM_DEGREES // 2
fig = plt.figure()
fig.suptitle("Polynomial regression results")

for i in range(1, NUM_OF_POLYNOM_DEGREES + 1):
    poly = PolynomialFeatures(degree=i)
    diabetes_X_features = poly.fit_transform(diabetes_X)
    diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = train_test_split(
        diabetes_X_features, diabetes_y, test_size=0.2, random_state=1
    )

    lm = LinearRegression()
    lm.fit(diabetes_X_train, diabetes_y_train)

    diabetes_y_pred = lm.predict(diabetes_X_test)
    mse = mean_squared_error(diabetes_y_test, diabetes_y_pred)
    r2_result = r2_score(diabetes_y_test, diabetes_y_pred)

    ax = fig.add_subplot(nrows, ncols, i)
    ax.set_title("Polynomial degree: {}\nmse: {:.2f}\nr2: {:.2f}".format(i, mse, r2_result))
    ax.scatter(diabetes_X_test[:, 1], diabetes_y_test, color='black')
    ax.scatter(diabetes_X_test[:, 1], diabetes_y_pred, color='red')
   # ax.plot(diabetes_X_test[:, 1], diabetes_y_pred, color='blue', linewidth=3)

plt.show()




































