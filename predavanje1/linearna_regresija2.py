import sklearn.linear_model 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score



def show_feature_relations(x_values,y_values):
    nrows,ncolumns=2,5
    fig=plt.figure()
    fig.suptitle("Feature relations")
    for i,column_name in enumerate(x_values):
        ax=fig.add_subplot(nrows,ncolumns,i+1)
        ax.scatter(x_values[column_name],y_values)
        ax.set_title(column_name)
        
    plt.show()

diabetes_x,diabetes_y=datasets.load_diabetes(return_X_y=True,as_frame=True)
#print(diabetes_x.shape)

#show_feature_relations(diabetes_x,diabetes_y)


def fit_and_visualoze_model(diabetes_X,diabetes_y):
    diabetes_y = diabetes_y.values
    nrows, ncols = 2, 5
    fig = plt.figure()
    fig.suptitle("Linear regression results")
    for i, feature_name in enumerate(diabetes_X):
        diabetes_X_feature = diabetes_X[feature_name].values
        diabetes_X_feature = np.expand_dims(diabetes_X_feature, axis=1)
        diabetes_x_train,\
        diabetes_x_test,\
        diabetes_y_train,\
        diabetes_y_test = train_test_split(diabetes_X_feature, diabetes_y, test_size=0.2, random_state=1)

        lm = linear_model.LinearRegression()
        lm.fit(diabetes_x_train, diabetes_y_train)

        diabetes_y_pred = lm.predict(diabetes_x_test)
        mse = mean_squared_error(diabetes_y_test, diabetes_y_pred)
        r2_result = r2_score(diabetes_y_test, diabetes_y_pred)

        ax = fig.add_subplot(nrows, ncols, i + 1)
        ax.set_title("{}\nmse: {:.2f}\nr2: {:.2f}".format(feature_name, mse, r2_result))
        ax.scatter(diabetes_x_test, diabetes_y_test, color="black", label="real values")
        ax.plot(diabetes_x_test, diabetes_y_pred, color="blue", marker="o", linewidth=3, label="predicted values")
    plt.show()
        
        
    

fit_and_visualoze_model(diabetes_x,diabetes_y)


































































