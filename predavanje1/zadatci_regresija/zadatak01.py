#1. Zadatak
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score
#a. Učitajte California housing skup podataka iz Scikit-learn dataset modula.
housing_x,housing_y=datasets.fetch_california_housing(return_X_y=True, as_frame=True)
#print(housing_x.shape)
#b. Eksplorativnom analizom podataka istražite odnose među varijablama.
def feature_relations(x_values,y_values):
    nrows,ncolumns=2,4
    fig=plt.figure()
    fig.suptitle("Feature relations")
    for i,column_name in enumerate(x_values):
        ax=fig.add_subplot(nrows,ncolumns,i+1)
        ax.scatter(x_values[column_name],y_values)
        ax.set_title(column_name)
        
    plt.show()


 
feature_relations(housing_x,housing_y)
#c. Odaberite jednu ulaznu veličinu i podijelite podatke na trening i testni skup u
#odnosu 70%-30%.
#d. Izgradite linearni model.
#e. Vrjednovanjem modela odredite srednju kvadratnu pogrešku i koeficijent
#determinacije
def fir_and_visualize_model(hosuing_X,housing_y):
    housing_y=housing_y.values
    nrows,ncols=2,4
    fig=plt.figure()
    fig.suptitle("Linear regression results")
    for i,feature_name in enumerate(hosuing_X):
        hosuing_X_feature=hosuing_X[feature_name].values
        hosuing_X_feature=np.expand_dims(hosuing_X_feature,axis=1)
        housing_x_train,\
        housing_x_test,\
        housing_y_train,\
        housing_y_test=train_test_split(hosuing_X_feature,housing_y,test_size=0.3,random_state=1)
        lm=linear_model.LinearRegression()
        lm.fit(housing_x_train,housing_y_train)
        housing_y_pred=lm.predict(housing_x_test)
        mse=mean_squared_error(housing_y_test,housing_y_pred)
        r2_result=r2_score(housing_y_test,housing_y_pred)
        
        ax=fig.add_subplot(nrows,ncols,i+1)
        ax.set_title("{}\nmse: {:.2f}\nr2: {:.2f}".format(feature_name, mse, r2_result))
        ax.scatter(housing_x_test,housing_y_test,color='black',label='real values')
        ax.plot(housing_x_test, housing_y_pred, color="blue", marker="o", linewidth=3, label="predicted values")
    plt.show()

fir_and_visualize_model(housing_x,housing_y)
    
    

#f. Eksperimentirajte s različitim ulaznim veličinama. Koja veličina je najprikladnija?