from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,ConfusionMatrixDisplay
import matplotlib.pyplot as plt

iris=load_iris()

X=iris.data
y=iris.target

class_names=iris.target_names

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

classifier=LogisticRegression()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

cnf_matrix=confusion_matrix(y_test,y_pred)
clf_report=classification_report(y_test,y_pred)

print(clf_report)

confusion_matrix_plot=ConfusionMatrixDisplay(cnf_matrix,display_labels=class_names)

confusion_matrix_plot.plot()
plt.show()






