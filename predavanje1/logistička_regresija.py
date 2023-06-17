import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,ConfusionMatrixDisplay

x_train=np.array([[1],[2],[3],[4],[5],[6]])
y_train=np.array([0,0,0,1,1,1])

plt.scatter(x_train,y_train)
plt.show()

classifier=LogisticRegression()
classifier.fit(x_train,y_train)

x_test=np.array([[2.5],[3.51]])
y_test=np.array([0,1])

y_pred=classifier.predict(x_test)

report=classification_report(y_test,y_pred)
cnf_matrix=confusion_matrix(y_test,y_pred)

#print(report,cnf_matrix)

confusion_matrix_plot =ConfusionMatrixDisplay(cnf_matrix)

confusion_matrix_plot.plot()
plt.show()






