from sklearn.model_selection import train_test_split
from preprocessor import preprocessor
from svm import MultiSVM
import numpy as np

p = preprocessor()
X_train, X_test, y_train, y_test = train_test_split(p.processed_features, p.labels, test_size=0.2, random_state=0)
'''from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='poly') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))'''
svm = MultiSVM()
svm.fit(X_train, y_train)
svm.save()
#svm.show()
y_pred = svm.predict(X_test)
y_pred = np.reshape(y_pred, (1, len(y_pred)))
y_test[y_test == 'negative'] = 0
y_test[y_test == 'positive'] = 2
y_test[y_test == 'neutral'] = 1
#y_test = np.reshape(y_test.astype(np.float), (1, len(y_test)))
#print(np.unique(y_pred))
#
#from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
#mse = np.square(np.subtract(y_test ,y_pred)).mean() 
#print(mse)
#print(np.sum(np.equal(y_test, y_pred))/float(y_test.size))
from sklearn.metrics import precision_recall_fscore_support as score, accuracy_score

precision, recall, fscore, support = score(np.reshape(y_test.astype(np.float), (2928,1)), np.reshape(y_pred, (2928,1)))

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

print(accuracy_score(np.reshape(y_test.astype(np.float), (2928,1)), np.reshape(y_pred, (2928,1))))
