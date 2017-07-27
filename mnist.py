from sklearn.datasets import fetch_mldata

#fetch MNIST dataset
mnist = fetch_mldata('MNIST original')

#
#print mnist

X, y = mnist["data"], mnist["target"]

#70000 images, each image has 784 features -> each image is 28x28 pixels
#  each feature represents one pixel's intensity - from 0 to 255 (white to black)
#print X.shape
#print y.shape

#take one instance
some_digit = X[36000]
#reshape it to 28x28 array
some_digit_image = some_digit.reshape(28, 28)
#print the label of the chosen feature
#print y[36000]
#display it
import matplotlib
import matplotlib.pyplot as plt
#plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
#plt.axis("off") #commented out because its better with the axis.
#plt.show()

#first 60000 images are training set, last 10000 images are test set
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#shuffle training set
import numpy as np
np.random.seed(17)
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

######
##Training a binary classifier

#keeping it simple:
#try to identifiy one digit
#binary classifier - 5 or not 5

#true if 5, false if any other digit
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

#example: 5 digits and five booleans
print y_train[:5]
print y_train_5[:5]

#1) pick a classifier and train it.
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
#train it on labels that are true/false depending on whether the digit is 5 or not
sgd_clf.fit(X_train, y_train_5)


print y[36000], "-> prediction: ", sgd_clf.predict([some_digit])

#evaluating the model

######
#Performance measures

#cross-validation
from sklearn.model_selection import cross_val_score
#K-fold cross-validation - splitting training set into 3-folds (cv=3)
#then making predictions and evaluating them on each fold
#using a model trained on the remaining folds
cvs = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
print "Cross validation score:", cvs
#around 95% accuracy. Nothing too amazing -> around 10% of all digits are 5. If you always
#guess "not 5", you will be correct in cca 90% of the time

#! accuracy is generally not the preffered performance measure for classifiers!
# especially when dealing with skewed datasets

########
#Confusion matrix
##much better way to evaluate the performance of a classifier is with confusion matrix
#count number of times instances of class A are classified as class B

from sklearn.model_selection import cross_val_predict

#compute set of predictions so theey can be compared to actual targets
#function performs K-fold cross-validation, returns predictions made on each test fold.
#paramter method="predict" is default value
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

print "Cross validation predictions for first 5 instances:", y_train_pred[:5]

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train_5, y_train_pred)
#each row presents actual class
#each column presents predicted class
print "\nConfusion matrix:\n", cm
print "First row considers non-5 images:\n\t", cm[0,0], "correctly classified as non-5s (TRUE NEGATIVES) \n\t", cm[0,1], "wrongly classified as 5s (FALSE POSITIVES)"
print "\t", cm[1,0], "wrongly classified as non-5s (FALSE NEGATIVES)\n\t", cm[1,1], "correctly classified as 5s (TRUE POSITIVES)"
print "Perfect classifier: only TRUE POSITIVES and TRUE NEGATIVES\n"


#precision of the classifier: the accuracy of the positive predictions
#precision = TP / (TP + FP)

#precision is used with recall - sensitivity or true positive rate (TPR)
#this is the ratio of positive instances correctly detected by classifier
#recall = TP / ( TP + FN )


####Precision and recall
from sklearn.metrics import precision_score, recall_score
ps = precision_score(y_train_5,y_train_pred)
print "Precision score:", ps
print "precision = TP / (TP + FP) =>", cm[1,1]/float((cm[1,1] + cm[0,1]))

rs = recall_score(y_train_5, y_train_pred)
print "Recall score:", rs
print "recall = TP / ( TP + FN ) =>", cm[1,1]*1./(cm[1,1] + cm[1,0])

#F1 score
#Combining precision and recall into a single metric - specially if you need to compare two classifiers
#F1 score is the harmonic mean of precision and recall
#F1 = TP / (TP + ((FN + FP)/2))
from sklearn.metrics import f1_score
fs = f1_score(y_train_5, y_train_pred)
print "f1 score: ", fs
print "F1 = TP / (TP + ((FN + FP)/2)) =>", cm[1,1]*1. / (cm[1,1] + ((cm[1,0] + cm[0,1])/2))
#F1 score favours classifiers with similar precision and recall (not always what you want)
#Increasing precision reduces recall, and vice versa - precision/recall tradeoff
print "\n"

####Precision/Recall tradeoff
y_scores = sgd_clf.decision_function([some_digit])
print "decision_function method returns score for each instance. For example for instance X[36000] (digit 5) returns score ", int(y_scores)*1.

y_scores = sgd_clf.decision_function([X[2]])
print "decision_function method returns score for each instance. For example for instance X[2] (not digit 5) returns score ", int(y_scores)*1.

##!! Classifier used earlier - SGDClassifier uses a threshold equal to 0 - method predict() uses it. The threshold can be changed
#and one can manually calculate True/False for any digit ->
threshold = 20000
y_some_digit_pred = (y_scores > threshold)
print y_some_digit_pred
#raising threshold decreases recall

##How to decide which threshold to use?
#get the scores of all instances in the training set using cross_val_predict() and specifying you want to return decision scores
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
#precision and recall can be computed for all possible thresholds
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="center left")
    plt.ylim([0, 1.1])

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()
#from the plot, the threshold can be simply selected. The threshold that gives best precision/recall tradeoff.
#if someone wants 90% precision, you check the plot - in this case it is around 140000 (the x value for y=0.9)
#BUT high precision classifier might have low recall!

#Another way to select good precision/recall tradeoff is to plot precision against recall
plt.plot(recalls, precisions)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()
