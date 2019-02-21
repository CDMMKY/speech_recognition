from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

COUNT_SPEAKERS = 20
COUNT_RECORDS_TRAINING = 40
COUNT_RECORDS_TEST = 50 - COUNT_RECORDS_TRAINING
IS_HTK = True
FEATURED_DIRECTORY = r"D:\study\nir\signs" if IS_HTK else r"D:\study\nir\signs2"
IS_DIMENSIONALITY = True
DIMENSIONALITY = 100
TOL = 0.000001

CLASSIFIER_MAP = {
    'svm-linear': SVC(kernel='linear', max_iter=5000, tol=TOL, probability=True),
    'svm-rbf': SVC(kernel='rbf', max_iter=5000, tol=TOL, probability=True),
    'svm-poly': SVC(kernel='poly', max_iter=5000, tol=TOL, probability=True),
    'lda': LinearDiscriminantAnalysis(tol=TOL),
    'bayes': GaussianNB(),
    'tree': DecisionTreeClassifier(random_state=0),
    'adaboost': AdaBoostClassifier()
}