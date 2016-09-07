import pandas as pd
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier

url = "/home/elliotnam/project/brfss/brfss_extracted_data.csv"

colNames = ["state","gender","race","excercise_in_30","sleep_duration",
            "employee_status","income_level","use_tobacco","alcohol_con","smoked_least_100","freq_smoke",
            "stopped_smoked_12month","interval_last_smoke","still_asthema"]

colNames2 = ["state","gender","race","excercise_in_30","sleep_duration",
            "employee_status","income_level","use_tobacco","alcohol_con","freq_smoke",
            "interval_last_smoke","still_asthema"]

dataframe = pd.read_csv(url)
dataframe.drop(dataframe.index[[0]])


array = dataframe.values

X = array[:,0:14]
Y = array[:,14]

#Y = label_binarize(Y, classes=['1','2'])
num_folds = 5
num_instances = len(X)
seed = 7

Y = label_binarize(Y, classes=[1,2])
n_classes = Y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2,
                                                    random_state=0)

def runRoc():
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    #classifier = OneVsRestClassifier(DecisionTreeClassifier(max_depth=5))
    #classifier = DecisionTreeClassifier(max_depth=5)
    #classifier = SVC(C=1,gamma=0.1,kernel='rbf',probability=True)
    #classifier =  LogisticRegression(C=0.001,n_jobs=-1)
    num_trees = 100
    #classifier =  GradientBoostingClassifier(learning_rate=0.005,n_estimators=num_trees, random_state=seed,max_depth=5, min_samples_split=1600,min_samples_leaf=50, subsample=0.8)
    classifier = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
    #num_trees = 250
    #classifier = XGBClassifier(n_estimators=num_trees)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    for i, (train, test) in enumerate(kfold):
        probas_ = classifier.fit(X[train], Y[train].ravel()).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(Y[test].ravel(), probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1)#, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    mean_tpr /= len(kfold)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Ada Boost Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def runOneOut():
    num_folds = 2
    num_instances = len(X)
    num_trees = 100
    loocv = cross_validation.LeaveOneOut(n=num_instances)
    model = LogisticRegression()
    #results = cross_validation.cross_val_score(model, X, Y.ravel(), cv=loocv,n_jobs=-1)
    #print("LogisticRegression Accuracy: %.3f%% (%.3f%%)") % (results.mean() * 100.0, results.std() * 100.0)

    model = DecisionTreeClassifier(max_depth=5)
    #results = cross_validation.cross_val_score(model, X, Y.ravel(), cv=loocv,n_jobs=-1)
    #print("Decision Tree Accuracy: %.3f%% (%.3f%%)") % (results.mean() * 100.0, results.std() * 100.0)

    #model = GradientBoostingClassifier(learning_rate=0.005,n_estimators=num_trees, random_state=seed,max_depth=5, min_samples_split=1600,min_samples_leaf=50, subsample=0.8)
    #results = cross_validation.cross_val_score(model, X, Y.ravel(), cv=loocv,n_jobs=-1)


    model = XGBClassifier()
    results = cross_validation.cross_val_score(model, X, Y.ravel(), cv=loocv, n_jobs=-1)
    print("Xboost Accuracy: %.3f%% (%.3f%%)") % (results.mean() * 100.0, results.std() * 100.0)

runRoc()

runOneOut()


