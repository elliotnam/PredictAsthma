import pandas as pd
from sklearn import cross_validation,metrics
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy
from sklearn.feature_selection import RFE
from sklearn.grid_search import GridSearchCV
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from xgboost import plot_tree
from matplotlib import pyplot
from sklearn.feature_selection import SelectFromModel
from numpy import sort
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import LabelEncoder



url = "/home/elliotnam/project/brfss/brfss_extracted_data.csv"
colNames = ["state","gender","race","ever_told_asthma","excercise_in_30","sleep_duration",
            "employee_status","income_level","use_tobacco","alcohol_con","smoked_least_100","freq_smoke",
            "stopped_smoked_12month","interval_last_smoke"]


colNames2 = ["state","gender","race","excercise_in_30","sleep_duration",
            "employee_status","income_level","use_tobacco","alcohol_con","freq_smoke",
            "interval_last_smoke","still_asthema"]


colNames3 = ["state","gender","race","excercise_in_30","sleep_duration",
            "employee_status","income_level","use_tobacco","alcohol_con","freq_smoke",
            "interval_last_smoke"]

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

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2,
                                                    random_state=0)
kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)

def runLogisticRegression():

    model = LogisticRegression()
    results = cross_validation.cross_val_score(model, X, Y.ravel(), cv=kfold)
    i = 0

    print(results.mean())


def runLinearDiscriment():
    model = LinearDiscriminantAnalysis()
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
    print(results.mean())

def runKNNClassification():
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds,
                                   random_state=seed)
    model = KNeighborsClassifier()
    results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
    print(results.mean())

def runNaiveBayes():
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    model = GaussianNB()
    results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
    print(results.mean())


def runDecisionTree():
    scoring = 'accuracy'
    model = DecisionTreeClassifier()
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    results = cross_validation.cross_val_score(model, X, Y.ravel(), cv=kfold, scoring=scoring, n_jobs=-1)


    #results = cross_validation.cross_val_score(model, X, Y, cv=kfold, scoring=scoring, n_jobs=-1)
    print(results.mean())
    i =0
    for train_index, test_index in kfold:
        if i == 2:
            model.fit(X[train_index],Y[train_index].ravel())
            predictions = model.predict(X[test_index])
            dtrain_predprob = model.predict_proba(X[train_index])[:,1]
            feat_imp = pd.Series(model.feature_importances_,colNames).sort_values(ascending=False)
            feat_imp.plot(kind='bar', title='Feature Importances')
            plt.ylabel('Decision Tree Feature Importance Score')
            plt.show()
        i += 1


def runXGBoost():
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    #model = XGBClassifier(max_depth=3,n_estimators=250)
    model = XGBClassifier()
    print(model)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    model = XGBClassifier()
    n_estimators = [100, 200, 300, 400, 500]
    learning_rate = [0.0001, 0.001, 0.01, 0.1]
    param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators)
    kfold = StratifiedKFold(Y.ravel(), n_folds=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="log_loss", n_jobs=-1, cv=kfold)
    result = grid_search.fit(X, Y.ravel())
    # summarize results
    print("Best: %f using %s" % (result.best_score_, result.best_params_))


    learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
    param_grid = dict(learning_rate=learning_rate)
    kfold = StratifiedKFold(Y.ravel(), n_folds=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="log_loss", n_jobs=-1, cv=kfold)
    result = grid_search.fit(X, Y.ravel())
    # summarize results
    print("Best: %f using %s" % (result.best_score_, result.best_params_))

    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    results = cross_validation.cross_val_score(model, X, Y.ravel(), cv=kfold)
    print(results.mean())
    print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

    print(model.feature_importances_)

    n_estimators = range(50, 400, 50)
    param_grid = dict(n_estimators=n_estimators)
    kfold = StratifiedKFold(Y.ravel(), n_folds=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="log_loss", n_jobs=-1, cv=kfold)
    result = grid_search.fit(X, Y.ravel())
    # summarize results
    print("Best: %f using %s" % (result.best_score_, result.best_params_))
    means, stdevs = [], []
    for params, mean_score, scores in result.grid_scores_:
        stdev = scores.std()
    means.append(mean_score)
    stdevs.append(stdev)
    print("%f (%f) with: %r" % (mean_score, stdev, params))


    max_depth = range(1, 11, 2)
    print(max_depth)
    param_grid = dict(max_depth=max_depth)
    kfold = StratifiedKFold(Y.ravel(), n_folds=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="log_loss", n_jobs=-1, cv=kfold,
                               verbose=1)
    result = grid_search.fit(X, Y.ravel())
    # summarize results
    print("Best: %f using %s" % (result.best_score_, result.best_params_))
    means, stdevs = [], []
    for params, mean_score, scores in result.grid_scores_:
        stdev = scores.std()
    means.append(mean_score)
    stdevs.append(stdev)
    print("%f (%f) with: %r" % (mean_score, stdev, params))


#    thresholds = sort(model.feature_importances_)
##    for thresh in thresholds:
    # select features using threshold
#        selection = SelectFromModel(model, threshold=thresh, prefit=True)
#        select_X_train = selection.transform(X_train)
#        # train model
#        selection_model = XGBClassifier()
#        selection_model.fit(select_X_train, y_train.ravel())
#        # eval model
#        select_X_test = selection.transform(X_test)
#        y_pred = selection_model.predict(select_X_test)
#        predictions = [round(value) for value in y_pred]
#        accuracy = accuracy_score(y_test, predictions)
#        print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1],accuracy*100.0))

def runSVM():
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    model = SVC()
    results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
    print(results.mean())

def tunningLogisticRegression():
    #estimater = LogisticRegression()
    research =GridSearchCV(cv=None,
                 estimator=LogisticRegression(C=1.0, intercept_scaling=1, dual=False, fit_intercept=True,
                                              penalty='l2', tol=0.0001),
                 param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]})

    research.fit(X, Y)
    print(research.best_score_)
    print(research.best_params_)

#def tunningGradientBoost():


def testGradeintBoost():
    param_test1 = {'n_estimators': range(20, 81, 10)}
    kfold = cross_validation.KFold(n=num_instances, n_folds=2, random_state=seed)
    gsearch1 = GridSearchCV(
        estimator=GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500, min_samples_leaf=50, max_depth=8,
                                             max_features='sqrt', subsample=0.8, random_state=10),
        param_grid=param_test1, scoring='roc_auc', n_jobs=4, iid=False, cv=5)

    for i, (train, test) in enumerate(kfold):
        gsearch1.fit(X[train], Y[train].ravel())
        print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)
        print()

    param_test2 = {'max_depth':range(5,16,2), 'min_samples_split':range(200,1001,200)}
    gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=30, max_features='sqrt', subsample=0.8, random_state=10),
    param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

    for i, (train, test) in enumerate(kfold):
        gsearch2.fit(X[train], Y[train].ravel())
        print(gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_)
        print()

    param_test3 = {'min_samples_split':range(1000,2100,200), 'min_samples_leaf':range(30,71,10)}
    gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=30,max_depth=5,max_features='sqrt', subsample=0.8, random_state=10),
    param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    for i, (train, test) in enumerate(kfold):
        gsearch3.fit(X[train], Y[train].ravel())
        print(gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_)
        print()
    param_test4 = {'max_features':range(6,10,2)}
    gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=30,max_depth=5, min_samples_split=1600, min_samples_leaf=50, subsample=0.8, random_state=7),
    param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    for i, (train, test) in enumerate(kfold):
        gsearch4.fit(X[train], Y[train].ravel())
        print(gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_)

    param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
    gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=30,max_depth=5,min_samples_split=1600, min_samples_leaf=50, subsample=0.8, random_state=7,max_features=11),
    param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

    for i, (train, test) in enumerate(kfold):
        gsearch5.fit(X[train], Y[train].ravel())
        print(gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_)

    predictors = [x for x in X[0]]
    gbm_tuned_1 = GradientBoostingClassifier(learning_rate=0.05, n_estimators=30,max_depth=5, min_samples_split=1600,min_samples_leaf=50, subsample=0.8, random_state=7)
    modelfit(gbm_tuned_1, X_train, predictors)

def modelfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
    rcParams['figure.figsize'] = 12,4
    target = 'Disbursed'
    IDcol = 'ID'
    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'])

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain['Disbursed'], cv=cv_folds,
                                                    scoring='roc_auc')

    # Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob)

    if performCV:
        print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (
            np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score))

    # Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')


def runCompareAlgorithms():
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('DCN', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    #models.append(('SVM', SVC()))

    num_trees = 250
    models.append(('XGB',XGBClassifier(n_estimators=num_trees)))

    models.append(('GBC',GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)))

    num_trees = 30
    models.append(('ABC',AdaBoostClassifier(n_estimators=num_trees, random_state=seed)))

    num_trees = 100
    max_features = 13
    models.append(('RFC',RandomForestClassifier(n_estimators=num_trees, max_features=max_features)))
    # evaluate each model in turn
    results = []
    names = []
    scoring = 'accuracy'
    for name, model in models:
        kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
        cv_results = cross_validation.cross_val_score(model, X, Y.ravel(), cv=kfold, scoring=scoring,n_jobs=-1)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle("Algorithm Comparison")
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

def runBaggedDecisionTree():
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    cart = DecisionTreeClassifier()
    num_trees = 100
    model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
    results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
    print(results.mean())

def runRandomForest():
    num_trees = 100
    max_features = 3
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
    results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
    print(results.mean())

def runExtraTrees():
    num_trees = 100
    max_features = 12
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)
    results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
    print(results.mean())

def runadaBust():
    num_trees = 30
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
    results = cross_validation.cross_val_score(model, X, Y.ravel(), cv=kfold)
    print(results.mean())
    print(results.mean())
    i =0
    for train_index, test_index in kfold:
        if i == 2:
            model.fit(X[train_index],Y[train_index].ravel())
            predictions = model.predict(X[test_index])
            dtrain_predprob = model.predict_proba(X[train_index])[:,1]
            feat_imp = pd.Series(model.feature_importances_,colNames).sort_values(ascending=False)
            feat_imp.plot(kind='bar', title='Feature Importances')
            plt.ylabel('Ada Boost Feature Importance Score')
            plt.show()
        i += 1


def runGradientBust():
    num_trees = 100
    print('run gradeint')
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    model = GradientBoostingClassifier(learning_rate=0.005,n_estimators=num_trees, random_state=seed,max_depth=5, min_samples_split=1600,min_samples_leaf=50, subsample=0.8)
    results = cross_validation.cross_val_score(model, X, Y.ravel(), cv=kfold,scoring='roc_auc')
    print(results)

    i = 0
    for train_index,test_index in kfold:
        if i == 2:
            model.fit(X[train_index],Y[train_index].ravel())
            predictions = model.predict(X[test_index])
            dtrain_predprob = model.predict_proba(X[train_index])[:,1]
            feat_imp = pd.Series(model.feature_importances_,colNames).sort_values(ascending=False)
            feat_imp.plot(kind='bar', title='Feature Importances')
            plt.ylabel('Gradient Boost Feature Importance Score')
            plt.show()
        i += 1


def runVotingLearning():
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    # create the sub models
    estimators = []
    model1 = LogisticRegression()
    estimators.append(("logistic", model1))
    model2 = DecisionTreeClassifier()
    estimators.append(("cart", model2))
    #model3 = SVC()
    model3=GaussianNB()
    estimators.append(("basyen", model3))
    # create the ensemble model
    ensemble = VotingClassifier(estimators)
    results = cross_validation.cross_val_score(ensemble, X, Y, cv=kfold)
    print(results.mean())


def runTunning():
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    # create the sub models
    #estimators = []
    model1 = LogisticRegression()
    #estimators.append(("logistic", model1))
    model2 = DecisionTreeClassifier()
    #estimators.append(("cart", model2))
    #model3 = SVC()
    model3=GaussianNB()
    #estimators.append(("basyen", model3))
    # create the ensemble model
    ensemble = VotingClassifier(estimators)
    param_grid = {"alpha": uniform()}
    iterations = 100
    research = RandomizedSearchCV(estimator=ensemble,param_distributions=param_grid,n_iter=iterations,random_state=seed)
    research.fit(X,Y)
    print(research.best_score_)
    print(research.best_estimator_.alpha)

def runUnivariateSelection():
    test = SelectKBest(score_func=chi2, k=7)
    fit = test.fit(X, Y)
    # summarize scores
    numpy.set_printoptions(precision=3)
    print(fit.scores_)
    features = fit.transform(X)
    # summarize selected features
    print(features[0:7, :])


def runRecursiveFeaureDelete():
    model = LogisticRegression()
    rfe = RFE(model, 7)
    fit = rfe.fit(X, Y)
    print("Num Features: %d") % fit.n_features_
    print("Selected Features: %s") % fit.support_
    print("Feature Ranking: %s") % fit.ranking_


def runExtraTreeFeatureImportance():
    model = ExtraTreesClassifier()
    model.fit(X, Y)
    print(model.feature_importances_)


runLogisticRegression()
runLinearDiscriment()
runKNNClassification()
runNaiveBayes()
runDecisionTree()
runSVM()
runCompareAlgorithms()
runBaggedDecisionTree()
runRandomForest()
runExtraTrees()
runadaBust()
runGradientBust()
runVotingLearning()
runTunning()
runUnivariateSelection()
runRecursiveFeaureDelete()
runExtraTreeFeatureImportance()

tunningLogisticRegression()
testGradeintBoost()
runXGBoost()