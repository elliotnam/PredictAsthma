using DataFrames
using ScikitLearn
using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn.CrossValidation: KFold
using ScikitLearn.GridSearch: GridSearchCV
using ScikitLearn.CrossValidation: cross_val_predict
using ScikitLearn.CrossValidation: LeaveOneOut
using ScikitLearn.Pipelines: Pipeline
#using ScikitLearn.Preprocessing: label_binarize
using PyPlot
using PyCall
@pyimport scipy

@pyimport scipy.stats as stats

@sk_import linear_model: LogisticRegression
@sk_import discriminant_analysis: LinearDiscriminantAnalysis
@sk_import ensemble: GradientBoostingClassifier
@sk_import ensemble: RandomForestClassifier
@sk_import preprocessing: PolynomialFeatures
@sk_import linear_model: LinearRegression
@sk_import metrics: roc_curve
@sk_import metrics: auc
@sk_import preprocessing: label_binarize

#@sk_import xgboost: XGBClassifier
#print(pwd())
dData = readtable("brfss_julia/brfss_extracted_data.csv")
#dData = readtable("project/julia_scikit/brfss_extracted_data.csv")
number_folds =5
X = convert(Array,dData[1:end-1]);
y = label_binarize(convert(Array,dData[end]),classes=[1,2]);
#print(y)
kf = KFold(size(X)[1],n_folds=number_folds,random_state=7)

function runRandomForest(X::Array{Int64,2},y::Array{Int64,2})
  param_grid = Dict("max_depth"=> [3, nothing],
              "max_features"=> [1, 3, 10],
         #     "min_samples_split"=> [1, 3, 10],
              "min_samples_leaf"=> [1, 3, 10],
              "bootstrap"=>[true, false],
              "criterion"=> ["gini", "entropy"])
  model = RandomForestClassifier(n_estimators=20,n_jobs=-1)
  grid_search = GridSearchCV(model,param_grid,cv=kf)
  ScikitLearn.fit!(grid_search,X,y)
  report(grid_search.grid_scores_)
end


function runXGBoost(X::Array{Int64,2},y::Array{Int64,2})
  param_grid = Dict("learning_rate"=> [0.0001,0.001,0.01,0.1],
            "max_depth"=> [3, nothing],
            "min_samples_leaf"=> [1, 3, 10])

  model = GradientBoostingClassifier(n_estimators=20)
  grid_search = GridSearchCV(model,param_grid,cv=kf)
  ScikitLearn.fit!(grid_search,X,y)
  report(grid_search.grid_scores_)
  #println("best parameters: $(grid_search.best_params_)")
  #print(result)
end

function runLogisticRegression(X::Array{Int64,2},y::Array{Int64,2})
  tt = cross_val_score(LogisticRegression(),X,y;cv=kf)
  print(tt)
end

function runLinearDiscriment(X::Array{Int64,2},y::Array{Int64,2})
  tt = cross_val_score(LinearDiscriminantAnalysis(),X,y;cv=kf)
  print(tt)
end

function test(X::Array{Int64,2},y::Array{Int64,2})
  gridsearch = GridSearchCV(LogisticRegression(),Dict(:C=>0.1:0.1:2.0))
  ScikitLearn.fit!(gridsearch,X,y)
  println("best parameters: $(gridsearch.best_params_)")
end


function report(grid_scores, n_top=3)
    top_scores = sort(grid_scores, by=x->x.mean_validation_score, rev=true)[1:n_top]
    for (i, score) in enumerate(top_scores)
        println("Model with rank:$i")
        @printf("Mean validation score: %.3f (std: %.3f)\n",
                score.mean_validation_score,
                std(score.cv_validation_scores))
        println("Parameters: $(score.parameters)")
        println("")
    end
end

function runCrossPredictionGraph(X::Array{Int64,2},y::Array{Int64,2})
  model = LogisticRegression()
  predicted = cross_val_predict(model, X, y, cv=kf)
  println(predicted)
  println(y)
  scatter(y, predicted)
  plot([minimum(y), maximum(y)], [minimum(y), maximum(y)], "k--", lw=4)
  xlabel("Measured")
  ylabel("Predicted");
end

function runModelGraph(X::Array{Int64,2},y::Array{Int64,2})
  degrees = [1, 4, 15]
  figure(figsize=(14, 5))
  for (i, degree) in enumerate(degrees)
    ax = subplot(1, length(degrees), i)
    setp(ax, xticks=(), yticks=())

    polynomial_features = PolynomialFeatures(degree=degree, include_bias=false)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    ScikitLearn.fit!(pipeline, X'', y)

    # Evaluate the models using crossvalidation
    scores = cross_val_score(pipeline, X'', y, scoring="mean_squared_error", cv=kf)

    X_test = linspace(0, 1, 100)
    plot(X_test, ScikitLearn.predict(pipeline, X_test''), label="Model")
    plot(X_test, true_fun(X_test), label="True function")
    scatter(X, y, label="Samples")
    xlabel("x")
    ylabel("y")
    xlim((0, 1))
    ylim((-2, 2))
    legend(loc="best")
    title(@sprintf("Degree %d\nMSE = %.2e +/- %.2e", degree, -mean(scores), std(scores)))
  end
end

function runRoc(X::Array{Int64,2},y::Array{Int64,2})
  print(typeof(X))
  print(typeof(y))
  num_trees = 100
  model = GradientBoostingClassifier(n_estimators=20)

  mean_tpr = 0.0
  mean_fpr = linspace(0, 1, 100)
  all_tpr = []

  for (id,trainMatrix) in enumerate(kf)
    ScikitLearn.fit!(model,X[trainMatrix[1],:],y[trainMatrix[1]])
    probas_ = predict_proba(model,X[trainMatrix[2],:])
    fpr, tpr, thresholds = roc_curve(y[trainMatrix[2],:], probas_[:,2])
    mean_tpr += scipy.interp(mean_fpr, fpr, tpr)
    mean_tpr[1] = 0.0
    roc_auc = auc(fpr, tpr)
    plot(fpr, tpr, lw=1)#, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
  end

  plot([0, 1], [0, 1], "--", color=(0.6, 0.6, 0.6), label="Luck")

  mean_tpr /= number_folds
  mean_tpr[end] = 1.0
  mean_auc = auc(mean_fpr, mean_tpr)
  plot(mean_fpr, mean_tpr, "k--",label="Mean ROC (area = $mean_auc)", lw=2)
  xlim([-0.05, 1.05])
  ylim([-0.05, 1.05])
  xlabel("False Positive Rate")
  ylabel("True Positive Rate")
  title("Ada Boost Receiver operating characteristic example")
  legend(loc="lower right")

end

function runOneOut(X::Array{Int64,2},y::Array{Int64,2})
  num_folds = 10
  num_instaces = size(X,1)
  print(num_instaces)
  loocv = LeaveOneOut(n=num_instaces)
  model = LogisticRegression()
  results = cross_validation.cross_val_score(model, X, y, cv=loocv,n_jobs=8)
  print(results)
end

function main()
  runLogisticRegression(X,y)
  runXGBoost(X,y)
  runLinearDiscriment(X,y)
  runXGBoost(X,y)
  test(X,y)
  runRandomForest(X,y)
  runCrossPredictionGraph(X,y)
  runModelGraph(X,y)
  runRoc(X,y)
  #runOneOut(X,y)

end

main()
