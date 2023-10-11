import time
import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from pre_processing import X_train, y_train, X_test, y_test

def hyper_parameter_tuning(estimator_name, estimator, parameters):
    start = time.time()
    grid = GridSearchCV(estimator, parameters)
    grid.fit(X_train, y_train)
    print('***************************** Best Parameters ****************************')
    print(grid.best_params_)
    print('***************************** Best Estimator *****************************')
    print(grid.best_estimator_)
    params = pd.DataFrame(grid.cv_results_['params'])
    score = pd.DataFrame(grid.cv_results_['mean_test_score'], columns=['score'])
    print('************************** Score and parameters ***************************')
    params_and_scores = pd.concat([params, score], axis=1)
    params_and_scores.to_csv(f'reports/{estimator_name}.csv', index=False)
    print(params_and_scores)
    end = time.time()
    print(f'Time taken = {end-start}')
    return grid

def model_creation(model, X_train, X_test, y_train, y_test, cv=10):
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    report = classification_report(y_test, predicted)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>> Classificatio Report >>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(report)
    print('========================== Cross validation score ===========================')
    print(np.mean(cross_val_score(model, X_train, y_train, cv=cv)))

print("Support vector machine")
svc_parameters = {'kernel': ['linear', 'poly', 'sigmoid', 'rbf']}
svc = SVC()
svc_hyper_parameter_tuning = hyper_parameter_tuning("Support vector machine",svc, svc_parameters)
svc = svc_hyper_parameter_tuning.best_estimator_
model_creation(svc, X_train, X_test, y_train, y_test)

print("K-Nearest Neighbor Classification")
knn_parameters = {'n_neighbors': [5, 10, 20, 40, 80],
                  'metric':['minkowski', 'euclidean', 'manhattan']}
knn = KNeighborsClassifier()
knn_hyper_parameter_tuning = hyper_parameter_tuning("K-Nearest Neighbor Classification",knn, knn_parameters)
knn = knn_hyper_parameter_tuning.best_estimator_
model_creation(knn, X_train, X_test, y_train, y_test)

print("Logistic Regression")
logistic_parameters = {'penalty':['l2', 'none']}
logistic = LogisticRegression()
logistic_hyper_parameter_tuning = hyper_parameter_tuning("Logistic Regression",logistic, logistic_parameters)
logistic = logistic_hyper_parameter_tuning.best_estimator_
model_creation(logistic, X_train, X_test, y_train, y_test)

print("Decision Tree")
decision_tree_parameters = {'criterion':['gini', 'entropy'],
                            'max_depth':[None, 5, 10, 20, 40],
                            'min_samples_split':[i for i in range(2, 5)],
                            'min_samples_leaf':[i for i in range(1, 5)],
                            'max_features':['auto', 'sqrt', 'log2']
                            }
decision_tree = DecisionTreeClassifier()
decision_tree_hyper_parameter_tuning = hyper_parameter_tuning("Decision Tree",decision_tree, decision_tree_parameters)
decision_tree = decision_tree_hyper_parameter_tuning.best_estimator_
model_creation(decision_tree, X_train, X_test, y_train, y_test)

print("Random Forest")
random_forest_parameters = {'criterion':['gini', 'entropy'],
                            'n_estimators':[10, 50, 100, 200, 400, 600],
                            'max_depth':[None, 5, 10, 20, 40],
                            'min_samples_split':[i for i in range(2, 5)],
                            'min_samples_leaf':[i for i in range(1, 5)],
                            'max_features':['auto', 'sqrt', 'log2'],
                            'bootstrap': [True, False]
                            }
random_forest = RandomForestClassifier()
random_forest_hyper_parameter_tuning = hyper_parameter_tuning("Random Forest",random_forest, random_forest_parameters)
random_forest = random_forest_hyper_parameter_tuning.best_estimator_
model_creation(random_forest, X_train, X_test, y_train, y_test)

print("Extra Trees")
extra_tree_parameters = {'criterion':['gini', 'entropy'],
                            'n_estimators':[10, 50, 100, 200, 400, 600],
                            'max_depth':[None, 5, 10, 20, 40],
                            'min_samples_split':[i for i in range(2, 5)],
                            'min_samples_leaf':[i for i in range(1, 5)],
                            'max_features':['auto', 'sqrt', 'log2'],
                            'bootstrap': [True, False]
                            }
extra_tree = ExtraTreesClassifier()
extra_tree_hyper_parameter_tuning = hyper_parameter_tuning("Extra Trees",extra_tree, extra_tree_parameters)
extra_tree = extra_tree_hyper_parameter_tuning.best_estimator_
model_creation(extra_tree, X_train, X_test, y_train, y_test)
