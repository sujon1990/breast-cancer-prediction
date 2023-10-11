import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

dataset = pd.read_csv('breastCancer.csv')

# Data Pre-processing
dataset = dataset.replace('?',np.nan)
dataset = dataset.fillna(dataset.median())
dataset['bare_nucleoli'] = dataset['bare_nucleoli'].astype('int64')
dataset.drop('id',axis=1,inplace=True)

X = dataset.drop('class',axis=1)
y = dataset['class']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=1)

best_model = RandomForestClassifier(
    n_estimators = 10,
    criterion = 'gini',
    max_depth = 10,
    min_samples_leaf = 4,
    min_samples_split = 2,
    bootstrap = True,
    max_features = 'sqrt'
)

pipeline = Pipeline([('scalar', StandardScaler()),
                     ('pca', PCA(n_components=5)),
                     ('classifier', best_model)])

pipeline.fit(X_train, y_train)
pickle.dump(pipeline, open('model/best_model', 'wb'))
y_pred = pipeline.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)
print(np.mean(cross_val_score(pipeline, X_train, y_train, cv=10)))