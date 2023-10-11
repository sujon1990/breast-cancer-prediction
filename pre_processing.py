import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
dataset = pd.read_csv('breastCancer.csv')

#for col in dataset.columns:
#    print(col)
#    print(dataset[col].unique())


# Data Pre-processing
dataset = dataset.replace('?',np.nan)
dataset = dataset.fillna(dataset.median())
dataset['bare_nucleoli'] = dataset['bare_nucleoli'].astype('int64')
dataset.drop('id',axis=1,inplace=True)

X = dataset.drop('class',axis=1)
y = dataset['class']

sc_x = StandardScaler()
X = sc_x.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=1)

pca = PCA()
pca.fit(X_train)
pca_variance_ratio = pca.explained_variance_ratio_
pca_components = np.arange(len(pca_variance_ratio)) + 1

cumsum_pca_variance_ratio = np.cumsum((pca_variance_ratio))
plt.bar(pca_components, cumsum_pca_variance_ratio)
plt.plot(pca_components, cumsum_pca_variance_ratio, "ro-")
plt.xticks(pca_components, ["Comp." + str(i) for i in pca_components], rotation=90)
plt.title("Cumulative Variance")
plt.ylabel("Variance")
plt.show()


pca = PCA(n_components=5)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

