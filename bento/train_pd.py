import bentoml

from sklearn import svm
from sklearn import datasets
import pandas as pd

# load training data set
iris = datasets.load_iris()
X = pd.DataFrame(iris['data'], columns=['sepal_len', 'sepal_width', 'petal_len', 'petal_width'])
y = iris['target']

# train the model
clf = svm.SVC(gamma='scale')
clf.fit(X, y)

# save model to the BentoML local Model Store
saved_model = bentoml.sklearn.save_model("iris_clf", clf)
