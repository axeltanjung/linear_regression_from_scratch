import numpy as np
from ml_from_scratch.linear_model import Lasso

# REGRESSION CASE - 1
# -------------------
X = np.array([[0,0],[1,1],[2,2]])
y = np.array([1,2,3])

clf = Lasso(alpha=0.1)
clf.fit(X, y)
print(clf.coef_)
print(clf.intercept_)
