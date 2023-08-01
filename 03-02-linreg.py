import numpy as np
from ml_from_scratch.linear_model import LinearRegression

X = np.array([[1,1],
              [1,2],
              [2,2],
              [2,3]])
y = np.dot(X, np.array([1,2])) + 3

reg = LinearRegression()
reg.fit(X,y)

print(reg.coef_)
print(reg.intercept_)
print(reg.predict([np.array([[3,5]])]))