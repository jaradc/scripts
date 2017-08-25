from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
np.set_printoptions(suppress=True)

X,y,coefs = make_regression(n_samples=1000, n_features=10, n_informative=6,
                            noise=3.23, bias=27.43, coef=True)
bias = np.zeros((X.shape[0], 1))
X = np.hstack((bias, X))

w = np.zeros(X.shape[1])
costs = []

for epoch in range(10000):
  learning_rate = 0.1 / (0.2 + (0.1*epoch))
  pred = X.dot(w)
  error = pred - y
  gradient = X.T.dot(error) / X.shape[0]
  w = w - learning_rate*gradient
  mse = error.dot(error) / X.shape[0]
  costs.append(mse)
  #print('Cost: {:.5f}, Learning Rate: {:.5f}'.format(mse, learning_rate))

#R2 = 1 - (sse / sst)
sse = ((y - X.dot(w))**2).sum()
sst = ((y - y.mean())**2).sum()

r2 = 1 - (sse / sst)
print('R2 manual calc ', r2)
print('R2 double-check', r2_score(y, X.dot(w)))

plt.plot(costs)
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.title('Batch Gradient Descent')
plt.ylim(ymin=0)
plt.show()

lr = LinearRegression()
lr.fit(X, y)
for label,c in [('True Coefficients', coefs), ('Linear Regression Coef', lr.coef_[1:]),
                ('Gradient Descent Discovered Weights', w[1:])]:
  print('{}\n{}'.format(label, c.reshape((2,5))))

