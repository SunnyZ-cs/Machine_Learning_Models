from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

"""
Logistic Regression Demonstration
Uses scikit-learn's built-in logistic regression for predicting Iris species.
Important note: x[0] is not set to be 1!
"""

def main():
  x, y = load_data()
  x_train, x_test, y_train, y_test = train_test_split(x,y)
  weights = train(x_train, y_train)
  print('weights:\n', weights, '\n')
  manual_test(weights, x_test.values, y_test.values)

def manual_test(weights, x_test, y_test):
  n_correct = 0
  print('test data:')
  for i in range(len(x_test)):
    x = x_test[i]
    actual = y_test[i]
    prediction = predict(weights, x)
    print(x, actual, prediction)
    if prediction == actual:
      n_correct += 1
  print(f'\nAccuracy: {n_correct / len(x_test)}')

def predict(weights, x):
  # warning: we have not prependend a 1 to x
  z = weights[0]
  for i in range(len(x)):
    z += weights[i+1] * x[i]
  p_hat = sigmoid(z)
  return 1 if p_hat >= 0.5 else 0

def sigmoid(x):
  return 1 / (1 + np.exp(-x))









def get_weights(model):
  """ concat the intercept and all the weights together"""
  weights = []
  weights.append(model.intercept_[0])
  for weight in model.coef_[0]:
    weights.append(weight)
  return weights

def train(x_train, y_train):
  model = LogisticRegression(max_iter=1000)
  model.fit(x_train, y_train)
  return get_weights(model)

def load_data():
  iris = datasets.load_iris()
  iris = pd.DataFrame(
      data= np.c_[iris['data'], iris['target']],
      columns= iris['feature_names'] + ['target']
      )
  # the iris dataset is actually 3 different types of irises
  # for this demo I only want to predict between two different types
  iris = iris[iris['target'] != 2]
  x = iris.drop(['target'], axis=1)
  y = iris['target']
  return x, y

if __name__ == '__main__':
  main()
