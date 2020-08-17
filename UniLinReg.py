import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



dataFrame = pd.read_csv('california_housing_train.csv')  # grabbing training values from colab sample data
median_age = (dataFrame[['housing_median_age']].values - 28.589)/52  # Scaling
rooms = (dataFrame[['total_rooms']].values - 2643.664411764706)/37937.0
value = (((dataFrame[['median_house_value']].values) - 207300.9)/500001)  # Scaling 
#value = (((dataFrame[['median_house_value']].values)/30000))  # Scaling
median_income = (dataFrame[['median_income']].values - 3.8835781)/15
#median_income = (dataFrame[['median_income']].values)/5
latitude = dataFrame[['latitude']].values
latitude = dataFrame[['latitude']].values
households = dataFrame[['households']].values

#plt.scatter(median_age,value,s=2)  # graphing 1-feature simple scatter
#plt.show()

quad = median_income ** 2

learning_rate = .1
n = 1  # number of features
m = median_income.size  # number of training examples

output = value  # output matrix, a m by 1 vector 
parameters = np.full((n+3,1), 0)  # starting parameter 1 by n+1 row vector of all 0s 
features = np.insert(np.column_stack((median_income, median_income**2, median_income**3)),0,1,axis=1)  #  feature matrix, a m by n+1 matrix where the first column is all 1 (x0=1)


def linear(parameters, feature_vector):  # solving a n+1 term linear equation 
  """
  Calculates the linear equation h(x) = a0x0 + a1x1 + a2x2 + .... . Input: parameters vector, i'th row of the feature vector 
  output: scalar
  """
  return np.transpose(parameters).dot(feature_vector)

def linear_cost(parameters, features, i):
  """
  Calculates the least squares cost for the i'th training example. Input: parameters vector, features vector, i'th training example
  Output: scalar cost
  """
  return (0.5) * (linear(parameters, features[i]) - output[i]) ** 2

def linear_loss(parameters, features):
  """
  Calculates loss function of least squares cost function. Input: parameters vector, features matrix. Output: scalar loss
  """
  loss = 0
  for i in range(m):
    loss = loss + linear_cost(parameters, features, i)
  return (1/m)*loss

def partial_loss(parameters, features, parameter_number):  
  """
  Calculates partial derivative function of least squares loss function to be used in gradient descent algorithm. 
  Input: parameters vector, features matrix, number of parameter the derivative is with respect to {0:n}. Output: scalar loss 
  """
  loss = 0
  for i in range(m):  # starting the sum at beginning (0) for you
    loss = loss + (linear(parameters, features[i]) - output[i])*features[i][parameter_number]
  return (1/m)*loss




def vectorized_linear(parameters, features):
  """
  Calculates the linear equation h(x) = a0x0 + a1x1 + a2x2 for the entire feature vector. 
  Input: parameters vector, features matrix. Output: vector
  """
  return (features).dot(parameters)

def vectorized_linear_cost(parameters, feautres):
  """
  Calculates the least squares cost for the whole features matrix. Input: parameters vector, features matrix
  Output: vector of costs
  """
  return (0.5) * (vectorized_linear(parameters, features) - output) 

def vectorized_linear_loss(parameters, features):
  """
  """
  return (np.full((1,m),1).dot(vectorized_linear_cost(parameters, features))/m)


def gradient_descent(parameters, features, learning_rate):
  """
  Runs the gradient descent algorithm to minimize the linear loss function with respect n parameters. 
  Input: parameter vector, feature matrix, learning rate. Output: optimized parameters for minimized loss function
  """
  x_vals = []
  y_vals = []
  iter = 0
  converge = False

  for i in range(5000):  # calling convergence at .0001 difference 
    iter = iter + 1
    parameters = parameters-(learning_rate/m)*(np.transpose(features)).dot(vectorized_linear(parameters, features) - output)  # p = p - x^T (h(x)-y)
    #if (vectorized_linear_loss(parameters,features) - vectorized_linear_loss(parameters,features)) < 0.00001:
    #  converge = True

    if iter%10==0:  # tracking progress
      print("Current loss: " + str(vectorized_linear_loss(parameters, features)))
      x_vals.append(iter)
      y_vals.append(vectorized_linear_loss(parameters, features)[0][0])
      
  print(parameters)
  plt.plot(x_vals,y_vals)
  plt.show()
  return parameters

linreg_param = gradient_descent(parameters, features, learning_rate)

plt.figure()
plt.scatter(median_income,value,s=2)
x = np.linspace(-0.5,0.9,2000)
y = linreg_param[0] + linreg_param[1]*x + linreg_param[2]*x**2 + linreg_param[3]*x**3
plt.plot(x,y, c='red')
plt.show()








