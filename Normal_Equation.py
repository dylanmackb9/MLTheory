import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataFrame = pd.read_csv('california_housing_train.csv')  # grabbing training values from colab sample data


median_age = dataFrame[['housing_median_age']].values*100  # Scaling
rooms = dataFrame[['total_rooms']].values
value = dataFrame[['median_house_value']].values/10000  # Scaling 
median_income = dataFrame[['median_income']].values
latitude = dataFrame[['latitude']].values
households = dataFrame[['households']].values

n = 2 
m = median_income.size  

x0 = np.full((m,1), 1)  # all ones x0 vector
feature_matrix = np.column_stack((x0, median_income))  # a m by n+1 matrix of features and their values for each training example
predict_vector = value  # a m dimensional vector of predict values


def sol_normal(feature_matrix, predict_vector):  # solving for parameters of n feature linear regression using normal equation 
	return (np.linalg.inv((np.transpose(feature_matrix)).dot(feature_matrix))).dot(np.transpose(feature_matrix)).dot(predict_vector)

print(sol_normal(feature_matrix, predict_vector))



