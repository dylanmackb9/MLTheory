import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math 
import seaborn as sns


dataFrame = pd.read_csv("breast-cancer_train.csv")
irrad = dataFrame[['no.1']].values  # possibility of radiation treatment
menopause = dataFrame[['premeno']]  # what stage of menopause at time of diagnosis x={premeno,It40,Ge40} 
age = dataFrame[['30-39']].values  # age of patient
tsize = dataFrame[['30-34']].values  # in mm 
breast = dataFrame[['left']].values  # left or right breast, x={left,right}
degree_malig = dataFrame[['3']].values  # gives the degree of cell abnormality x={}
node_caps = dataFrame[['no']].values  # if cancer is contained to lymphnode or has penetrated into outside tissue x={no,yes}


# organizing/creating data
class1 = np.empty([irrad.size, 1])  # Class1 where 0 is no radiation and 1 is yes radiation 
for i in range(irrad.size):
	if irrad[i][0] == "no":
		ins = 0
	elif irrad[i][0] == "yes":
		ins = 1
	class1[i][0] = ins


feature_nodes = np.empty([node_caps.size, 1])
#Lymphnode containment, 0=yes 1=no
for i in range(node_caps.size):
	if node_caps[i][0] == 'yes':
		ins = 1
	elif node_caps[i][0] == 'no':
		ins = 0
	else: 
		ins = 0
	feature_nodes[i][0] = ins


feature_age = np.empty([age.size, 1])  
#Patient age, 10-19=0 20-29=1 30-39=2 40-49=3 50-59=4 60-69=5 70-79=6 80-89=7 90-99=8
for i in range(age.size):
  	if age[i][0] == '10-19':
  		ins = 1
  	elif age[i][0] == '20-29':
  		ins = 2
  	elif age[i][0] == '30-39':
  		ins = 3
  	elif age[i][0] == '40-49':
  		ins = 4
  	elif age[i][0] == '50-59':      
  		ins = 5
  	elif age[i][0] == '60-69':
  		ins = 6
  	elif age[i][0] == '70-79':
  		ins = 7
  	elif age[i][0] == '80-89':
  		ins = 8
  	elif age[i][0] == '90-99':
  		ins = 9
  	feature_age[i][0] = ins

feature_degmal = degree_malig
#degree of malignancy, level 1,2,3 malignancy = 1,2,3

feature_size = np.empty([tsize.size, 1])  # tumor size in mm
#0-4=1,5-9=2,10-14=3,15-19=4,20-24=5,25-29=6,30-34=7,35-39=8,40-44=9,45-49=10,50-54=11,55-59=12
for i in range(tsize.size):
	if tsize[i][0] == '0-4':
	    ins = 1
	elif tsize[i][0] == '5-9':
	    ins = 2
	elif tsize[i][0] == '10-14':
	    ins = 3
	elif tsize[i][0] == '15-19':
	    ins = 4
	elif tsize[i][0] == '20-24':
	    ins = 5
	elif tsize[i][0] == '25-29':
	    ins = 6
	elif tsize[i][0] == '30-34':
	    ins = 7
	elif tsize[i][0] == '35-39':
	    ins = 8
	elif tsize[i][0] == '40-44':
	    ins = 9
	elif tsize[i][0] == '45-49':
	    ins = 10
	elif tsize[i][0] == '50-54':
	    ins = 11
	elif tsize[i][0] == '55-59':
	    ins = 12
	feature_size[i][0] = ins


fdata = np.column_stack((feature_degmal, feature_nodes, class1))
g = sns.scatterplot(x=fdata[:,0], y=fdata[:,1], hue=fdata[:,2])
plt.show()



learning_rate = .01
m = class1.size
n = 2
parameters = np.full((n+1,1), 0)
features = np.insert(np.column_stack((feature_age, feature_size)),0,1,axis=1) 



def vectorized_linear(parameters, features):
	"""
	Calculates the linear equation h(x) = a0x0 + a1x1 + a2x2 for the entire feature vector. 
	Input: parameters vector, features matrix. Output: vector
	"""
	return features.dot(parameters)

def sigmoid(x):  # takes transpose theta * x, returns input value as function of sigmoid (logistic) 
	"""
	Calculates linear function input into the sigmoid function. Input: m by n+1 matrix parameters*features. Output: m by n+1
	"""
	temp0 = (math.e) ** -x
	return 1 / (1 + (temp0))


def logistic_loss(parameters, features):  
	"""
	Calculates log loss function. Input: parameters vector, and features matrix. Output: scalar loss at given parameters
	"""
	return (1/m) * ((np.transpose(-class1)).dot(np.log(sigmoid(vectorized_linear(parameters, features)))) - np.transpose(1-class1).dot(np.log(1-sigmoid(vectorized_linear(parameters, features)))))



def gradient_descent(parameters, features, learning_rate):
	"""
	"""
	x_vals = []
	y_vals = []
	iter = 0

	for i in range(220):
		iter = iter + 1
		parameters = parameters - (learning_rate/m)*(np.transpose(features)).dot(vectorized_linear(parameters, features) - class1)

		if iter%10 == 0: 
			print("Current loss: " + str(logistic_loss(parameters, features)))
			x_vals.append(iter)
			y_vals.append(logistic_loss(parameters, features)[0][0])

	plt.plot(x_vals, y_vals)
	#plt.show()
	print(parameters)
	return parameters 


#params = gradient_descent(parameters, features, learning_rate)

def predict(parameters, features, f_age, f_size): 
	"""
	Input: Optimized parameters, feature matrix, age from 1-9, size from 1-12, both as matrix scalars
	"""
	feature_vector = np.insert(np.column_stack((f_age,f_size)),0,1,axis=1)
	h = feature_vector.dot(parameters)
	probability = sigmoid(h)
	if probability >= 0.5:
		pred = 1
	elif probability < 0.5:
		pred = 0
	print("Predicted: " + str(probability) + " to " + str(pred))

#predict(params, features, 1, 9)



