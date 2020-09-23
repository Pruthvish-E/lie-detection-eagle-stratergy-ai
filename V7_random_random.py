# Author : Ameya Bhamare
import random
from sklearn.metrics import confusion_matrix
from datetime import datetime
startTime = datetime.now()

l = []
def generateColumns(start, end):
    for i in range(start, end+1):
        l.extend([str(i)+'X', str(i)+'Y'])
    return l

eyes = generateColumns(1, 12)

# reading in the csv as a dataframe
import pandas as pd
df = pd.read_csv('Eyes.csv')

# selecting the features and target
X = df[eyes]
y = df['truth_value']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 42)

# Data Normalization
from sklearn.preprocessing import StandardScaler as SC
sc = SC()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

import numpy as np
X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)




# importing the required layers from keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation

# layering up the cnn
model = Sequential()
model.add(Dense(4, input_dim = X_train.shape[1])) 
model.add(Activation('relu'))
model.add(Dense(4))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# model compilation
opt = 'adam'
model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])


# training
#model.fit(X_train, y_train, batch_size = 2, epochs = 50, validation_data = (X_test, y_test), verbose = 2)

'''
# serialize model to JSON
model_json = model.to_json()
with open("modelEyes.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("modelEyes.h5")
print("Saved model to disk")   
'''
#np.random.seed(100)
def exploration(model,X_train,y_train,k):
	max_accuracy = 0
	for i in range(k):
		print("Iteration:------------",i)
		present_weights = model.get_weights()
		new_weights = []
		for i in present_weights:
			if(len(i.shape)==2):

				new_weights.append(np.random.rand(i.shape[0],i.shape[1]))
			else:
				new_weights.append(np.random.rand(i.shape[0]))

		model.set_weights(new_weights)




		# using the learned weights to predict the target
		y_pred = model.predict(X_train)
		#model.set_weights([i*0 for i in model.get_weights()])
		# setting a confidence threshhold of 0.9
		y_pred_labels = list(y_pred > 0.9)

		for i in range(len(y_pred_labels)):
		    if int(y_pred_labels[i]) == 1 : y_pred_labels[i] = 1
		    else : y_pred_labels[i] = 0

		# plotting a confusion matrix
		from sklearn.metrics import confusion_matrix
		cm = confusion_matrix(y_train, y_pred_labels)
		print("\n")
		print("Confusion Matrix : ")
		print(cm)  
		accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
		if(accuracy>max_accuracy):
			weights = model.get_weights()
			max_accuracy = accuracy
		#	print(max_accuracy)
		print("Accuracy:- ",(cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))
		print("\n")

	print("max train accuracy:- ",max_accuracy)
	return max_accuracy



def getAccuracy(model,X_train,y_train,present_weights):
	model.set_weights(present_weights)
	y_pred = model.predict(X_train)
	y_pred_labels = list(y_pred > 0.9)
	for i in range(len(y_pred_labels)):
		    if int(y_pred_labels[i]) == 1 : y_pred_labels[i] = 1
		    else : y_pred_labels[i] = 0

	# 	# plotting a confusion matrix
	# from sklearn.metrics import confusion_matrix
	cm = confusion_matrix(y_train, y_pred_labels)
		#print("\n")
		#print("Confusion Matrix : ")
		#print(cm)  
	accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
	# right = 0
	# for i in range(len(y_train)):
	# 	if(y_train[i] == y_pred[i]):
	# 		right+=1
	# accuracy = right/len(y_train)
	return accuracy




def cross_local_maxima(model,X_train,y_train,accuracy,update):
	present_weights = model.get_weights()
	weights = present_weights
	for i in range(len(present_weights)):
			for j in range(len(present_weights[i])):
				if(random.randint(0,10)>8):
					present_weights[i][j]*=random.uniform(-1*update,update)
					present_weights[i][j]%=1e+10

	#model.set_weights(present_weights)
	acc = getAccuracy(model,X_train,y_train,present_weights)
	print("before:- ",accuracy)
	print("after :- ",acc)
	if(acc> accuracy ):
		return present_weights

	return weights






def explore(model,X_train,y_train,k):
	max_accuracy = 0
	best_weights = []
	for i in range(k):
		#print("Iteration:------------",i)
		present_weights = model.get_weights()
		new_weights = []
		np.random.seed(random.randint(0,1000))
		for i in present_weights:
			if(len(i.shape)==2):

				new_weights.append(np.random.rand(i.shape[0],i.shape[1]))
			else:
				new_weights.append(np.random.rand(i.shape[0]))

		model.set_weights(new_weights)




		# using the learned weights to predict the target
		y_pred = model.predict(X_train)
		#model.set_weights([i*0 for i in model.get_weights()])
		# setting a confidence threshhold of 0.9
		y_pred_labels = list(y_pred > 0.9)

		for i in range(len(y_pred_labels)):
		    if int(y_pred_labels[i]) == 1 : y_pred_labels[i] = 1
		    else : y_pred_labels[i] = 0

		# plotting a confusion matrix
		cm = confusion_matrix(y_train, y_pred_labels)
		#print("\n")
		#print("Confusion Matrix : ")
		#print(cm)  
		accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
		if(accuracy>max_accuracy):
			best_weights = model.get_weights()
			max_accuracy = accuracy


		#best_weights = cross_local_maxima(model,X_train,y_train,max_accuracy,10000)
		#max_accuracy = getAccuracy(model,X_train,y_train,best_weights)
		print("##########In explore:- ######",max_accuracy)
		#print("Accuracy:- ",(cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))
		#print("\n")
	model.set_weights(best_weights)
	#print("max train accuracy:- ",max_accuracy)
	return (max_accuracy,model)



def exploitation(model,X_train,y_train,k,rate):
	
	accuracy,model = explore(model,X_train,y_train,k)
	print("prior:- ",accuracy)
	present_weights = model.get_weights()
	acc = accuracy
	best_weights = present_weights
	Iterations = 10
	while(accuracy<0.79):
		for i in range(len(present_weights)):
			for j in range(len(present_weights[i])):
				if(random.randint(0,10)>8):
					present_weights[i][j]*=random.uniform(-10,10)
		acc = getAccuracy(model,X_train,y_train,present_weights)
		if(acc<accuracy):
			present_weights = best_weights
		else:
			best_weights = present_weights
			accuracy = acc

		if(abs(accuracy-acc)<0.01 and accuracy!=acc):
			acc,model = explore(model,X_train,y_train,k)
			if(acc>accuracy):
				best_weights = model.get_weights()
				accuracy = acc

		#Iterations-=1

	print("Acc: - -- - - - ",accuracy)
	#model.set_weights(best_weights)
	print("train  -- - - - - - - - :- ",getAccuracy(model,X_train,y_train,best_weights))		
	return model


model = exploitation(model,X_train,y_train,2,0.001)


print("train :- ",getAccuracy(model,X_train,y_train,model.get_weights()))	






y_pred = model.predict(X_test)
#model.set_weights([i*0 for i in model.get_weights()])
# setting a confidence threshhold of 0.9
y_pred_labels = list(y_pred > 0.9)

for i in range(len(y_pred_labels)):
    if int(y_pred_labels[i]) == 1 : y_pred_labels[i] = 1
    else : y_pred_labels[i] = 0

# plotting a confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_labels)
print("\n")
print("Confusion Matrix : ")
print(cm)  
accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
# if(accuracy>max_accuracy):
# 	weights = model.get_weights()
# 	max_accuracy = accuracy
#	print(max_accuracy)
print(" Test Accuracy:- ",(cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))
print("\n")


# from sklearn.metrics import accuracy_score

# # Predict labels for train set and assess accuracy
# y_train_pred = model.predict(X_train)

# y_train_accuracy = accuracy_score(y_train, y_train_pred)

# print("train accuracy:- ",y_train_accuracy)



# creating a dataframe to show results
#df_results = pd.DataFrame()
#df_results['Actual label'] = y_test
#df_results['Predicted value'] = y_pred
#df_results['Predicted label'] = y_pred_labels
#df_results.to_csv(r'C:\Users\91974\Desktop\EYE\Results.csv')

# printing execution time of script
print("\n")
print("Execution time in seconds = ", datetime.now() - startTime)
