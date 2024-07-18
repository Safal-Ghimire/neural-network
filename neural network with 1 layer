import numpy
import scipy.special
#we are extracting the sigmoid from scipy.special

import matplotlib.pyplot



class neuralnetwork:
	def __init__(self,inputnodes,outputnodes,hiddennodes,learningrate):

		#defining the nodes
		self.inodes=inputnodes
		self.hnodes=hiddennodes
		self.onodes=outputnodes

		#defining the learning rate
		self.lr=learningrate

		#defining the weight for the nodes
	 #the below funstion creates weights random from -1 to 1 of the dimensions assignedcenterd around 0
		#as i have done -0.5 the values are from -0.5 to0.5

		self.wih=numpy.random.normal(0.0,pow(self.inodes,-0.5),(self.hnodes,self.inodes))#?why not self.inodes,hnodes okay now i understand it
		self.who=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.onodes,self.hnodes))


		#activation funstion
		self.activation_function=lambda x:scipy.special.expit(x)


	def train (self,inputs_list, target_list):
		#converting the input lists to 2d the reason is at the query function
		inputs=numpy.array(inputs_list,ndmin=2).T

		#doing the same ti targets
		targets=numpy.array(target_list,ndmin=2).T

		#calculating the signal into the hidden layer
		hidden_inputs=numpy.dot(self.wih,inputs)

		#calculating the emerging signals with the help of activation_function
		hidden_outputs=self.activation_function(hidden_inputs)

		#calculating the signals into the output layer
		final_inputs=numpy.dot(self.who,hidden_outputs)

		#calculating the final signals emerging in the output with the activation_function

		final_outputs=self.activation_function(final_inputs)

		#error of output layer is target-actual
		output_errors=targets- final_outputs

		#hidden layer error is the error split by weight
		hidden_errors=numpy.dot(self.who.T,output_errors)
		#updating the weights for the link weights between hidden and output layers
		self.who+=self.lr*numpy.dot((output_errors*final_outputs*(1.0- final_outputs)),numpy.transpose(hidden_outputs))

		#updating the weights for links between input and the hdden layers
		self.wih+=self.lr*numpy.dot((hidden_errors*hidden_outputs*(1.0- hidden_outputs)),numpy.transpose(inputs))


		pass

	def query(self,inputs_list):
		#converting the input list to 2 d arry
		inputs=numpy.array(inputs_list, ndmin=2).T #we are doing converting the 1 d array to 2d
		#to transpose it only
		#because 1d array cant be transposed and also remember our 2d matrix has the dimensionof n,1


		#perform necessary multiplications in order to pluck those values in activation_function
		hidden_inputs=numpy.dot(self.wih,inputs)

		#calculating the activation values
		hidden_outputs=self.activation_function(hidden_inputs)

		#again doing the multiplication for the output layer
		final_inputs=numpy.dot(self.who,hidden_outputs)

		#the signal emerging from the final output layer
		final_outputs=self.activation_function(final_inputs)

		return final_outputs

		pass

#loading the training data
training_data_file=open('mnist_test.csv','r')
training_data_all_list=training_data_file.readlines()
training_data_list=training_data_all_list[:200]#reading 1500 data not dataset individual data
	
training_data_file.close()

#setting teh necessary variables

input_nodes=784
hidden_nodes=100
output_nodes=10
learning_rate=0.09





n=neuralnetwork(input_nodes,output_nodes,input_nodes,learning_rate)

#train the neural network
epochs=5
for e in range (epochs):

	for record in training_data_list:
		#spliting the records by ','

		all_values=record.split(',')

		# scaling and shifting the inputes because we want to have all the values between 0.1 and 0.99

		inputs=(numpy.asfarray(all_values[1:])/255.0 *0.99)+0.01
		#create the target values all 0.01 except desired value 0.99 

		targets=numpy.zeros(output_nodes)+0.01

		# all values[0] is the target value for this record
		targets[int(all_values[0])]=0.99
		n.train(inputs,targets)
		pass
	pass
#loading the test dataset
test_data_file=open("mnist_train_100.csv","r")
test_data_list=test_data_file.readlines()
test_data_file.close()


#test the neural network

scorecard= []

for record in test_data_list:
	#spliting the record by";"
	all_values=record.split(',')

	#corect answer is the first value
	correct_label=int(all_values[0])

	#scale and shift the inputs

	inputs=(numpy.asfarray(all_values[1:])/255.0 * 0.99)+0.01


	#querying the neral network

	outputs=n.query(inputs)

	#the index of the highest value corresponds to the label
	label=numpy.argmax(outputs)

	if (label==correct_label):
		scorecard.append(1)

	else:
		scorecard.append(0)

	



#calculate the performence

scorecard_array=numpy.asarray(scorecard)

print("performance=",scorecard_array.sum()/scorecard_array.size)

