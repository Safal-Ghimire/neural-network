import numpy
import scipy.special




class neuralNetwork:
	def __init__(self,input_nodes,hidden_nodes1,hidden_nodes2,output_nodes,learning_rate):


		self.inodes=input_nodes
		self.h1nodes=hidden_nodes1
		self.h2nodes=hidden_nodes2
		self.onodes=output_nodes

		#definfing the 	weight matrix


		self.wih=numpy.random.normal(0.0,pow(self.inodes,-0.5),(self.h1nodes,self.inodes))
		self.whh=numpy.random.normal(0.0,pow(self.h1nodes,-0.5),(self.h2nodes,self.h1nodes))
		self.who=numpy.random.normal(0.0,pow(self.h2nodes,-0.5),(self.onodes,self.h2nodes))

		#deifining the learning rate
		self.lr=learning_rate


		#creatring the activation(sigmoid function)

		self.activation_function=lambda x:scipy.special.expit(x)


	def train(self,input_list,target) :
		#converting input_list to 2d array to transpose it

		inputs=numpy.array(input_list,ndmin=2).T

		#cinverting the target to 2d matri to transpose it
		targets=numpy.array(target,ndmin=2).T

		#ca;cu;ating the va;use to be passed at hidden layer 1
		h1_input=numpy.dot(self.wih,inputs)

		#calculating the outputs form hidden layer one
		h1_output=self.activation_function(h1_input)

		#calculating the valuse to be entered ant hidden layer 2

		h2_input=numpy.dot(self.whh,h1_output)

		#calculating the output from the final hidden layer
		h2_output=self.activation_function(h2_input)

		#calculating the inputs at the output layer

		out_input=numpy.dot(self.who,h2_output)

		#calculating the final output!!!
		final_output=self.activation_function(out_input)
		#calculating the errors at final node
		output_error=targets- final_output

		#error at final hidden node
		errorh2=numpy.dot(self.who.T,output_error)
		#error at the first hidden node
		errorh1=numpy.dot(self.whh.T,errorh2)


		self.who=self.who+self.lr*numpy.dot(output_error*final_output*(1-final_output),numpy.transpose(h2_output))
		self.whh=self.whh+self.lr*numpy.dot(errorh2*h2_output*(1- h2_output),numpy.transpose(h1_output))
		self.wih=self.wih+self.lr*numpy.dot(errorh1*h1_output*(1- h1_output),numpy.transpose(inputs))






		
		



	def query (self,input_list):

		#converting input_list to 2d array to transpose it

		inputs=numpy.array(input_list,ndmin=2).T

		#ca;cu;ating the va;use to be passed at hidden layer 1
		h1_input=numpy.dot(self.wih,inputs)

		#calculating the outputs form hidden layer one
		h1_output=self.activation_function(h1_input)

		#calculating the valuse to be entered ant hidden layer 2

		h2_input=numpy.dot(self.whh,h1_output)

		#calculating the output from the final hidden layer
		h2_output=self.activation_function(h2_input)

		#calculating the inputs at the output layer

		out_input=numpy.dot(self.who,h2_output)

		#calculating the final output!!!
		final_output=self.activation_function(out_input)

		return final_output


#definng the constraint they are self explanatory
input_nodes=784
hidden_nodes1=200
hidden_nodes2=100
output_nodes=10
learning_rate=0.09

epochs=5


#creating the instance of the neural network
n=neuralNetwork(input_nodes,hidden_nodes1,hidden_nodes2,output_nodes,learning_rate)


#loading the training data
training_data_file=open('mnist_test.csv','r')
training_data_all_list=training_data_file.readlines()
training_data_list=training_data_all_list[:200]#reading 1500 data not dataset individual data
	
training_data_file.close()

#training the neural network

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



