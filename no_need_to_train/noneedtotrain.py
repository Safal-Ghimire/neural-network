import numpy
import scipy.special
import imageio.v2 as imageio    #this is to convert pixels to array
#we are extracting the sigmoid from scipy.special

# Function to read matrix from file
def read_matrix_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()  # Read all lines from the file
    
    # Initialize an empty list to store rows of the matrix
    matrix = []
    
    for line in lines:
        # Strip newline character and split by comma to get elements
        row = list(map(float, line.strip().split(',')))
        matrix.append(row)
    
    # Convert the list of lists into a NumPy array
    matrix = numpy.array(matrix)
    return matrix

# File name to read from
filename_wih = 'wih.txt' 
filename_who = 'who.txt'

# Read matrix from file
wih = read_matrix_from_file(filename_wih)
who = read_matrix_from_file(filename_who)

class neuralnetwork:
	def __init__(self,wih,who,learningrate):

		#defining the learning rate
		self.lr=learningrate
		self.wih=wih		
		self.who=who
		#activation funstion
		self.activation_function=lambda x:scipy.special.expit(x)

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


learning_rate=0.2

n=neuralnetwork(wih,who,learning_rate)

#reading the image iand storing it in an array

img_array=imageio.imread('unno.jpg')
img_data=255.0- img_array.reshape(784)# we subtract this becaude our train dataset has 0 to white and 255 to black

img_data=(img_data/255.0 *0.99)+0.01
#querying the neral network
outputs=n.query(img_data)
#the index of the highest value corresponds to the label
label=numpy.argmax(outputs)
print(label)
print("completed")
