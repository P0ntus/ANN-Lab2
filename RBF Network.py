import math
import random
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan) #Always print the whole matrix

x_lower_interval = 0
x_upper_interval = 2*math.pi
y_lower_interval = -2
y_upper_interval = 2
step_length = 0.1

learning_rate = 0.01

random.seed(a=None)

# Class that represents a 2d input and what type it should be classified as
class Node:
    def __init__(self, x, y, v):
        self.x = x
        self.y = y
        self.variance = v
    def __repr__(self):
        return "<x:%s y:%s v:%s>" % (self.x, self.y, self.variance)
    def __str__(self):
        return "member of Test"

# Mean squared error
def mean_squared_error(expected, predicted):
	return np.sum((expected - predicted) ** 2)/len(expected)

# Squared error
def squared_error(expected, predicted):
	return np.sum((expected - predicted) ** 2)
	
# Absolute residual error
def absolute_residual_error(expected, predicted):
	return np.sum(abs(expected - predicted))/len(expected)
	
# We use Gaussian RBF's with the following transfer function
def transfer_function(x, position, variance):
	return (math.exp((-(x - position)**2) / (2*(variance**2))))

# We use Gaussian RBF's with the following transfer function
def transfer_function_2d(x, y, position_x, position_y, variance):
	return (math.exp((-(euclidean_distance(x, y, position_x, position_y))**2) / (2*(variance**2))))

def euclidean_distance(x1, y1, x2, y2):
	return (math.sqrt(((x1 - x2)**2) + ((y1 - y2)**2)))
	
def sin_function(x):
	return math.sin(2*x)
	
def square_function(x):
	if (math.sin(2*x) > 0): 
		return 1
	return -1
	
# Sets values in input_pattern that are >= 0 to 1 and values < 0 to -1
def binary(input_pattern):
	for v in range (0, len(input_pattern)):
		if (input_pattern[v] >= 0):
			input_pattern[v] = 1
		else:
			input_pattern[v] = -1
	return input_pattern
	
# Adds noise to input_pattern and then returns it as output_pattern
def noise(input_pattern):
	output_pattern = []
	for i in range(0, len(input_pattern)):
		output_pattern.append(input_pattern[i] + np.random.normal(0, 0.1, 1)[0])
	return output_pattern

# Adds shuffles pattern A and B
def shuffle(A, B):
	temp_A = np.copy(A)
	temp_B = np.copy(B)
	C = np.asarray(np.arange(0, len(A), 1))
	random.shuffle(C)
	for i in range(0, len(C)):
		A[i] = temp_A[C[i]]
		B[i] = temp_B[C[i]]

# Adds shuffles pattern A and B
def shuffle_3(A, B, C):
	temp_A = np.copy(A)
	temp_B = np.copy(B)
	temp_C = np.copy(B)
	D = np.asarray(np.arange(0, len(A), 1))
	random.shuffle(D)
	for i in range(0, len(D)):
		A[i] = temp_A[D[i]]
		B[i] = temp_B[D[i]]
		C[i] = temp_C[D[i]]

for nodes in range(33, 34):
	# Generate function data

	# SIN DATA----------------------------------------------------------------------------------------------------------
	# Training patterns - Generate values between 0 and 2pi with step length 0.1 using our sin_function
	sin_training_input_pattern = np.asarray(np.arange(x_lower_interval, x_upper_interval, step_length))
	#sin_training_input_pattern = noise(sin_training_input_pattern)
	sin_training_output_pattern = list(map(sin_function, sin_training_input_pattern))

	# Testing patterns - Generate values between 0.05 and 2pi with step length 0.1 using our sin_function
	sin_test_input_pattern = np.asarray(np.arange(x_lower_interval + (step_length/2), x_upper_interval, step_length))
	#sin_test_input_pattern = noise(sin_test_input_pattern)
	sin_test_output_pattern = list(map(sin_function, sin_test_input_pattern))

	# SIN DATA----------------------------------------------------------------------------------------------------------

	# SQUARE DATA-------------------------------------------------------------------------------------------------------
	# Training patterns - Generate values between 0 and 2pi with step length 0.1 using our square_function
	#square_training_input_pattern = np.asarray(np.arange(x_lower_interval, x_upper_interval, step_length))
	#square_training_output_pattern = list(map(square_function, square_training_input_pattern))

	# Testing patterns - Generate values between 0.05 and 2pi with step length 0.1 using our square_function
	#square_test_input_pattern = np.asarray(np.arange(x_lower_interval + (step_length/2), x_upper_interval, step_length))
	#square_test_output_pattern = list(map(square_function, square_test_input_pattern))
	# SQUARE DATA-------------------------------------------------------------------------------------------------------

	errors = []
	RANDOM_errors = []
#for nodes in range(0, 101):
	# Initiate RBF nodes and WEIGHTS
	NUM_NODES_ROW = nodes # Using len(sin_training_output_pattern) or len(square_training_output_pattern) gives good results
	NUM_NODES_COL = 3
	variance = 0.5
	mu, sigma = 0, 0.1 # used for weight initialization
	RBF_Nodes = []
	weight = []
	RANDOM_RBF_Nodes = []
	for c in range(0, NUM_NODES_COL):
		for r in range(0, NUM_NODES_ROW):
			x = (x_lower_interval + ((x_upper_interval - x_lower_interval)/NUM_NODES_ROW/2)) + r * ((x_upper_interval - x_lower_interval)/NUM_NODES_ROW)
			y = (y_lower_interval/2 + ((y_upper_interval - y_lower_interval)/NUM_NODES_COL/2)) + c * ((y_upper_interval - y_lower_interval)/NUM_NODES_COL)
			RANDOM_x = float(random.randint(x_lower_interval*1000, math.ceil(x_upper_interval)*1000))/1000
			RANDOM_y = float(random.randint(y_lower_interval*1000, y_upper_interval*1000))/1000
			#print(float(random.randint(x_lower_interval*1000, math.ceil(x_upper_interval)*1000))
			#print(x, y)
			weight.append(np.random.normal(mu, sigma, 1)[0])
			RBF_Nodes.append(Node(x, y, variance))
			RANDOM_RBF_Nodes.append(Node(RANDOM_x, RANDOM_y, variance))

	# Calculate SIN phi
	sin_train_phi = np.zeros((len(sin_training_input_pattern), len(RBF_Nodes)))
	sin_test_phi = np.zeros((len(sin_training_input_pattern), len(RBF_Nodes)))
	RANDOM_sin_train_phi = np.zeros((len(sin_training_input_pattern), len(RBF_Nodes)))
	RANDOM_sin_test_phi = np.zeros((len(sin_training_input_pattern), len(RBF_Nodes)))
	for p in range (0, len(sin_training_input_pattern)):
		for n in range (0, len(RBF_Nodes)):
			#sin_train_phi[p][n] = transfer_function(sin_training_input_pattern[p], RBF_Nodes[n].x, RBF_Nodes[n].variance)
			#sin_test_phi[p][n] = transfer_function(sin_test_input_pattern[p], RBF_Nodes[n].x, RBF_Nodes[n].variance)
			sin_train_phi[p][n] = transfer_function_2d(sin_training_input_pattern[p], sin_training_output_pattern[p], RBF_Nodes[n].x, RBF_Nodes[n].y, RBF_Nodes[n].variance)
			sin_test_phi[p][n] = transfer_function_2d(sin_test_input_pattern[p], sin_test_output_pattern[p], RBF_Nodes[n].x, RBF_Nodes[n].y, RBF_Nodes[n].variance)
			RANDOM_sin_train_phi[p][n] = transfer_function_2d(sin_training_input_pattern[p], sin_training_output_pattern[p], RANDOM_RBF_Nodes[n].x, RANDOM_RBF_Nodes[n].y, RANDOM_RBF_Nodes[n].variance)
			RANDOM_sin_test_phi[p][n] = transfer_function_2d(sin_test_input_pattern[p], sin_test_output_pattern[p], RANDOM_RBF_Nodes[n].x, RANDOM_RBF_Nodes[n].y, RANDOM_RBF_Nodes[n].variance)
	'''	
	# Calculate SQUARE phi
	square_train_phi = np.zeros((len(square_training_input_pattern), len(RBF_Nodes)))
	square_test_phi = np.zeros((len(square_training_input_pattern), len(RBF_Nodes)))
	for p in range (0, len(square_training_input_pattern)):
		for n in range (0, len(RBF_Nodes)):
			square_train_phi[p][n] = transfer_function(square_training_input_pattern[p], RBF_Nodes[n].x, RBF_Nodes[n].variance)
			square_test_phi[p][n] = transfer_function(square_test_input_pattern[p], RBF_Nodes[n].x, RBF_Nodes[n].variance)
	'''
	'''
	# Least squares
	# SIN calculate weights and absolute residual error 
	sin_least_squares_weight = np.linalg.solve(np.matmul(sin_train_phi.T, sin_train_phi), np.matmul(sin_train_phi.T, sin_training_output_pattern))
	sin_least_squares_output_pattern = np.sum(sin_test_phi * sin_least_squares_weight, axis = 1)
	print("Nodes:", nodes, "SIN Least squares absolute residual error:", absolute_residual_error(sin_test_output_pattern, sin_least_squares_output_pattern))
	'''
	'''
	#SQUARE calculate weights and absolute residual error
	square_least_squares_weight = np.linalg.solve(np.matmul(square_train_phi.T, square_train_phi), np.matmul(square_train_phi.T, square_training_output_pattern))
	square_least_squares_output_pattern = np.sum(square_test_phi * square_least_squares_weight, axis = 1)
	binary(square_least_squares_output_pattern)
	print("Nodes:", nodes, "SQUARE Least squares absolute residual error:", absolute_residual_error(square_test_output_pattern, square_least_squares_output_pattern))
	'''

	# Delta rule
	# Initiate weights
	sin_sequential_weight  = []
	RANDOM_sin_sequential_weight  = []
	#square_sequential_weight = []
	#sin_batch_weight = []
	#square_batch_weight = []
	for i in range(0, len(weight)):
		sin_sequential_weight.append(weight[i])
		RANDOM_sin_sequential_weight.append(weight[i])
		#square_sequential_weight.append(weight[i])
		#sin_batch_weight.append(weight[i])
		#square_batch_weight.append(weight[i])

	epochs = 10000
	
	# Sequential Delta rule--------------------------------------------------------------------------------------------------------------------------------------------
	# SIN
	for i in range(0, epochs):
		shuffle_3(sin_training_output_pattern, sin_train_phi, RANDOM_sin_train_phi)
		for o in range(0, len(sin_training_output_pattern)):
			sin_sequential_weight = sin_sequential_weight + (learning_rate*(sin_training_output_pattern[o] - np.sum(sin_train_phi[o] * sin_sequential_weight))*(sin_train_phi[o]))
			RANDOM_sin_sequential_weight = RANDOM_sin_sequential_weight + (learning_rate*(sin_training_output_pattern[o] - np.sum(RANDOM_sin_train_phi[o] * RANDOM_sin_sequential_weight))*(RANDOM_sin_train_phi[o]))
		sin_sequential_output_pattern = np.sum(sin_test_phi * sin_sequential_weight, axis = 1)
		RANDOM_sin_sequential_output_pattern = np.sum(RANDOM_sin_test_phi * RANDOM_sin_sequential_weight, axis = 1)
		errors.append(squared_error(sin_test_output_pattern, sin_sequential_output_pattern))
		print("Epoch:", i, "SIN Sequential Delta rule error:", squared_error(sin_test_output_pattern, sin_sequential_output_pattern))
		RANDOM_errors.append(squared_error(sin_test_output_pattern, RANDOM_sin_sequential_output_pattern))
		print("Epoch:", i, "RANDOM SIN Sequential Delta rule error:", squared_error(sin_test_output_pattern, RANDOM_sin_sequential_output_pattern))
	#print("Nodes:", nodes, "SIN Sequential Delta rule error:", squared_error(sin_test_output_pattern, sin_sequential_output_pattern))
	
	'''
	# SQUARE
	for i in range(0, epochs):
		for o in range(0, len(square_training_output_pattern)):
			square_sequential_weight = square_sequential_weight + (learning_rate*(square_training_output_pattern[o] - np.sum(square_train_phi[o] * square_sequential_weight))*(square_train_phi[o]))
		square_sequential_output_pattern = np.sum(square_test_phi * square_sequential_weight, axis = 1)
		binary(square_sequential_output_pattern)
		#print("Epoch:", i, "SQUARE Sequential Delta rule error:", squared_error(square_test_output_pattern, square_sequential_output_pattern))
	binary(square_sequential_output_pattern)
	print("Nodes:", nodes, "SQUARE Sequential Delta rule error:", squared_error(square_test_output_pattern, square_sequential_output_pattern))
	# Sequential Delta rule--------------------------------------------------------------------------------------------------------------------------------------------
	
	# Batch Delta rule-------------------------------------------------------------------------------------------------------------------------------------------------
	# SIN
	for i in range(0, epochs):
		sin_batch_output_pattern = np.sum(sin_train_phi * sin_batch_weight, axis = 1)
		sin_batch_weight = sin_batch_weight + (learning_rate*np.sum((sin_training_output_pattern - sin_batch_output_pattern)*sin_train_phi.T, axis = 1))
		#print("Epoch:", i, "SIN Batch Delta rule error:", squared_error(sin_test_output_pattern, sin_batch_output_pattern))
	sin_batch_output_pattern = np.sum(sin_test_phi * sin_batch_weight, axis = 1)
	print("Nodes:", nodes, "SIN Batch Delta rule error:", squared_error(sin_test_output_pattern, sin_batch_output_pattern))

	# SQUARE
	for i in range(0, epochs):
		square_batch_output_pattern = np.sum(square_train_phi * square_batch_weight, axis = 1)
		square_batch_weight = square_batch_weight + (learning_rate*np.sum((square_training_output_pattern - square_batch_output_pattern)*square_train_phi.T, axis = 1))
		binary(square_batch_output_pattern)
		#print("Epoch:", i, "SQUARE Batch Delta rule error:", squared_error(square_test_output_pattern, square_batch_output_pattern))
	square_batch_output_pattern = np.sum(square_test_phi * square_batch_weight, axis = 1)
	binary(square_batch_output_pattern)
	print("Nodes:", nodes, "SQUARE Batch Delta rule error:", squared_error(square_test_output_pattern, square_batch_output_pattern))
	# Batch Delta rule-------------------------------------------------------------------------------------------------------------------------------------------------
	'''
	

# Plot
ax = plt.gca()
'''
# Plot nodes
X = []
Y = []
Circles = []
for node in RBF_Nodes:
	X.append(node.x)
	Y.append(node.y)
	Circles.append(plt.Circle((node.x, node.y), node.variance, color='k', fill=False))

ax.plot(X, Y, "ro")
for circle in Circles:
	ax.add_artist(circle)
'''
# Plot data
#ax.plot(sin_test_input_pattern, sin_sequential_output_pattern)
#ax.plot(sin_test_input_pattern, sin_test_output_pattern)
ax.plot(errors)
ax.plot(RANDOM_errors)

plt.show()



