import math
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse

from sklearn.neural_network import MLPRegressor

np.set_printoptions(threshold=np.nan) #Always print the whole matrix

def CL(RBF_Nodes, input_pattern, output_pattern, leaky = False):
  iters = 1000
  cl_learning_rate = 0.1
  for _ in range (0, iters):

    # pick random training vector
    i = random.randint(0, len(input_pattern) - 1)
    training_vector = np.asarray((input_pattern[i], output_pattern[i]))

    # find closest rbf_node
    closest_node = None
    closest_distance = float('inf')
    for node in RBF_Nodes:
      npNode = np.asarray((node.x, node.y))
      distance = np.linalg.norm(training_vector - npNode)
      if distance < closest_distance:
        closest_distance = distance
        closest_node = node

    if closest_node == None:
      continue

    # move closest rbf_node closer to traning vector, dw = eta(x - w)
    delta_node = cl_learning_rate * (training_vector - np.asarray((closest_node.x, closest_node.y)))

    closest_node.x += delta_node[0]
    closest_node.y += delta_node[1]

    # consider strategy for dead units (e.g., leaky cl)
    if leaky:
      leaky_learning_rate = 0.01
      for node in RBF_Nodes:
        if node != closest_node:
          # use gauss function to limit how leaky it is, nodes further away are less affected
          npNode = np.asarray((node.x, node.y))
          distance = np.linalg.norm(training_vector - npNode)
          gauss_factor = transfer_function(distance, 0, 0.5)
          delta_node = gauss_factor * leaky_learning_rate * (training_vector - np.asarray((node.x, node.y)))
          node.x += delta_node[0]
          node.y += delta_node[1]


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
	#sin_training_input_pattern[0] += 0.0001
	#sin_training_input_pattern = noise(sin_training_input_pattern)
	sin_training_output_pattern = list(map(sin_function, sin_training_input_pattern))

	# Testing patterns - Generate values between 0.05 and 2pi with step length 0.1 using our sin_function
	sin_test_input_pattern = np.asarray(np.arange(x_lower_interval + (step_length/2), x_upper_interval, step_length))
	#sin_test_input_pattern = noise(sin_test_input_pattern)
	sin_test_output_pattern = list(map(sin_function, sin_test_input_pattern))

	# SIN DATA----------------------------------------------------------------------------------------------------------

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
	for c in range(0, NUM_NODES_COL):
		for r in range(0, NUM_NODES_ROW):
			x = (x_lower_interval + ((x_upper_interval - x_lower_interval)/NUM_NODES_ROW/2)) + r * ((x_upper_interval - x_lower_interval)/NUM_NODES_ROW)
			y = -(NUM_NODES_COL - 1) * variance + c * 2 * variance
			#print(float(random.randint(x_lower_interval*1000, math.ceil(x_upper_interval)*1000))
			#print(x, y)
			weight.append(np.random.normal(mu, sigma, 1)[0])
			RBF_Nodes.append(Node(x, y, variance))

	CL(RBF_Nodes, sin_training_input_pattern, sin_training_output_pattern, True)

	# Calculate SIN phi
	sin_train_phi = np.zeros((len(sin_training_input_pattern), len(RBF_Nodes)))
	sin_test_phi = np.zeros((len(sin_training_input_pattern), len(RBF_Nodes)))
	for p in range (0, len(sin_training_input_pattern)):
		for n in range (0, len(RBF_Nodes)):
			sin_train_phi[p][n] = transfer_function(sin_training_input_pattern[p], RBF_Nodes[n].x, RBF_Nodes[n].variance)
			sin_test_phi[p][n] = transfer_function(sin_test_input_pattern[p], RBF_Nodes[n].x, RBF_Nodes[n].variance)
			#sin_train_phi[p][n] = transfer_function_2d(sin_training_input_pattern[p], sin_training_output_pattern[p], RBF_Nodes[n].x, RBF_Nodes[n].y, RBF_Nodes[n].variance)
			#sin_test_phi[p][n] = transfer_function_2d(sin_test_input_pattern[p], sin_test_output_pattern[p], RBF_Nodes[n].x, RBF_Nodes[n].y, RBF_Nodes[n].variance)

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

	epochs = 20000
	
	# Sequential Delta rule--------------------------------------------------------------------------------------------------------------------------------------------
	# SIN
	for i in range(0, epochs):
		shuffle(sin_training_output_pattern, sin_train_phi)
		for o in range(0, len(sin_training_output_pattern)):
			sin_sequential_weight = sin_sequential_weight + (learning_rate*(sin_training_output_pattern[o] - np.sum(sin_train_phi[o] * sin_sequential_weight))*(sin_train_phi[o]))
		sin_sequential_output_pattern = np.sum(sin_test_phi * sin_sequential_weight, axis = 1)
		errors.append(squared_error(sin_test_output_pattern, sin_sequential_output_pattern))
		print("Epoch:", i, "SIN Sequential Delta rule error:", absolute_residual_error(sin_test_output_pattern, sin_sequential_output_pattern))
		err = absolute_residual_error(sin_test_output_pattern, sin_sequential_output_pattern)
		if err < 0.01:
			break
	#print("Nodes:", nodes, "SIN Sequential Delta rule error:", squared_error(sin_test_output_pattern, sin_sequential_output_pattern))
'''
	#print(sin_training_output_pattern)
	#sin_training_input_pattern = np.array(sin_training_output_pattern)
	sin_training_input_pattern = sin_training_input_pattern.reshape(len(sin_training_input_pattern), 1)
	sin_training_output_pattern = np.array(sin_training_output_pattern)
	#sin_training_output_pattern = sin_training_output_pattern.reshape(len(sin_training_output_pattern), 1)
	sin_test_input_pattern = np.array(sin_test_input_pattern)
	sin_test_input_pattern = sin_test_input_pattern.reshape(len(sin_test_input_pattern), 1)
	#print(sin_training_output_pattern)
	#print()
	#print(sin_training_output_pattern - training_target_pattern.mean(axis=0))/training_target_pattern.std(axis=0) / 2.5)
	#sin_training_input_pattern = np.transpose(sin_training_input_pattern)
	#sin_test_input_pattern = np.transpose(sin_test_input_pattern)

	parser = argparse.ArgumentParser(description='MLP network for Mackey-Glass time series predictions.')
	parser.add_argument('-n', '--hidden-nodes', type=int, nargs='+', default=30,
					   help='number of nodes in the hidden layers (max 8 per layer)')
	parser.add_argument('-l', '--learning-rate', type=float, default=0.001,
					   help='the learning rate, controls how fast it converges')
	parser.add_argument('-a', '--alpha', type=float, default=0.0001,
					   help='the L2 regularization factor')
	#parser.add_argument('-b', '--batch_size', type=int, default=len(sin_training_input_pattern),
					  # help='??')
	args = parser.parse_args()
	#print(sin_training_input_pattern.shape)
	#print(sin_training_output_pattern.shape)
	reg = MLPRegressor(hidden_layer_sizes=args.hidden_nodes, early_stopping=True, max_iter=10000,
                   learning_rate_init=args.learning_rate, alpha=args.alpha, batch_size=len(sin_training_input_pattern))
	reg = reg.fit(sin_training_input_pattern, sin_training_output_pattern)
	two_layer_output = reg.predict(sin_test_input_pattern)
'''

# Plot
ax = plt.gca()

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
# Plot data
# 
#ax.plot(sin_test_input_pattern, sin_sequential_output_pattern)
ax.plot(sin_test_input_pattern, sin_test_output_pattern)
#ax.plot(sin_test_input_pattern, sin_least_squares_output_pattern)
#ax.plot(sin_test_input_pattern, two_layer_output)

plt.show()



