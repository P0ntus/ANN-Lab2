import math
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan) #Always print the whole matrix

x_lower_interval = 0
x_upper_interval = 2*math.pi
y_lower_interval = -1
y_upper_interval = 1
step_length = 0.1

learning_rate = 0.01

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
def mean_squared_error( expected, predicted ):
	return np.sum((expected - predicted) ** 2)/len(expected)

# Squared error
def squared_error( expected, predicted ):
	return np.sum((expected - predicted) ** 2)
	
# We use Gaussian RBF's with the following transfer function
def transfer_function(x, position, variance):
	return (math.exp((-(x - position)**2) / (2*(variance**2))))

def euclidean_distance(x1, y1, x2, y2):
	return (math.sqrt(((x1 - x2)**2) + ((y1 - y2)**2)))
	
def sin_function(x):
	return math.sin(2*x)
	
	
# Generate function data

# Training patterns - Generate values between 0 and 2π with step length 0.1 using our sin_function
sin_training_input_pattern = np.asarray(np.arange(x_lower_interval, x_upper_interval, step_length))
sin_training_output_pattern = list(map(sin_function, sin_training_input_pattern))

# Testing patterns - Generate values between 0.05 and 2π with step length 0.1 using our sin_function
sin_test_input_pattern = np.asarray(np.arange(x_lower_interval + (step_length/2), x_upper_interval, step_length))
sin_test_output_pattern = list(map(sin_function, sin_test_input_pattern))

# Initiate RBF nodes
NUM_NODES_ROW = len(sin_training_output_pattern)
NUM_NODES_COL = 1
variance = 1.25
mu, sigma = 0, 0.1 # used for weight initialization
RBF_Nodes = []
weight = []
for c in range(0, NUM_NODES_COL):
	for r in range(0, NUM_NODES_ROW):
		x = (x_lower_interval + ((x_upper_interval - x_lower_interval)/NUM_NODES_ROW/2)) + r * ((x_upper_interval - x_lower_interval)/NUM_NODES_ROW)
		y = (y_lower_interval + ((y_upper_interval - y_lower_interval)/NUM_NODES_COL/2)) + c * ((y_upper_interval - y_lower_interval)/NUM_NODES_COL)
		weight.append(np.random.normal(mu, sigma, 1)[0])
		#weight = 1
		RBF_Nodes.append(Node(x, y, variance))
		
# Calculate phi
phi = np.zeros((len(sin_training_input_pattern), len(RBF_Nodes)))
for p in range (0, len(sin_training_input_pattern)):
	for n in range (0, len(RBF_Nodes)):
		phi[p][n] = transfer_function(sin_training_input_pattern[p], RBF_Nodes[n].x, RBF_Nodes[n].variance)

# Calculate weights using Least squares
least_squares_weight = np.linalg.solve(phi.T @ phi, phi.T @ sin_training_output_pattern)
output_pattern = np.sum(phi * least_squares_weight, axis = 1)
print("Least squares error:", squared_error(sin_training_output_pattern, output_pattern))

# Calculate weights using Delta rule
sequential_weight = []
batch_weight = []
for i in range(0, len(weight)):
	sequential_weight.append(weight[i])
	batch_weight.append(weight[i])

epochs = 100

# Sequential Delta rule
for i in range(0, epochs):
	for o in range(0, len(output_pattern)):
		sequential_weight = sequential_weight + (learning_rate*(sin_training_output_pattern[o] - np.sum(phi[o] * sequential_weight))*(phi[o]))
	sequential_output_pattern = np.sum(phi * sequential_weight, axis = 1)
	print("Sequential Delta rule error:", squared_error(sin_training_output_pattern, sequential_output_pattern))

# Batch Delta rule
for i in range(0, epochs):
	batch_output_pattern = np.sum(phi * batch_weight, axis = 1)
	batch_weight = batch_weight + (learning_rate*(sin_training_output_pattern - batch_output_pattern)*phi)
	print("Batch Delta rule error:", squared_error(sin_training_output_pattern, batch_output_pattern))

'''	
# Plot function and nodes
X = []
Y = []
Circles = []
for node in RBF_Nodes:
	X.append(node.x)
	Y.append(node.y)
	Circles.append(plt.Circle((node.x, node.y), node.variance, color='k', fill=False))

ax = plt.gca()
ax.plot(sin_training_input_pattern, sin_training_output_pattern)
ax.plot(X, Y, "ro")

for circle in Circles:
	ax.add_artist(circle)

plt.show()
'''

