import math
import numpy as np
import matplotlib.pyplot as plt

x_lower_interval = 0
x_upper_interval = 2*math.pi
y_lower_interval = -1
y_upper_interval = 1
step_length = 0.1

# Class that represents a 2d input and what type it should be classified as
class Node:
    def __init__(self, x, y, v):
        self.x = x
        self.y = y
        self.variance = v
    def __repr__(self):
        return "<x:%s y:%s>" % (self.x, self.y)
    def __str__(self):
        return "member of Test"

# Mean squared error
def mean_squared_error( expected, predicted ):
	return np.sum((expected - predicted) ** 2)/len(expected)

# Like mean_squared_error but without dividing the result by total length
def squared_error( expected, predicted ):
	return np.sum((expected - predicted) ** 2)
	
# We use Gaussian RBF's with the following transfer function
def transfer_function(x, position, variance):
	return (math.exp((-(x - position)**2) / (2*(variance**2))))
	
def sin_function(x):
	return math.sin(2*x)
	
	
# Generate function data

# Training patterns - Generate values between 0 and 2π with step length 0.1 using our sin_function
sin_training_input_pattern = np.asarray(np.arange(x_lower_interval, x_upper_interval, step_length))
sin_training_output_pattern = list(map(sin_function, sin_training_input_pattern))

# Testing patterns - Generate values between 0.05 and 2π with step length 0.1 using our sin_function
sin_test_input_pattern = np.asarray(np.arange(x_lower_interval + 0.05, x_upper_interval, step_length))
sin_test_output_pattern = list(map(sin_function, sin_test_input_pattern))

# Initiate RBF nodes
NUM_NODES_ROW = 4
NUM_NODES_COL = 2
variance = 0.25
RBF_Nodes = []
for c in range(0, NUM_NODES_COL):
	for r in range(0, NUM_NODES_ROW):
		x = (x_lower_interval + ((x_upper_interval - x_lower_interval)/NUM_NODES_ROW/2)) + r * ((x_upper_interval - x_lower_interval)/NUM_NODES_ROW)
		y = (y_lower_interval + ((y_upper_interval - y_lower_interval)/NUM_NODES_COL/2)) + c * ((y_upper_interval - y_lower_interval)/NUM_NODES_COL)
		RBF_Nodes.append(Node(x, y, variance))

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

