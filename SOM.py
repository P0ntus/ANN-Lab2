import os
import math
import numpy as np
import matplotlib.pyplot as plt

# We format the given data from animals.dat
os.chdir( os.path.dirname(os.path.abspath(__file__)) )

# Open data files
animalnames_f = open("data_lab2/animalnames.txt", "r")
animalnames = animalnames_f.read()
animalnames_f.close()

animalattr_f = open("data_lab2/animalattributes.txt", "r")
animalattr = animalattr_f.read()
animalattr_f.close()

matrix_f = open("data_lab2/animals.dat", "r")
matrix_a = matrix_f.read()
matrix_f.close()

#Format data
animals = animalnames.split("\r\n")
del animals[len(animals) - 1] # An empty element is added in the end, I do not know why ?
attr = animalattr.split("\r\n")

raw_matrix = matrix_a.split(",")

matrix = [ [raw_matrix[j * i] for j in range(0, len(attr)) ] for i in range(0,len(animals)) ]

# HERE the data is ready --- matrix formated with 84 attributes in each line

# Distance whithout the root to same computing time
def distance( x, w ):
	result = []
	for i in range( 0, len(x)):
		result[i] = (x[i] - w[i])**2
	return np.sum( result )

#INITIALISATION
# We generate random weights for a matrix of 100x84 (84 attributes for each node)
nodes = 100
weights = np.array([])
mu = 0
sigma = 0.1
epochs = 20
learning_rate = 0.2
learning_rate_neightbours = 0.2
neightbours_parameter = 5 # Unidimensional, integer suffisant

for i in range(0, nodes):
	weights[i] = np.random.normal(mu, sigma, len(animalattr))
	
for i in range(0, epochs):
	for j in range(0, len(animalnames)): # len(animalnames) is the number of points we have (inner loop)
		# New loop to calculate the distance :
		min_distance = distance( matrix[j], weights[k] )
		index = 0

		for k range( 1, nodes) : 
			result = distance( matrix[j], weights[k] )
			if min_distance > result :
				min_distance = result
				index = k
		
		# Once we have the index of the winner, we can update the weights of the winner and those of the neighbourhoods
		
		weights[index] += learning_rate * (matrix[j] - weights[k])
	



























