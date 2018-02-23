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

