# the following piece of code finds a solution to the traveling salesman problem using the Genetic Algorithm
# for this it reads the coordinates of cities from a user generated input file and prints the best tour for the salesman
# as per other conditions of the iteration. The output is something like 2-4-3-1-0-2 which represent the tour starting from 
# city 2 and visiting cities 4,3,1,0 in that order and returning to 2. Here the cities as numbered in the order in which they
# appear in the input file

import sys
from random import *
from math import *
import random
import operator

# this function reads the co-ordinates of the various cities in the Travelling Salesman Problem 
# and saves them in a 2D list cityCoordinates
def readCoordinates (inputFile):
	cityCoordinates = []
	
	# creating file object 'data' to read the coordinates
	data = open(inputFile) 
	
	# looping through the file and avoiding empty last lines if any. Returing the 2D list containing city coordinates
	for line in data:
		if ',' in line:
			x,y = line.strip().split(',')
			cityCoordinates.append((float(x),float(y)))
	return cityCoordinates

# the following function takes in the all city coordinates and generates a 'distances' dictionary which contains the 
# Eucleidian (Straight line distance/ CArtesian distance) distances between each pair of city present in the problem. 
# In our case, distances for (i,j) will be same as (j,i) 
def calcDistances(cityCoordinates):
	distances = {}
	for i, (x1,y1) in enumerate (cityCoordinates):
		for j, (x2,y2) in enumerate (cityCoordinates):
			
			# calculate straight line distance
			distances[i,j] = sqrt((x2-x1)**2 + (y2-y1)**2)
	
	return distances

# the following function takes in a given tour of cities and returns the fitness value - 'fitnessValue' of that 
# particular tour/ chromsome. It does so by calculating the total tour length and the fitness value is then the 
# inverse of the tour length,since we look to maximize the fitness function. the tour length takes into account 
# that the salesman returns to the city he started from 
def fitnessFunction(distances, tour):
	tour_length = 0.0
	fitnessValue = 0.0
	N = len(tour)
	for i in range(N-1):
		
		# ensuring that we return to first city again 
		j = (i+1)%N
		
		# summing up values of all individual tour lengths. Let us assume the tour is 2-4-3-5-1-2 which is going from 
		# city 2 to city 4 to city 3 to city 5 to city 1 and returning to city 2.  
		tour_length+= distances[tour[i], tour[j]]
	
	if tour_length != 0: 
		fitnessValue = 1.0/tour_length 
	
	return fitnessValue

# the following function generates an initial population of chromosomes (tours) from the entire population of cities 
# as per the user input pop_size. This set 'population' would be considered for reproduction 
def generateInitialPopulation (pop_size, numberOfCities):
	counter = 0
	population = []
	while counter<pop_size:
		tour = range(numberOfCities)
		
		# generating a random tour of appropriate length
		shuffle(tour)
		
		# appending to initial population such that we have a unique set
		if tour not in population:
			population.append(tour) 
			counter+=1
	
	return population

# the following function returns the chromsome (tour) which is the fittest among the set of tours provided as input
def getFittest (population):
	blockSize = len(population)
	pop_fitness = {}
	fittest = 0.0
	
	# calculating fitness value of each tour in population
	for i in range(blockSize):
		pop_fitness[i] = fitnessFunction(distances, population[i])
	
	# getting the key of the entry with maximum value in the dictionary pop_fitness
	fittest = max(pop_fitness.iteritems(), key=operator.itemgetter(1))[0]
	
	# returning the chromosome(tour) which is the fittest
	return population[fittest]


# the following function generates 1 offspring of 2 parents. The way it does is using the concept of Tournament Selection: 
# it first selects 3 tours from the population, then it selects the fittest of the 3 as 'parent1'. Similarly 'parent2' 
# is selected these 2 parents are then passed on to the reproduce function which then generates a child/ offspring 
def generateOffspring(population):
	i = []
	parent1 = []
	parent2 = []
	
	for k in range(3):
		# selecting 3 random chromosomes
		i.append(random.choice(population))
	
	# selecting parent using Tournament Selection
	parent1 = getFittest(i)

	# parent2 selected in a similar manner as parent1
	for k in range(3):
		i.append(random.choice(population))
	parent2 = getFittest(i)
	
	# producing a single child of the two parents
	child = reproduce(parent1,parent2)
	
	return child

# the following function takes in 2 parent and returns a child by crossover. For this: a random number 'i' is 
# generated. Child contains elements as is from parent1 upto index 'i'. after that, elements from parent2 which are not
# already present in the child are appended to the child since we do not want to visit only unique cities and want to 
# cover all cities  
def reproduce (parent1, parent2):
	child = []
	i = random.choice(range(len(parent1)))
	
	# inheriting from parent1
	for j in range(i):
		child.insert(j,parent1[j])
	
	# inheriting other unique elements from parent2
	for k in range (len(parent2)):
		if parent2[k] not in child:
			child.append(parent2[k])
	
	return child

# the following function mutates the child with a user defined probability. For this a tworandomly selected elements 
# of the child are swapped  
def mutate(child):
	rand = random.random()
	
	# checking whether to mutate as per the probability
	if rand < p_mutate:
		# generating two random indices to swap elements 
		rand1 = randint(0,len(child)-1)
		rand2 = randint(0,len(child)-1)
	
		# mutating the child : swapping the two elements
		swap = child[rand1]
		child[rand1] = child[rand2]
		child[rand2] = swap
	
	return child

# the following function produces the next generation for the required number of times (user defined) and returns the 
# fittest solution/ tour/ chromosome after the required number of iterations are done 
def geneticAlgo(numberOfGenerations, population, pop_size):
	for i in range (numberOfGenerations):
		nextGen = []
		
		# producing a generation of children equal in number to the parent generation 
		for j in range(pop_size):
			# generating offspring/ child
			child = generateOffspring(population)
			# mutating the child
			child = mutate(child)
			# adding the child to next generation
			nextGen.append(child)
		
		# now the new generation becomes the parent generation for the next outcome
		population = nextGen
	
	# getting the fittest solution 
	fittestSolution = getFittest(population)
	
	# returing the fittest solution
	return fittestSolution

# running the code by taking required user inputs 
p_mutate = input("Enter prob of mutation ")
pop_size = input("Enter size of population of chromosomes to be considered for reproduction: ") 
numberOfGenerations = input("Enter number of generations to be considered: ")
inputFile = "input.txt"
nextGen = []

# generating the list of all city coordinates
cityCoordinates = readCoordinates(inputFile)

# calculating pairwise distances of all cities 
distances = calcDistances(cityCoordinates)

# calculating number of cities
numberOfCities = len(cityCoordinates)

# generating the first generation
population = generateInitialPopulation(pop_size, numberOfCities)

# running the genetic algorithm to get the best possible tour
fittestSolution = geneticAlgo(numberOfGenerations, population, pop_size)

# printing the best tour for the salesman to take
# the point of returning to the start city is taken care of in fitnessFunction() which calculates the 
# tour length and hence the fitness value accordingly. Thus fittestSolution contains the tour from the first 
# to the last city
print "The best tour is: "
for i in fittestSolution:
	print i,"-", 
print fittestSolution[0]
