from functools import reduce
from random import random
import numpy as np
from Functions.AckleyFunct import Ackley
from Individual.Individuo import Individuo

class GA:
    
    def __init__(self, nGenes, lowerBound, upperBound, populationSize):
        self.nGenes = nGenes
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.populationSize = populationSize
        self.population = []
        self.benchMark = Ackley(20, 0.2, 2*np.pi, self.nGenes)
        
    def init_population(self):
        for i in range(self.populationSize):
            X = np.random.uniform(self.lowerBound, self.upperBound)
            ind = Individuo(X, None, self.fitness(X))
            self.population.append(ind)
    
    def fitness(self, X):
        return self.benchMark.eval(X)
    
    def crossover(self):
        pass
    
    def mutation(self):
        pass
    
    def parent_selection(self):
        parents = []
        population_fitness = 1/(reduce(lambda a, b: a.fitness + b.fitness, self.population) + 0.1)
        
        for i in range(self.populationSize):
            sum_ = 0
            choice = np.random.rand() * population_fitness
            
            for j in range(self.populationSize):
                sum_ += 1/(self.population[j].fitness+0.1)
                if sum_ > choice:
                    parents.append(self.population[j])
                    break  
        return parents
