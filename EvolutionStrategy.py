import numpy as np
from AckleyFunct import Ackley
from Individuo import Individuo

class EvolutionStrategy():
    def __init__(self, lowerBound, upperBound, populationSize, parentsSize):
        self.xMax = upperBound
        self.xMin = lowerBound
        self.populationSize = populationSize
        self.sonSize = 7* self.populationSize
        self.dim = 30
        self.parentsSize = parentsSize
        self.benchMark = Ackley(20, 0.2, 2*np.pi, self.dim)
        self.population = []
        self.tau = 1/np.sqrt(self.dim)
        self.bestInd = None
        self.bestIndGenIt = -1
        self.MaxWaitForImp = 100

    def initPopulation(self):
        for _ in range(self.populationSize):
            ind = np.random.uniform(self.xMin, self.xMax, self.dim)
            fit = self.benchMark.eval(ind)
            self.population.append(Individuo(ind, fit))
    
    def isConverged(self, it):
        if self.bestIndGenIt == -1 or self.bestInd is None:
            return False

        if abs(it - self.bestIndGenIt) > self.MaxWaitForImp:
            return True
        
    def selectParents(self):
        parents = []
        for i in range(self.parentsSize):
            parents.append(np.random.choice(self.population))
        
        return parents
    
    def updatePopulation(self, nextGeneration, it):
        self.population = sorted(nextGeneration, key= lambda e: e.fitness, reverse= True)[0:self.populationSize]

        if self.bestInd is None or self.population[0].fitness < self.bestInd.fitness:
            self.bestIndGenIt = it 
            self.bestInd = self.population[0]
    
    

    