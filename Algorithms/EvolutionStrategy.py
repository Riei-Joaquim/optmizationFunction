import numpy as np
from Functions.AckleyFunct import Ackley
from Individual.Individuo import Individuo

class EvolutionStrategy():
    def __init__(self, lowerBound, upperBound, populationSize, parentsSize):
        self.xMax = upperBound
        self.xMin = lowerBound
        self.sigmaMaxInitial = 4
        self.sigmaMin = 0.5
        self.populationSize = populationSize
        self.sonSize = 7* self.populationSize
        self.dim = 30
        self.parentsSize = parentsSize
        self.benchMark = Ackley(20, 0.2, 2*np.pi, self.dim)
        self.population = []
        self.tau_global = 1/np.sqrt(2 * self.dim)
        self.tau_fine = 1/np.sqrt(2 * np.sqrt(self.dim))
        self.bestInd = None
        self.bestIndGenIt = -1
        self.MaxWaitForImp = 100

    def initPopulation(self):
        for _ in range(self.populationSize):
            X = np.random.uniform(self.xMin, self.xMax, self.dim)
            sigma = np.random.uniform(self.sigmaMin, self.sigmaMaxInitial, self.dim)
            
            fit = self.fitness(X)
            self.population.append(Individuo(X, sigma, fit))
    
    def fitness(self, X):
        return self.benchMark.eval(X)
    
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

    def crossover(self, parents):
        child = np.ndarray(self.dim)

        # two parents, each gene is the average of the two parents
        for i in range(self.dim):
            child[i] = (parents[0].X[i] + parents[1].X[i]) / 2

        # two parents, choice each gene from one of them
        for i in range(self.dim):
            child[i] = np.random.choice(parents).X[i]

        return child
        
    
    def mutation(self, ind):
        
        for i in range(len(ind.sigma)):
            new_sigma = ind.sigma[i] * np.exp(self.tau_global * np.random.normal() + self.tau_fine * np.random.normal())
            ind.sigma[i] = new_sigma if new_sigma > self.sigmaMin else self.sigmaMin
            ind.X[i] = ind.X[i] + ind.sigma[i] * np.random.normal()    
        return ind