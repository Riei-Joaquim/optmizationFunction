from statistics import mean, stdev
import numpy as np
from Functions.AckleyFunct import Ackley
from Individual.Individuo import Individuo
from random import random 
import matplotlib.pyplot as plt 
import datetime
from executionUtils import ExecutionStrategy

class EvolutionStrategy():
    def __init__(self, lowerBound, upperBound, populationSize, parentsSize, execMode):
        self.xMax = upperBound
        self.xMin = lowerBound
        self.sigmaMaxInitial = 4
        self.sigmaMin = 0.5
        self.populationSize = populationSize
        self.sonSize = 7* self.populationSize
        self.dim = 30
        self.executionMode = execMode
        self.parentsSize = parentsSize
        self.benchMark = Ackley(20, 0.2, 2*np.pi, self.dim)
        self.population = []
        self.tau_global = 1/np.sqrt(2 * self.dim)
        self.tau_fine = 1/np.sqrt(2 * np.sqrt(self.dim))
        self.bestInd = None
        self.bestIndGenIt = -1
        self.MaxWaitForImp = 1000
        self.medFitness = None
        self.devFitness = None
        self.minEvolutionStep = 1e-3
        self.curEvolutionStep = None

    def initPopulation(self):
        fitness = []
        for i in range(self.populationSize):
            X = np.random.uniform(self.xMin, self.xMax, self.dim)
            sigma = np.random.uniform(self.sigmaMin, self.sigmaMaxInitial, self.dim)
            
            fit = self.fitness(X)
            fitness.append(fit)
            self.population.append(Individuo(X, sigma, fit))
        self.medFitness = mean(fitness)
        self.devFitness = stdev(fitness)
    
    def fitness(self, X):
        return self.benchMark.eval(X)
    
    def isConverged(self, it):
        if self.bestIndGenIt == -1 or self.bestInd is None:
            return False

        if abs(it - self.bestIndGenIt) > self.MaxWaitForImp and (self.curEvolutionStep is not None and self.curEvolutionStep < self.minEvolutionStep):
            return True
        
        return False
        
    def selectParents(self):
        parents = []
        for i in range(self.parentsSize):
            parents.append(np.random.choice(self.population))
        
        return parents
    
    def selectParentsByRoulette(self):
        totFit = 0
        for i in range(self.populationSize):
            totFit += self.population[i].fitness
        
        rangeForIdx = []
        sum = 0
        for i in range(self.populationSize):
            step = self.population[i].fitness/totFit
            sum += step
            rangeForIdx.append(sum)


        parents = []
        for i in range(self.parentsSize):
            select = random()
            for j in range(self.populationSize):
                if select <= rangeForIdx[j] or j == (self.populationSize - 1):
                    parents.append(self.population[j])
                    break
        
        return parents
    
    def updatePopulation(self, nextGeneration, it):
        self.population = sorted(nextGeneration, key= lambda e: e.fitness)[0:self.populationSize]

        if self.bestInd is None or self.population[0].fitness < self.bestInd.fitness:
            if self.bestInd is not None:
                self.curEvolutionStep = abs(abs(self.bestInd.fitness) - abs(self.population[0].fitness))
                
                if self.curEvolutionStep > self.minEvolutionStep:
                    self.bestIndGenIt = it 

            self.bestInd = self.population[0]
        
        fitness = []
        for np in self.population:
            fitness.append(np.fitness)
        
        self.medFitness = mean(fitness)
        self.devFitness = stdev(fitness)

    def crossover(self, parents):
        child_X = np.zeros(self.dim)
        child_sigma = np.zeros(self.dim)
        
        # two parents, each gene is the average of the two parents
        #for i in range(self.dim):
        #   for j in range(self.parentsSize):
        #       child_sigma[i] += parents[j].sigma[i]
        #   child_sigma[i] /= self.parentsSize
        #   child_X[i] = np.random.choice(parents).X[i]

        for i in range(self.dim):
            child_sigma[i] = np.random.choice(parents).sigma[i]
            child_X[i] = np.random.choice(parents).X[i]

        return child_X, child_sigma
    
    def mutation(self, ind):
        
        for i in range(len(ind.sigma)):
            new_sigma = ind.sigma[i] * np.exp(self.tau_global * np.random.normal() + self.tau_fine * np.random.normal())
            #old_sigma = ind.sigma[i]
            #if (new_sigma > 100 or old_sigma > 100):
                #print(new_sigma, old_sigma)
            
            ind.sigma[i] = new_sigma #if new_sigma > self.sigmaMin else self.sigmaMin
            valor = ind.X[i] + ind.sigma[i] * np.random.normal(0,self.tau_fine)
            
            ind.X[i] = valor
        return ind
    
    def fit(self, n_iterations):
        self.initPopulation()
        
        iteration = []
        fitness_it = []
        popFitnessMed = []
        popFitnessDev = []
        for it in range(n_iterations):
            childs = []
            for _ in range(self.sonSize):
                parents = self.selectParentsByRoulette()
                child_x, child_sigma = self.crossover(parents)
                ind = Individuo(child_x, child_sigma, self.fitness(child_x))
                ind = self.mutation(ind)
                childs.append(ind)
            self.updatePopulation(childs, it)
            
            
            if (it % 10 == 0):
                # print(self.population)
                iteration.append(it)
                fitness_it.append(self.bestInd.fitness)
                popFitnessMed.append(self.medFitness)
                popFitnessDev.append(self.devFitness)
                
                print("Iteration: ", it, " Best fitness: ", self.population[0].fitness, " Best alltime: ", self.bestInd)
            
            if self.isConverged(it):
                print("Converged -> Iteration: ", it, " Best fitness: ", self.population[0].fitness, " Best alltime: ", self.bestInd)
                break
        
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness')
        plt.plot(iteration, fitness_it, popFitnessMed, popFitnessDev)
        
        generation_time = datetime.datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
        path_graph = "data/individuals_execution_" + generation_time + ".png"
        plt.savefig(path_graph)
