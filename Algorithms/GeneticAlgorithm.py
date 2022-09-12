from functools import reduce
from random import paretovariate, random
import numpy as np
from Functions.AckleyFunct import Ackley
from Individual.Individuo import Individuo

class GA:
    
    def __init__(self, nGenes, lowerBound, upperBound, populationSize):
        self.nGenes = nGenes
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.populationSize = populationSize
        self.prob_crossover = 0.9
        self.prob_mutation = 0.4
        self.n_children = 2
        self.alpha = 0.6
        self.population = []
        self.benchMark = Ackley(20, 0.2, 2*np.pi, self.nGenes)
        self.bestInd = None
        self.bestIndGenIt = -1
        
    def init_population(self):
        for i in range(self.populationSize):
            X = np.random.uniform(self.lowerBound, self.upperBound, self.nGenes)
            ind = Individuo(X, None, self.fitness(X))
            self.population.append(ind)
    
    def fitness(self, X):
        return self.benchMark.eval(X)
    
    def update_population(self, next_generation, it):
        self.population = sorted(next_generation, key= lambda e: e.fitness)[0:self.populationSize]

        if self.bestInd is None or self.population[0].fitness < self.bestInd.fitness:
            self.bestIndGenIt = it 
            self.bestInd = self.population[0]
    
    def crossover(self, parents):
        children = []
        
        if self.prob_crossover < np.random.rand():
            for _ in range(self.n_children):
                X = np.zeros(self.nGenes)
                for j in range(self.nGenes):
                    X[j] = self.alpha * parents[0].X[j] + (1 - self.alpha) * parents[1].X[j]
                children.append(Individuo(X, None, self.fitness(X)))
        else:
            X1 = parents[0].X.copy()
            X2 = parents[1].X.copy()
            children = [Individuo(X1, None, self.fitness(X1)),
                        Individuo(X2, None, self.fitness(X2))]
        
        return children
    
    def mutation(self, ind):
        # mutação linear
        if self.prob_mutation < np.random.rand():
            p = 1/self.nGenes
            for i in range(self.nGenes):
                if np.random.rand() < p:
                    ind.X[i] = np.random.uniform(self.lowerBound, self.upperBound)
        return ind
    
    def parent_selection(self, n_parents):
        sumFitness = 0
        roulette = []
        sumRange = 0

        for p in self.population:
            sumFitness += p.fitness

        for i in range(len(self.population)):
            roulette.append({'individuo': self.population[i], 'botomLimit': sumRange/sumFitness, 'upperLimit': (
                sumRange + self.population[i].fitness)/sumFitness})
            sumRange += self.population[i].fitness

        choices = np.random.rand(n_parents)
        parents = []
        for i in range(n_parents):
            for j in range(len(roulette)):
                e = roulette[j]
                if((e['botomLimit'] >= choices[i] and choices[i] <= e['upperLimit']) or j == (len(roulette) - 1)):
                    parents.append(e['individuo'])
                    break
        return parents

    def fit(self, n_iterations):
        self.init_population()
        
        for it in range(n_iterations):
            parents = self.parent_selection(2)
            children = self.crossover(parents)
            for i in range(len(children)):
                children[i] = self.mutation(children[i])
            self.update_population(children + self.population, it)
            
            if (it % 1000 == 0):
                # print(self.population)
                print("Iteration: ", it, " Best fitness: ", self.population[0].fitness, " Best alltime: ", self.bestInd)
