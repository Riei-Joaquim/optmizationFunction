from enum import Enum
from Algorithms.EvolutionStrategy import EvolutionStrategy
from executionUtils import ExecutionStrategy
from Algorithms.GeneticAlgorithm import GA

ALGORTHM = "EvolutionAlgorithm"

def main():
    if ALGORTHM == "EvolutionAlgorithm":
        evolutionStrategy = EvolutionStrategy(-15, 15, 100, 5, ExecutionStrategy.EEExplorationCompensation)
        evolutionStrategy.fit(2000)
        print(evolutionStrategy.bestInd.X)
        print(evolutionStrategy.bestInd.sigma)
        print(evolutionStrategy.bestInd.fitness)
    else:
        geneticAlgorithm = GA(30, -15, 15, 100)
        geneticAlgorithm.fit(500000)
        print(geneticAlgorithm.bestInd.X)
        print(geneticAlgorithm.bestInd.sigma)
        print(geneticAlgorithm.bestInd.fitness)

    
if __name__ == "__main__":
    main()