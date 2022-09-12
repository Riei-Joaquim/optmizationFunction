from Algorithms.GeneticAlgorithm import GA

def main():

    evolutionStrategy = GA(30, -15, 15, 100)
    evolutionStrategy.fit(500000)
    print(evolutionStrategy.bestInd.X)
    print(evolutionStrategy.bestInd.sigma)
    print(evolutionStrategy.bestInd.fitness)

    
if __name__ == "__main__":
    main()