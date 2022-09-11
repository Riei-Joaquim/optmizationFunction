from Algorithms.EvolutionStrategy import EvolutionStrategy

def main():

    evolutionStrategy = EvolutionStrategy(-15, 15, 30, 2)
    evolutionStrategy.fit(200000)
    print(evolutionStrategy.bestInd.X)
    print(evolutionStrategy.bestInd.sigma)
    print(evolutionStrategy.bestInd.fitness)

    
if __name__ == "__main__":
    main()