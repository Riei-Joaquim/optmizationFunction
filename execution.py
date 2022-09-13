from Algorithms.EvolutionStrategy import EvolutionStrategy

def main():

    evolutionStrategy = EvolutionStrategy(-15, 15, 30, 5)
    evolutionStrategy.fit(2000)
    print(evolutionStrategy.bestInd.X)
    print(evolutionStrategy.bestInd.sigma)
    print(evolutionStrategy.bestInd.fitness)

    
if __name__ == "__main__":
    main()