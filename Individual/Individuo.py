from dataclasses import dataclass
import numpy as np

@dataclass
class Individuo:
    X:np.ndarray
    sigma:np.ndarray
    fitness:np.float32
    
    def __init__(self, X, sigma, fitness):
        self.X = X
        self.sigma = sigma
        self.fitness = fitness
