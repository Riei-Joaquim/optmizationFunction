from dataclasses import dataclass
import numpy as np

@dataclass
class Individuo:
    X:np.ndarray
    fitness:np.float32
    
    def __init__(self, X, fitness):
        self.X = X
        self.fitness = fitness
