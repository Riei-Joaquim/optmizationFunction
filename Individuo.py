from dataclasses import dataclass
import numpy as np

@dataclass
class Individuo:
    vObject:np.ndarray
    fitness:np.float32