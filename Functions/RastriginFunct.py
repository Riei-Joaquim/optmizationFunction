import numpy as np

class Rastrigin:
    def __init__(self, dim):
        self._d = dim
    
    def eval(self, xList):
        sumP1 = 0
        for xi in xList:
            sumP1 += np.power(xi, 2) - 10*np.cos(2*np.pi*xi)
        
        return 10*self._d + sumP1
