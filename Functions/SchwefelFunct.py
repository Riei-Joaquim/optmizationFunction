import numpy as np

class Schwefel:
    def __init__(self, dim):
        self._d = dim
    
    def eval(self, xList):
        sumP1 = 0
        for xi in xList:
            sumP1 += xi*np.sin(np.sqrt(np.abs(xi)))
        
        return 418.9829*self._d - sumP1
