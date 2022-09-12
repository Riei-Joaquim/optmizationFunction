import numpy as np

class Ackley:
    def __init__(self, c1, c2, c3, dim):
        self._a = c1
        self._b = c2
        self._c = c3
        self._d = dim
    
    def eval(self, xList):
        sumP1 = 0
        for xi in xList:
            sumP1 += np.power(xi, 2)

        sumP2 = 0
        for xi in xList:
            sumP2 += np.cos(self._c*xi)
        
        return -self._a*np.exp(- self._b*np.sqrt((1/self._d)*sumP1)) \
                    - np.exp((1/self._d)*sumP2) + self._a + np.exp(1)
