import numpy as np

class Ackley:
    def __init__(self, a, b, c, d):
        self._a = a
        self._b = b
        self._c = c
        self._d = d
    
    def eval(self, xList):
        sumP1 = 0
        for xi in xList:
            sumP1 += xi**2

        sumP2 = 0
        for xi in xList:
            sumP2 += np.cos(self._c*xi)
        
        return -self._a*np.exp(- self._b*np.sqrt((1/self._d)*sumP1)) \
                    - np.sqrt((1/self._d)*sumP2) + self._a + np.exp(1)
