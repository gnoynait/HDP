import unittest
import numpy as np
import hdpgmm as hdp

class GmmTest(unittest.TestCase):
    def test_FullFactorSpheGaussianMixture(self):
        dim = 2
        comp = 5
        datasize = 100
        gmm = hdp.FullFactorSpheGaussianMixture(comp, dim)
        data = np.random.randn(datasize, dim)
        logprob = gmm.calcLogProb(data)
        print logprob.shape, 'logprob.shape'
        gmm.update(data, logprob, 0.5)
class TestSheduler(unittest.TestCase): 
    def test_DecaySheduler(self):
        sh = hdp.DecaySheduler(2, 0.5, 0.5)
        for i in range(10):
            print sh.nextRate()
    def test_ConstSheduler(self):
        sh = hdp.ConstSheduler(0.5)
        for i in range(10):
            print sh.nextRate()
class TestWeight(unittest.TestCase):
    def test_NonBeyesianWeight(self):
        comp = 3
        weight = hdp.NonBayesianWeight(comp)
        lgw = weight.logWeight()
        print lgw
        weight.update(np.exp(np.random.randn(comp)), 0.5)
    def test_DirichletWeight(self):
        comp = 3
        weight = hdp.DirichletWeight(comp, 2)
        lgw = weight.logWeight()
        print lgw
        weight.update(np.exp(np.random.randn(comp)), 0.5)
    def test_StickBreakingWeight(self):
        comp = 3
        weight = hdp.StickBreakingWeight(comp, 2)
        lgw = weight.logWeight()
        print lgw
        weight.update(np.exp(np.random.randn(comp)), 0.5)
        
if __name__ == '__main__':
    unittest.main()
