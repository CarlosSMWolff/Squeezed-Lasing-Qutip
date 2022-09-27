from qutip import *
import numpy as np
from numpy import sqrt

class simulation:
    def __init__(self,Ncav = 30):
        self.s = tensor(destroy(Ncav),identity(2),identity(2),identity(2))
        self.sigma = tensor(identity(Ncav),destroy(2),identity(2),identity(2))
        self.sigma1 = tensor(identity(Ncav),identity(2),destroy(2),identity(2))
        self.sigma2 = tensor(identity(Ncav),identity(2),identity(2),destroy(2))


    def gen_H(self,g,gsens,Delta1,Delta2,r=0.):
        s = self.s; sigma = self.sigma; sigma1 = self.sigma1;
        sigma2 = self.sigma2; 
        a = np.cosh(r)*s -np.sinh(r)*s.dag()

        return Delta1*sigma1.dag()*sigma1 + Delta2*sigma2.dag()*sigma2\
         + g*(s.dag()*sigma + s*sigma.dag()) + gsens*(a.dag()*(sigma1+sigma2) + a*(sigma1+sigma2).dag()) 

    def gen_Lops(self,kappa,P,Gamma):
        s = self.s; sigma = self.sigma; sigma1 = self.sigma1; sigma2 = self.sigma2; 
        return [sqrt(kappa)*s, sqrt(P)*sigma.dag(),sqrt(Gamma)*sigma1,sqrt(Gamma)*sigma2 ]

    def g2g1(self,DeltaPair,g,gsens,kappa,P,Gamma,r):
        Delta1 = DeltaPair[0]
        Delta2 = DeltaPair[1]
        H = self.gen_H(g,gsens,Delta1,Delta2,r)
        Lops = self.gen_Lops(kappa,P,Gamma)

        rho_ss = steadystate(H,Lops,method='iterative-lgmres')

        sigma1 = self.sigma1; sigma2 = self.sigma2; 
        G2w1w2 = expect(sigma1.dag()*sigma2.dag()*sigma1*sigma2,rho_ss)
        n1 = expect(sigma1.dag()*sigma1,rho_ss)
        n2 = expect(sigma2.dag()*sigma2,rho_ss)

        g2w1w2 = G2w1w2/(n1*n2)

        return g2w1w2
