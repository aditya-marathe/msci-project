# This is a set of functions to model the flux of hadrons from neutrino targets
# The parameterisations are taken from the BMPT paper for the CNGS Beryllium target, https://arxiv.org/abs/hep-ph/0101163
# Bonesini, M., Marchionni, A., Pietropaolo, F. et al. On particle production for high energy neutrino beams. Eur. Phys. J. C 20, 13–27 (2001). 
# https://doi.org/10.1007/s100520100656
# Ryan Nichol <r.nichol@ucl.ac.uk>
# 27th July 2022

import numpy as np
from particle import PDGID, Particle
from hepunits.units import keV, MeV, GeV
from hepunits import units as u

class BMPTFlux:
    
    def getAlpha(self,xF,pT):
        return (0.74-0.55*xF +0.26*xF**2 )*(0.98+0.21*(pT/GeV)**2 )

    def getScalingFactor(self,A1,A2,xF,pT):
        return np.power(A1/A2,self.getAlpha(xF,pT))

    def getPiKFlux(self,particle,xr,pt):
        A=62.3
        B=1.57
        alpha=3.45
        beta=0.517
        a=6.1
        b=0
        gamma=0.153
        delta=0.478
        r0=1.05
        r1=2.65
        negFac=r0*np.power(1+xr,r1)
        if particle.name[0] == 'K': #Probably a better way to check for Kaon-ness
            A=7.74
            B=0
            alpha=2.45
            beta=0.444
            a=5.04
            gamma=0.121
            delta=2*gamma
            r0=1.15
            r1=-3.17
            negFac=r0*np.power(1-xr,r1)
        aprime=a*np.power(xr,gamma)
        bprime=a**2/(2*np.power(xr,delta))
        value=A*np.power(1-xr,alpha)*(1+B*xr)*np.power(xr,-beta)*(1+aprime*(pt/GeV)+bprime*(pt/GeV)**2)*np.exp(-aprime*(pt/GeV))
        value=value*self.getScalingFactor(12,8,xr,pt) #Scaling factor is expecting pT in MeV
        if particle.charge > 0:
            return value
        if particle.charge < 0:
            return value/negFac
        if particle.charge == 0 and particle.name[0] == 'K':
            return 0.25*(1+3/negFac)*value