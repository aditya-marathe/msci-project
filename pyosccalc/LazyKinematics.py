# This is a set of functions to wrap up some kinematic calculations useful in simulating neutrino beams
# Ryan Nichol <r.nichol@ucl.ac.uk>
# 27th July 2022

from particle import Particle
import numpy as np
from hepunits.units import keV, MeV, GeV
from hepunits import units as u

class LazyKinematics:
    def getGammaFromBeta(beta):
        return 1/np.sqrt(1-beta**2)


class NeutrinoTarget:
    """A simple class to wrap-up a bunch of kinematic tools related to simulating """
    
    proton=Particle.from_pdgid(2212)
    carbonMass=12*931.49410242*MeV #MeV/c^2
    
    def __init__(self,useCarbon=True,protonP=120*GeV):
        self.protonP=protonP
        if useCarbon:
            self.targetMass=self.carbonMass
        else:
            self.targetMass=self.proton.mass
        self.sqrtS=self.getSqrtS()
        self.beta=self.getBeta()
        self.gamma=LazyKinematics.getGammaFromBeta(self.beta)

    #Switch to using MeV everywhere
    def getBeta(self):
        E=np.sqrt(self.protonP**2 + self.proton.mass**2)
        return self.protonP/(self.targetMass+E)

    def getSqrtSPP(self):  #This is the sqrt(s) for a proton-proton collision, not proton-carbon
        mp=self.proton.mass
        E=np.sqrt(self.protonP**2 + mp**2)
        return np.sqrt(2*mp*E + 2*mp**2)

    def getSqrtS(self):  #This is the sqrt(s) for a proton-proton collision, not proton-carbon
        mp=self.proton.mass
        E=np.sqrt(self.protonP**2 + mp**2)
        return np.sqrt(2*self.targetMass*E + mp**2 + self.targetMass**2)

    def getEStarMaxPP(self,particle): #Need to check this for carbon
        Mxbar=1.88*GeV
        if(particle == Particle.from_pdgid(211)):
            Mxbar=1.88*GeV
        if(particle == Particle.from_pdgid(-211)):
            Mxbar=2.02*GeV
        if(particle == Particle.from_pdgid(321)):
            Mxbar=2.05*GeV
        if(particle == Particle.from_pdgid(-321)):
            Mxbar=2.37*GeV
        Mxbar=0  #RJN hack
        return (self.sqrtS**2 - Mxbar**2 + particle.mass**2)/(2*self.sqrtS)

    def getXf(self,pL_lab,pT_lab,particle=Particle.from_pdgid(211)):
        p_lab=np.sqrt(pL_lab**2 + pT_lab**2)
        E_lab=np.sqrt(p_lab**2 + (particle.mass)**2)
        plStar=self.gamma*(pL_lab - self.beta*E_lab)
        return 2* plStar / self.sqrtS

    def getXr(self,pL_lab,pT_lab,particle=Particle.from_pdgid(211)):
        p_lab=np.sqrt(pL_lab**2 + pT_lab**2)
        E_lab=np.sqrt(p_lab**2 + (particle.mass)**2)
        Estar=self.gamma*(E_lab-self.beta*pL_lab)
        return Estar/self.getEStarMaxPP(particle)

    def getXlab(self,pL_lab,pT_lab,particle=Particle.from_pdgid(211)):
        return np.sqrt(pL_lab**2 + pT_lab**2)/(self.protonP)
    
    def getPstar(self,pL_lab,pt_lab,particle=Particle.from_pdgid(211)):
        p_lab=np.sqrt(pL_lab**2 + pt_lab**2)
        E_lab=np.sqrt(p_lab**2 + (particle.mass)**2)
        return -self.beta*self.gamma*E_lab + self.gamma*pL_lab
    
    def getPlabFromxLab(self,xLab,pT=0,particle=Particle.from_pdgid(211)):
        return xLab*self.protonP

    def getPlabFromxF(self,xF,pT,particle=Particle.from_pdgid(211)):
        plstar=xF*self.sqrtS/2 
        Estar=np.sqrt(plstar**2 + pT**2 + (particle.mass)**2)
        P_L=self.gamma*plstar + self.beta*self.gamma*Estar
        return np.sqrt(P_L**2 +pT**2)
    
    def getPlabFromxr(self,xR,pT,particle=Particle.from_pdgid(211)):
        Estar=xR*self.getEStarMaxPP(particle)
        #print("Estar",Estar)
        plstar=np.sqrt(Estar**2 -pT**2 -(particle.mass)**2)
        P_L=self.gamma*plstar + self.beta*self.gamma*Estar
        return np.sqrt(P_L**2 +pT**2)