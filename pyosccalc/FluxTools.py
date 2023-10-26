# This is a class to fake neutrino fluxes for neutrino oscillation experiments
# Ryan Nichol <r.nichol@ucl.ac.uk>
# 3rd June 2020

from scipy.stats import lognorm
from scipy.stats import lognorm
import numpy as np

class Flux:
    def __init__(self):
        self.name="Raw Flux"

    def pdf(self,x):
        return lognorm.pdf(x,0.3,0,2)

    def name(self):
        return self.name

class LogNormalFlux(Flux):
    #Here is the lognormal flux
    def __init__(self,shape=0.3,loc=0,scale=2,exposure=500):
        self.name="Lognormal Flux"
        self.shape=shape
        self.loc=loc
        self.scale=scale
        self.exposure=exposure

    def pdf(self,x):
        return lognorm.pdf(x,self.shape,self.loc,self.scale)

    def flux(self,x):
        return self.exposure*self.pdf(x)

    

class FluxTools: 
    def __init__(self):
        self.binEdges=[0,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5,3.75,4,4.25,4.5,4.75,5.0]
        self.xvals=np.linspace(0.25,5.25,501)
        self.lastnumuflux=None
        self.lastnumuosccalc=None
        self.lastnueflux=None
        self.lastnueosccalc=None


    def pdfwrap(self,flux):
        return flux.pdf(self.xvals)

    def unity(self,x):
        return 1.0

    def getAsimov(self,flux,probcalc):
        fluxVals=flux.flux(self.xvals)
        binVals=[0]*(len(self.binEdges)-1)
        thisBin=0
        lastx=0
        for x,y in zip(self.xvals,fluxVals):
            if(thisBin<len(self.binEdges)-2):
                if(x>self.binEdges[thisBin+1]):
                    thisBin+=1
            binVals[thisBin]+=y*(x-lastx)*probcalc(x)
            lastx=x
        return binVals

    def getNoOscAsimov(self,flux):
        return self.getAsimov(flux,self.unity)

    def getNuMuAsimov(self, flux, osccalc,force=False):
        if(flux!=self.lastnumuflux or hash(osccalc)!=self.lastnumuosccalc or force):
            binVals=self.getAsimov(flux,osccalc.MuToMu)
            self.lastnumuflux=flux
            self.lastnumuosccalc=hash(osccalc)
            self.lastnumuVals=binVals
            return binVals
        return self.lastnumuVals
    
    def genNuMuExperiment(self,flux,osccalc):
        binVals=self.getNuMuAsimov(flux,osccalc)
        return [np.random.poisson(x) for x in binVals]


    def getNuElecAsimov(self, flux, osccalc,force=False):
        if(flux!=self.lastnueflux or hash(osccalc)!=self.lastnueosccalc or force):
            binVals=self.getAsimov(flux,osccalc.MuToElec)
            self.lastnueflux=flux
            self.lastnueosccalc=hash(osccalc)
            self.lastnueVals=binVals
            return binVals
        return self.lastnueVals
    
    def genNuElecExperiment(self,flux,osccalc):
        binVals=self.getNuElecAsimov(flux,osccalc)
        return [np.random.poisson(x) for x in binVals]

    def makeNuMuAsimovArray(self,flux,osccalc,dm32Array,sinSq23Array,dcpArray):
        a = []
        for dcp in dcpArray:
            mat = []
            for sSq23 in sinSq23Array:
                row = []
                for dm32 in dm32Array:
                    osccalc.updateOscParams(sinSqTheta23=sSq23,deltamSq32=dm32,dcp=dcp)
                    val=self.getNuMuAsimov(flux,osccalc,force=True)
                    row.append(val)
                mat.append(row)
            a.append(mat)
        return a

