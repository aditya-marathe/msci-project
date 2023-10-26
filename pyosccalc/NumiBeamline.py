# This is a class to approximate the simulation of the NuMI Beamline
# It is not meant to be a replacement of the full simulation but instead is useful (perhaps) for quick studies.
# Ryan Nichol <r.nichol@ucl.ac.uk>
# 27th July 2022

from math import sqrt
import numpy as np
import math
import scipy
from particle import PDGID, Particle
from hepunits.units import keV, MeV, GeV
from hepunits import units as u


#Probably all these functions should move into a class at some point
#Below code is transcribed by RJN from G4NuMI
inch=2.54*u.cm #cm->mm
fDeltaOuterThickness=0.9525*u.cm #cm->mm
def getHorn1OCRout(z):
    #Martens 3/26/10
    # The Horn 1 outer conductor is thinner for NOvA than for MINOS.
    # For MINOS the Horn 1 outer conductor is 1 inch thick.
    # For NOvA the Horn 1 outer conductor is 5/8 inch thick.
    # Thus subtract 3/8 inch from the Horn 1 outer conductor outer radius
    # but don't subract the 3/8 inch from the end flanges.
    
    #OC dimensions from drawings
    conds = [z < 0, 
             ((z>=0.*inch)&(z<0.756*inch)),
             ((z>=0.756*inch)&(z<1.756*inch)),
             ((z>=1.756*inch)&(z<2.756*inch)),
             ((z>=2.756*inch)&(z<115.971*inch)),
             ((z>=115.971*inch)&(z<117.341*inch)),
             ((z>=117.341*inch)&(z<123.311*inch)),
              ((z>=123.311*inch)&(z<124.811*inch)),
             ((z>=124.811*inch)&(z<=126.092*inch)),
            ((z>=126.096*inch)&(z<=130.*inch))]   #The N regions of z
    funcs = [lambda z: 0, # for mother vol.
             lambda z:  3.763+2.436,  # OC dimensions from drawings
             lambda z: 16.25/2.,
             lambda z: 15.99/2.,
             lambda z: 13.750/2. -fDeltaOuterThickness,
             lambda z :(6.875+(z/inch-115.971)/(117.341-115.971)*(8.25-6.875))-fDeltaOuterThickness,
             lambda z: 16.5/2.- fDeltaOuterThickness,
             lambda z: 17.5/2., #RJN change
             lambda z: 15.5/2.,
            lambda z: 15.5/2]  #the lambda keyword is allowing us to define a quick function
    return inch*np.piecewise(z, conds, funcs)  #Now do the piecewise calculation and return it
    
def getHorn1OCRin(z):
    conds = [z < 0,
             ((z>=0.*inch)&(z<1.806*inch)),
            ((z>=1.806*inch)&(z<116.551*inch)),
            ((z>=116.551*inch)&(z<117.341*inch)),
            ((z>=117.341*inch)&(z<122.351*inch)),
            ((z>=122.351*inch)&(z<124.051*inch)),
            ((z>=124.051*inch)&(z<=126.096*inch)),
            ((z>=126.096*inch)&(z<=130.*inch))]
    funcs = [lambda z: 0, # for mother vol
            lambda z: 5.879,
            lambda z: 11.75/2.,
            lambda z: (5.875+(z/inch-116.551)/(117.341-116.551)*(7.25-5.875)),
            lambda z: 14.5/2.,
             lambda z: (14.5/2.-(z/inch-122.351)/(124.051-122.351)*(7.25-6.)),
             lambda z: 6.,
             lambda z: 5.815
            ]
    return inch*np.piecewise(z, conds, funcs)  #Now do the piecewise calculation and return it
    

def getHorn1ICRout(z):
    conds = [((z>=0.*inch)&(z<3.32645*inch)),
             ((z>=3.32645*inch)&(z<30.3150*inch)),
             ((z>=30.3150*inch)&(z<31.8827*inch)),
             ((z>=31.8827*inch)&(z<117.1126*inch)),
             ((z>=117.1126*inch)&(z<=128.1096*inch)),
             ((z>=128.1096*inch)&(z<=129.3566*inch)),
             (z>129.3566*inch)
             ]
    funcs= [lambda z: np.sqrt(1.975805-(0.05585)*(z/inch)),
            lambda z: np.sqrt(1.975805-(0.05585)*(z/inch)),
            lambda z: 1.063/2.,
            lambda z: np.sqrt(0.180183*z/inch-5.462253),
            lambda z: 8.5/2.,
            lambda z: 11.623/2.,
            lambda z: 0.
           ]
    return inch*np.piecewise(z, conds, funcs)  #Now do the piecewise calculation and return it
  #if (r < 0.100*mm) r = 0.1*mm;

def getHorn1ICRin(z):
    conds = [(z<0.0*inch),
             ((z>=0.0*inch)&(z<16.1602*inch)),
             ((z>=16.1602*inch)&(z<30.3150*inch)),
             ((z>=30.3150*inch)&(z<31.8827*inch)),
             ((z>=31.8827*inch)&(z<36.2709*inch)),
             ((z>=36.2709*inch)&(z<117.1126*inch)),
             ((z>=117.1126*inch)&(z<=129.3566*inch)),
             (z>129.3566*inch)
             ]
    funcs= [lambda z: 0,
            lambda z: np.sqrt(1.975805-(0.055858)*(z/inch))-0.078740,
            lambda z: np.sqrt(1.818869-(0.055858)*(z/inch)),
            lambda z: .709/2., #NECK!
            lambda z: np.sqrt(0.180183*(z/inch)-5.61919), #Note (z/inch) is the z coordinate in inches
            lambda z: np.sqrt(0.180183*z/inch-5.462253)-0.078740,
            lambda z: 7.75/2.,
            lambda z: 0.
           ]
    return inch*np.piecewise(z, conds, funcs)  #Now do the piecewise calculation and return it


def getSimpleHorn1ICRin(z):
    conds = [(z>0*u.m) & (z<=0.77*u.m),
             (z>0.77*u.m) & (z<=0.8098*u.m),
             (z>0.8098*u.m) & (z<=0.9213*u.m),
              (z>0.9213*u.m) & (z<=2.975*u.m),
              ((z>2.975*u.m) & (z<3.286*u.m))]
    funcs = [
        #lambda z: 0.01*np.sqrt((82.71-100*z/u.m)/7.048),
        lambda z: u.cm*np.sqrt((82.71-100*z/u.m)/7.2),
        lambda z: u.cm*0.90,
        lambda z: u.cm*np.sqrt((100*z/u.m-79.21)/2.185),
        lambda z: u.cm*(np.sqrt((100*z/u.m-77)/2.185)-0.2),
        lambda z: u.cm*(np.sqrt((100*2.975-77)/2.185)-0.2)]
    
    return np.piecewise(z, conds, funcs)  #Now do the piecewise calculation and return it

def getSimpleHorn1ICRout(z):
    conds = [(z>0*u.m) & (z<=0.77*u.m),
             (z>0.77*u.m) & (z<=0.8098*u.m),
             (z>0.8098*u.m) & (z<=2.975*u.m),
              ((z>2.975*u.m) & (z<3.286*u.m))]
    funcs = [
        #lambda z: u.cm**np.sqrt((82.71-100*z)/7.048),
        lambda z: u.cm*np.sqrt((89.85-100*z/u.m)/7.048),
        lambda z: u.cm*1.35,
        lambda z: u.cm*np.sqrt((100*(z/u.m)-77)/2.185),
        lambda z: u.cm*(np.sqrt((100*2.975-77)/2.185))]
    
    return np.piecewise(z, conds, funcs)  #Now do the piecewise calculation and return it
        
def getSimpleHorn1OCRout(z):
    conds = [((z>0*u.m) & (z<3.286*u.m))]
    funcs = [
        lambda z: 0.01*17.41*u.m]
    
    return np.piecewise(z, conds, funcs)  #Now do the piecewise calculation and return it     

def getSimpleHorn1OCRin(z):
    conds = [((z>0) & (z<3.286*u.m))]
    funcs = [
        lambda z: 0.01*14.92*u.m]
    
    return np.piecewise(z, conds, funcs)  #Now do the piecewise calculation and return it     

horn2Start=19.18*u.m

def getSimpleHorn2ICRin(z):
    x=z-horn2Start
    conds = [((x>0) & (x<97.617*u.cm)),
            ((x>97.617*u.cm) & (x<104.803*u.cm)),
            ((x>104.803*u.cm) & (x<300*u.cm))]
    funcs = [
        lambda z: u.cm*(np.sqrt((100-(z-horn2Start)/u.cm)/0.1351) - 0.3),
        lambda z: u.cm*3.9,
        lambda z: u.cm*(np.sqrt(((z-horn2Start)/u.cm-100)/0.2723) - 0.3)
    ]
    return np.piecewise(z, conds, funcs)


def getSimpleHorn2ICRout(z):
    x=z-horn2Start
    conds = [((x>0) & (x<97.617*u.cm)),
            ((x>97.617*u.cm) & (x<104.803*u.cm)),
            ((x>104.803*u.cm) & (x<300*u.cm))]
    funcs = [
        lambda z: u.cm*np.sqrt((100-(z-horn2Start)/u.cm)/0.1351),
        lambda z: u.cm*4.2,
        lambda z: u.cm*np.sqrt(((z-horn2Start)/u.cm-100)/0.2723)
    ]
    return np.piecewise(z, conds, funcs)
    
def getSimpleHorn2OCRout(z):
    x=z-horn2Start
    conds = [((x>0*u.m) & (x<3*u.m))]
    funcs = [
        lambda z: 39.54*u.cm]
    return np.piecewise(z, conds, funcs)  #Now do the piecewise calculation and return it     

def getSimpleHorn2OCRin(z):
    x=z-horn2Start
    conds = [((x>0*u.m) & (x<3*u.m))]
    funcs = [
        lambda z: 37*u.cm]
    return np.piecewise(z, conds, funcs)  #Now do the piecewise calculation and return it     

#Now we ant to turn this in to something we can use in a more modular fashion.
class NumiBeamline:
    """A simple class to propagate particles through the NuMI beamline"""
    
    def __init__(self,isFHC=True):
        self.horn1Start=0*u.m
        self.horn2Start=19.18*u.m
        self.numSteps=100
        self.zHorn1=np.linspace(self.horn1Start,self.horn1Start+3*u.m,self.numSteps)  #All these distances are now in mm
        self.horn1IC=getSimpleHorn1ICRout(self.zHorn1) #All these distances are now in mm
        #print(self.zHorn1)
        #print(self.horn1IC)
        self.horn1OC=getSimpleHorn1OCRin(self.zHorn1) #All these distances are now in mm
        self.zHorn2=np.linspace(self.horn2Start,self.horn2Start+3*u.m,self.numSteps)
        self.horn2IC=getSimpleHorn2ICRout(self.zHorn2) #All these distances are now in mm
        self.horn2OC=getSimpleHorn2OCRin(self.zHorn2) #All these distances are now in mm
        self.isFHC=isFHC #If false need to flip sign of K
        self.K=0.3*scipy.constants.mu_0*200e3/(2*math.pi) #Curvature constant dpt = K * l(m) / r(m) need to verify
        if not self.isFHC:
            K = -K
        self.DecayPipeZ0= 45.699*u.m; #was 45.28*m (08/09/05);
        self.DecayPipeRadius= 0.9906*u.m;
        self.DecayPipeLength= 676.681*u.m; #was 677.1*m (08/09/05);
    
    #def targetToHorn1(self,particle,tgtZ,pT,xr):
    def targetToHorn1(self,particle,z,r,pT,pLab):
        """
        Propagates particles from the point they leave the target (assumed to be at r=0 for now)
        to the start of NuMI Horn 1
        Arguments:
            particle: an integer
            b: an integer
        Returns:
            The sum of the two integer arguments
        """
        #xr stuff moved outside
        #print("xr",xr)
        #pLab=getPlabFromxr(xr,pT,particle)
        pL=np.sqrt(pLab**2 - pT**2)
        #print("p,p,p",pLab,pT,pL)
        tanTheta=pT/pL
        #print(tanTheta)
        rHorn=r+(self.zHorn1[0]-z)*tanTheta
        #print("rHorn",rHorn)
        #print(cosTheta)
        return particle,self.zHorn1[0],rHorn,pT,pLab

    def throughHorn1(self,particle,z,r,pT,pLab):
        #z must equal 0 but should check
        #Should also add some material energy loss maybe
        r1=r
        pt1=pT
        #print("r1",r1)
        #print("pt1",pt1)
        dz=self.zHorn1[1]-self.zHorn1[0] #Perhaps move into loop??
        #print("dz",dz)
        for i in range(1,self.numSteps):
            r0=r1
            pt0=pt1
            sinTheta=pt0/pLab
            cosTheta=np.sqrt(1-sinTheta**2)
            tanTheta=pt0/np.sqrt(pLab**2 - pt0**2)
            r1=r0+dz*tanTheta
            #print(i,"th1",r1,self.horn1IC[i],self.horn1OC[i])
            pt1=pt0
            #print("r1",i,r1)
            #print(cosTheta)
            a=np.array(r1>self.horn1IC[i],dtype=bool)
            #print("i","th1",a)
            b=r1<self.horn1OC[i]
            #print("i","th1",b)
            mask = np.logical_and(a, b)
            #print(i,"th1",mask)
            #print("pt0[mask]",pt0[mask])
            pt1[mask]=pt0[mask] - particle.charge*GeV*(self.K * dz) / (r1*cosTheta)[mask]
            #print("pt1",i,pt1)
        return particle,self.zHorn1[-1],r1,pt1,pLab
        
    def throughHorn1Original(self,particle,z,r,pT,pLab):
        #z must equal zHorn1[0] but should check
        #Should also add some material energy loss maybe
        r1=np.zeros(self.numSteps)  #Should think about how to do this
        pt1=np.zeros(self.numSteps) #Should think about how to do this
        r1[0]=r
        pt1[0]=pT
        dz=self.zHorn1[1]-self.zHorn1[0] #Perhaps move into loop??
        for i in range(1,self.numSteps):
            #print(i,r1[i-1])
            sinTheta=pt1[i-1]/pLab
            cosTheta=np.sqrt(1-sinTheta**2)
            tanTheta=pt1[i-1]/np.sqrt(pLab**2 - pt1[i-1]**2)
            r1[i]=r1[i-1]+dz*tanTheta
            if(r1[i-1]>self.horn1IC[i-1] and r1[i-1]<self.horn1OC[i-1]):
                pt1[i]=pt1[i-1] - particle.charge*GeV*(self.K * dz) / (r1[i-1]*cosTheta)
            else:
                pt1[i]=pt1[i-1]
            
        return particle,self.zHorn1[-1],r1[-1],pt1[-1],pLab
        
    def horn1ToHorn2(self,particle,z,r,pT,pLab):
        """
        Propagates particles from the point they leave the target (assumed to be at r=0 for now)
        to the start of NuMI Horn 1
        Arguments:
            particle: an integer
            b: an integer
        Returns:
            The sum of the two integer arguments
        """
        pL=np.sqrt(pLab**2 - pT**2)
        tanTheta=pT/pL
        rHorn2=r + tanTheta*(self.zHorn2[0]-self.zHorn1[-1])
        return particle,self.zHorn2[0],rHorn2,pT,pLab
    
    def horn2ToDecayPipe(self,particle,z,r,pT,pLab):
        """
        Propagates particles from the point they leave the target (assumed to be at r=0 for now)
        to the start of NuMI Horn 1
        Arguments:
            particle: an integer
            b: an integer
        Returns:
            The sum of the two integer arguments
        """
        pL=np.sqrt(pLab**2 - pT**2)
        tanTheta=pT/pL
        rHorn2=r + tanTheta*(self.DecayPipeZ0-self.zHorn2[-1])
        return particle,self.DecayPipeZ0,rHorn2,pT,pLab
    
    def horn1ToHorn2(self,particle,z,r,pT,pLab):
        """
        Propagates particles from the point they leave the target (assumed to be at r=0 for now)
        to the start of NuMI Horn 1
        Arguments:
            particle: an integer
            b: an integer
        Returns:
            The sum of the two integer arguments
        """
        pL=np.sqrt(pLab**2 - pT**2)
        tanTheta=pT/pL
        rHorn2=r + tanTheta*(self.zHorn2[0]-self.zHorn1[-1])
        return particle,self.zHorn2[0],rHorn2,pT,pLab
    
    def throughHorn2(self,particle,z,r,pT,pLab):
        #z must equal 0 but should check
        #Should also add some material energy loss maybe
        r1=r
        pt1=pT
        #print("r1",r1)
        #print("pt1",pt1)
        dz=self.zHorn2[1]-self.zHorn2[0] #Perhaps move into loop??
        #print("dz",dz)
        for i in range(1,self.numSteps):
            r0=r1
            pt0=pt1
            sinTheta=pt0/pLab
            cosTheta=np.sqrt(1-sinTheta**2)
            tanTheta=pt0/np.sqrt(pLab**2 - pt0**2)
            r1=r0+dz*tanTheta
            #print(i,r1,self.horn1IC[i],self.horn1OC[i])
            #print(r1>self.horn2IC[i])
            pt1=pt0
            #print("r1",i,r1)
            #print(cosTheta)
            a=np.array(r1>self.horn2IC[i],dtype=bool)
            b=r1<self.horn2OC[i]
            #print("a",a)
            #print("b",b)
            mask = np.logical_and(a, b)
            #print(mask)
            #print("pt0[mask]",pt0[mask])
            pt1[mask]=pt0[mask] - particle.charge*GeV*(self.K * dz) / (r1*cosTheta)[mask]
            #print("pt1",i,pt1)
        return particle,self.zHorn2[-1],r1,pt1,pLab
    
    def throughHorn2Original(self,particle,z,r,pT,pLab):
        #z must equal zHorn2[0] but should check
        #Should also add some material energy loss maybe
        r1=np.zeros(self.numSteps)  #Should think about how to do this
        pt1=np.zeros(self.numSteps) #Should think about how to do this
        r1[0]=r
        pt1[0]=pT
        dz=self.zHorn2[1]-self.zHorn2[0] #Perhaps move into loop??
        for i in range(1,self.numSteps):
            sinTheta=pt1[i-1]/pLab
            cosTheta=np.sqrt(1-sinTheta**2)
            tanTheta=pt1[i-1]/np.sqrt(pLab**2 - pt1[i-1]**2)
            r1[i]=r1[i-1]+dz*tanTheta
            if(r1[i-1]>self.horn2IC[i-1] and r1[i-1]<self.horn2OC[i-1]):
                pt1[i]=pt1[i-1] - particle.charge*GeV*(self.K * dz) / (r1[i-1]*cosTheta)
            else:
                pt1[i]=pt1[i-1]
            
        return particle,self.zHorn2[-1],r1[-1],pt1[-1],pLab

