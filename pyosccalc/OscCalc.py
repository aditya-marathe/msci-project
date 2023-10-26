# This is a slightly sleazy steal of Josh Boehm's C++ OscCalc class
# Although all bugs are almost certainly mine
# Ryan Nichol 3rd June 2020
# The original C++ header is below.
##*********************************************************************
## Welcome the the OscCalc class - hopefully your one stop shop for 
##   computing three flavor neutrino oscillations.
##
## This code works under the assumption that you want a quick solution
##  and so it stores and holds values for the sins and cosines, and 
##  other temporary values.  If you want a high precision answer you should find 
##  Mark Messier's code which was living in AtNu last I looked.
##  This code combines expansions in alpha and sinth13 to produce a pretty robust 
##  and rapid calculation.  For an evaluation of the difference between these 
##  formula and the exact solutions please see Appendix A in my thesis.  You will 
##  also find a derivation for these formula there - Josh Boehm 04-27-2009
## **************************************************************************

import math

class OscCalc:
    #Numbers pulled from 2006 PDG pg 97
    Z_A = 0.5 #average Z/self.A
    N_A = 6.0221415e23 #avogadro's number
    INV_CM_TO_EV = 1.97326968e-5 #convert 1/cm to eV
    INV_CM_TO_GEV = 1.97326968e-14 #convert 1/cm to GeV
    INV_KM_TO_EV = 1.97326968e-10 #convert 1/km to eV
    GF = 1.166371e-5 #Fermi constant (GeV^{-2})
    SQRT2 = math.sqrt(2) #Sqrt 2 ... why??

    def __init__(self,sinSqTheta12=0.307, sinSqTheta13=0.0218, sinSqTheta23=0.536,deltamSq21=7.53e-5,deltamSq32=2.444e-3,dcp=1.37*math.pi,density=2.7,L=811,isAntiNu=1):
        self.sinSqTheta12 = sinSqTheta12
        self.sinSqTheta13 = sinSqTheta13
        self.sinSqTheta23 = sinSqTheta23 
        self.deltamSq21 = deltamSq21 #Units??
        self.deltamSq32 = deltamSq32 #Units??
        self.deltamsq13 = deltamSq21+deltamSq32 #So this isn't necessarily correctly named?
        self.dcp = dcp
        self.density = density #Units??
        self.L = L  #Units??
        self.isAntiNu = isAntiNu #-1 for antineutrino, +1 for neutrino
        ne=self.Z_A*self.N_A*self.density
        self.elecDensity=ne*self.INV_CM_TO_EV*self.INV_CM_TO_GEV*self.INV_CM_TO_GEV
        self.fV=self.SQRT2*self.GF*self.elecDensity
        self.calcSinCos()

        #Some variables for speeding up calculations (hopefully)
        self.Delta=0
        self.C13=0

    def updateOscParams(self,**kwargs):
        #print(kwargs)
        for k in kwargs.keys():
            if k in ['sinSqTheta12','sinSqTheta13','sinSqTheta23','deltamSq21','deltamSq32','dcp','density','L','isAntiNu']:
                self.__setattr__(k, kwargs[k])
        self.calcSinCos()
        self.deltamsq13 = self.deltamSq21+self.deltamSq32 #So this isn't necessarily correctly named?
    
    def __hash__(self):
        return hash((self.sinSqTheta12,self.sinSqTheta13,self.sinSqTheta23,self.deltamSq21,self.deltamSq32,self.dcp,self.density,self.L,self.isAntiNu))

    
    def calcSinCos(self):
        self.sin12=math.sqrt(self.sinSqTheta12)
        self.cosSqTheta12=1-self.sinSqTheta12
        self.cos12=math.sqrt(self.cosSqTheta12)
        self.sin212=2*self.sin12*self.cos12
        self.cos212=self.cosSqTheta12-self.sinSqTheta12
        self.sinSq2Theta12=self.sin212*self.sin212
        self.sin23=math.sqrt(self.sinSqTheta23)
        self.cosSqTheta23=1-self.sinSqTheta23
        self.cos23=math.sqrt(self.cosSqTheta23)
        self.sin223=2*self.sin23*self.cos23
        self.cos223=self.cosSqTheta23-self.sinSqTheta23
        self.sinSq2Theta23=self.sin223*self.sin223
        self.sin13=math.sqrt(self.sinSqTheta13)
        self.cosSqTheta13=1-self.sinSqTheta13
        self.cos13=math.sqrt(self.cosSqTheta13)
        self.sin213=2*self.sin13*self.cos13
        self.cos213=self.cosSqTheta13-self.sinSqTheta13
        self.sinSq2Theta13=self.sin213*self.sin213

        self.sin_dcp = math.sin(self.dcp)
        self.cos_dcp = math.cos(self.dcp)

    def buildTerms(self,E):
        #Building the more complicated terms
        Delta = self.deltamsq13*self.L/(4*E*1e9*self.INV_KM_TO_EV)
        newDelta=False
        if math.fabs(Delta-self.Delta)>1e-10: #Arbitrary cut off for now
            newDelta=True
            self.Delta=Delta
            self.A = 2*self.fV*E*1e9/(self.deltamsq13)
            self.alpha = self.deltamSq21/self.deltamsq13

        # A and d_cp both change sign for antineutrinos
        plusminus = int(self.isAntiNu)
        self.APrime = self.A*plusminus
        self.d_cp = self.dcp*plusminus
        self.sin__dcp = self.sin_dcp* plusminus

        #Now calculate the resonance terms for alpha expansion (self.C13) and s13 expansion (C12)
        C13 = math.sqrt(self.sinSq2Theta13+(self.APrime-self.cos213)*(self.APrime-self.cos213))
        newC13=False
        if math.fabs(C13-self.C13)>1e-10:
            newC13=True
            self.C13=C13
        self.C12 = 1  #really C12 -> infinity when alpha = 0 but not an option really
        if(math.fabs(self.alpha) > 1e-10):  #want to be careful here
            temp = self.cos212 - self.APrime/self.alpha
            self.C12 = math.sqrt(self.sinSq2Theta12+(temp*temp))


        #More complicated sin and cosine terms
        if newC13 or newDelta:
            self.cosC13Delta = math.cos(self.C13*self.Delta)
            self.sinC13Delta = math.sin(self.C13*self.Delta)
  
            self.sin1pADelta = math.sin((self.APrime+1)*self.Delta)
            self.cos1pADelta = math.cos((self.APrime+1)*self.Delta)

            self.sinADelta = math.sin(self.APrime*self.Delta)
            self.sinAm1Delta = math.sin((self.APrime-1)*self.Delta)
            self.cosAm1Delta = math.cos((self.APrime-1)*self.Delta)
            self.sinApam2Delta = math.sin((self.APrime+self.alpha-2)*self.Delta)
            self.cosApam2Delta = math.cos((self.APrime+self.alpha-2)*self.Delta)
            self.sin1pAmCDelta = math.sin(0.5*(self.A+1-self.C13)*self.Delta)
            self.sin1pApCDelta = math.sin(0.5*(self.A+1+self.C13)*self.Delta)
            self.sinDelta = math.sin(self.Delta)
            self.sin2Delta = math.sin(2*self.Delta)
        self.cosdpDelta = math.cos(self.dcp+self.Delta)

        self.cosaC12Delta = 0
        self.sinaC12Delta = 0 
  
        if(math.fabs(self.alpha) > 1e-10):
            self.cosaC12Delta = math.cos(self.alpha*self.C12*self.Delta)
            self.sinaC12Delta = math.sin(self.alpha*self.C12*self.Delta)
            self.cosaC12pApam2Delta = math.cos((self.alpha*self.C12+self.A+self.alpha-2)*self.Delta)

      

    def MuToElec(self,E):
        sinsq_2th12 = self.sinSq2Theta12
        sinsq_2th13 = self.sinSq2Theta13                                                                   
        cos_th23 = self.cos23
        cos_th12 = self.cos12
        sin_th13 = self.sin13
        cos_th13 = self.cos13  
                                                                              
        sin_2th23 = self.sin223
        sin_2th12 = self.sin212
        cos_2th13 = self.cos213
        cos_2th12 = self.cos212

        sinsq_th23 = self.sinSqTheta23
        sinsq_th12 = self.sinSqTheta12

        self.buildTerms(E)
       
        #First we calculate the terms for the alpha expansion (good to all orders in th13)
        # this is the equivalent of Eq 47 & 48 corrected for Mu to E instead of E to Mu

        # Leading order term 
        p1 = sinsq_th23*sinsq_2th13*self.sinC13Delta*self.sinC13Delta/(self.C13*self.C13)

        # terms that appear at order alpha
        #first work out the vacuum case since we get 0/0 otherwise.......
        p2Inner = self.Delta*self.cosC13Delta

        if(math.fabs(self.APrime) > 1e-9):
            p2Inner = self.Delta*self.cosC13Delta*(1-self.APrime*cos_2th13)/self.C13 -self.APrime*self.sinC13Delta*(cos_2th13-self.APrime)/(self.C13*self.C13)
        p2 = -2*sinsq_th12*sinsq_th23*sinsq_2th13*self.sinC13Delta/(self.C13*self.C13)*p2Inner*self.alpha


        #again working out vacuum first.....
        p3Inner = self.Delta* cos_th13* cos_th13*(-2*self.sin__dcp*self.sinC13Delta*self.sinC13Delta+2*self.cos_dcp*self.sinC13Delta*self.cosC13Delta)

        if(math.fabs(self.APrime) > 1e-9):
            p3Inner = (self.sinC13Delta/(self.APrime*self.C13*self.C13))*(- self.sin__dcp*(self.cosC13Delta - self.cos1pADelta)*self.C13+ self.cos_dcp*(self.C13*self.sin1pADelta - (1-self.APrime*cos_2th13)*self.sinC13Delta))
            p3 = sin_2th12*sin_2th23*sin_th13*p3Inner*self.alpha

        #  p1 + p2 + p3 is the complete contribution for this expansion
        # Now for the expansion in orders of math.sin(th13) (good to all order alpha) 
        #  this is the equivalent of Eq 65 and 66 

        # leading order term
        pa1 = 0.0
        pa2 = 0.0

        # no problems here when A -> 0
        if(math.fabs(self.alpha) > 1e-10):
            # leading order term
            pa1 = cos_th23*cos_th23*sinsq_2th12*self.sinaC12Delta*self.sinaC12Delta/(self.C12*self.C12)

            # and now to calculate the first order in s13 term
            t1 = (cos_2th12 - self.APrime/self.alpha)/self.C12 - self.alpha*self.APrime*self.C12*sinsq_2th12/(2*(1-self.alpha)*self.C12*self.C12)
            t2 = -self.cos_dcp*(self.sinApam2Delta-self.sinaC12Delta*t1)
            t3 = -(self.cosaC12Delta-self.cosApam2Delta)*self.sin__dcp
 
            denom = (1-self.APrime-self.alpha+self.APrime*self.alpha*cos_th12*cos_th12)*self.C12
            t4 = sin_2th12*sin_2th23*(1-self.alpha)*self.sinaC12Delta/denom

            pa2 = t4*(t3+t2)*sin_th13
  
        #pa1+pa2 is the complete contribution from this expansion

        # In order to combine the information correctly we need to add the two
        #  expansions and subtract off the terms that are in both (alpha^1, s13^1) 
        #  these may be taken from the expansion to second order in both parameters
        #  Equation 31 

        t1 = self.Delta*self.sinC13Delta*self.cosdpDelta
        if(math.fabs(self.APrime) > 1e-9): 
            t1 = self.sinADelta*self.cosdpDelta*self.sinAm1Delta/(self.APrime*(self.APrime-1))

        repeated = 2*self.alpha*sin_2th12*sin_2th23*sin_th13*t1

        #  Calculate the total probability
        totalP = p1+p2+p3 + (pa1+pa2) - repeated
        return totalP


    def MuToTau(self,E):
        sinsq_2th12 = self.sinSq2Theta12
        sinsq_2th13 = self.sin213*self.sin213   
        sinsq_2th23 = self.sin223*self.sin223                                                                        
        cos_th23 = self.cos23
        cos_th12 = self.cos12
        sin_th12 = self.sin12
        sin_th13 = self.sin13
        cos_th13 = self.cos13  
                                                                              
        sin_2th23 = self.sin223
        sin_2th12 = self.sin212
        cos_2th13 = self.cos213
        cos_2th23 = self.cos223
        cos_2th12 = self.cos212

        sinsq_th23 = self.sin23*self.sin23
        sinsq_th12 = self.sinSqTheta12
        #Building the more complicated terms  
        self.buildTerms(E)     

        #First we calculate the terms for the alpha expansion (good to all orders in th13)
        # this is the equivalent of Eq 49 & 50 corrected for Mu to E instead of E to Mu

        # Leading order term
        pmt_0 = 0.5*sinsq_2th23
        pmt_0 *= (1 - (cos_2th13-self.APrime)/self.C13)*self.sin1pAmCDelta*self.sin1pAmCDelta +  (1 + (cos_2th13-self.APrime)/self.C13)*self.sin1pApCDelta*self.sin1pApCDelta - 0.5*sinsq_2th13*self.sinC13Delta*self.sinC13Delta/(self.C13*self.C13)

        # terms that appear at order alpha
        t0 = (cos_th12*cos_th12-sin_th12*sin_th12*sin_th13*sin_th13*(1+2*sin_th13*sin_th13*self.APrime+self.APrime*self.APrime)/(self.C13*self.C13))*self.cosC13Delta*self.sin1pADelta*2
        t1 = 2*(cos_th12*cos_th12*cos_th13*cos_th13-cos_th12*cos_th12*sin_th13*sin_th13+sin_th12*sin_th12*sin_th13*sin_th13+(sin_th12*sin_th12*sin_th13*sin_th13-cos_th12*cos_th12)*self.APrime)
        t1 *= self.sinC13Delta*self.cos1pADelta/self.C13

        t2 =  sin_th12*sin_th12*sinsq_2th13*self.sinC13Delta/(self.C13*self.C13*self.C13)
        t2 *= self.APrime/self.Delta*self.sin1pADelta+self.APrime/self.Delta*(cos_2th13-self.APrime)/self.C13*self.sinC13Delta- (1-self.APrime*cos_2th13)*self.cosC13Delta

        pmt_1 = -0.5*sinsq_2th23*self.Delta*(t0+t1+t2)   

        t0 = t1 = t2 = t3 = 0.0

        t0 = self.cosC13Delta-self.cos1pADelta
        t1 = 2*cos_th13*cos_th13*self.sin__dcp*self.sinC13Delta/self.C13*t0
        t2 = -cos_2th23*self.cos_dcp*(1+self.APrime)*t0*t0

        t3  = cos_2th23*self.cos_dcp*(self.sin1pADelta+(cos_2th13-self.APrime)/self.C13*self.sinC13Delta)
        t3 *= (1+2*sin_th13*sin_th13*self.APrime + self.APrime*self.APrime)*self.sinC13Delta/self.C13 - (1+self.APrime)*self.sin1pADelta


        if(math.fabs(self.APrime) > 1e-9): 
            pmt_1 = pmt_1 + (t1+t2+t3)*sin_th13*sin_2th12*sin_2th23/(2*self.APrime*cos_th13*cos_th13)
        else:
            pmt_1 = pmt_1 + sin_th13*sin_2th12*sin_2th23*cos_th13*cos_th13*self.Delta*(2*self.sin__dcp*self.sinC13Delta*self.sinC13Delta+self.cos_dcp*cos_2th23*2*self.sinC13Delta*self.cosC13Delta)

        pmt_1 *= self.alpha

        #  pmt_0 + pmt_1 is the complete contribution for this expansion
                                                                                                                       
        # Now for the expansion in orders of math.sin(th13) (good to all order alpha)
        #  this is the equivalent of Eq 67 and 68
                                                                                                                       
        # leading order term
        pmt_a0 =  0.5*sinsq_2th23

        pmt_a0 *= 1 - 0.5*sinsq_2th12*self.sinaC12Delta*self.sinaC12Delta/(self.C12*self.C12)- self.cosaC12pApam2Delta- (1 - (cos_2th12 - self.APrime/self.alpha)/self.C12)*self.sinaC12Delta*self.sinApam2Delta    
        denom = (1-self.APrime-self.alpha+self.APrime*self.alpha*cos_th12*cos_th12)*self.C12

        t0 = (self.cosaC12Delta-self.cosApam2Delta)*(self.cosaC12Delta-self.cosApam2Delta)          
        t1 = (cos_2th12 - self.APrime/self.alpha)/self.C12*self.sinaC12Delta+self.sinApam2Delta          
        t2 = ((cos_2th12 - self.APrime/self.alpha)/self.C12+2*(1-self.alpha)/(self.alpha*self.APrime*self.C12))*self.sinaC12Delta+ self.sinApam2Delta
        t3 = (self.alpha*self.APrime*self.C12)/2.0*cos_2th23*self.cos_dcp*(t0 + t1*t2)
        t3 += self.sin__dcp*(1-self.alpha)*(self.cosaC12Delta-self.cosApam2Delta)*self.sinaC12Delta
        pmt_a1 = sin_th13*sin_2th12*sin_2th23/denom*t3

        # pmt_a1+pmt_a2 is the complete contribution from this expansion
                                                                                                                       
        # In order to combine the information correctly we need to add the two
        #  expansions and subtract off the terms that are in both (alpha^1, s13^1)
        #  and lower order terms
        #  these may be taken from the expansion to second order in both parameters
        #  Equation 34


        # Now for the term of order alpha * s13 or lower order!
        t0 = t1 = t2 = t3 = 0.0

        t1 = +self.sin__dcp*self.sinDelta*self.sinADelta*self.sinAm1Delta/(self.APrime*(self.APrime-1))
        t2 = -1/(self.APrime-1)*self.cos_dcp*self.sinDelta*(self.APrime*self.sinDelta-self.sinADelta*self.cosAm1Delta/self.A)*cos_2th23
        t0 =  2*self.alpha*sin_2th12*sin_2th23*sin_th13*(t1+t2)

        t1 = sinsq_2th23*self.sinDelta*self.sinDelta - self.alpha*sinsq_2th23*cos_th12*cos_th12*self.Delta*self.sin2Delta

        repeated = t0+t1
        #  Calculate the total probability
        totalP = pmt_0 + pmt_1 + pmt_a0 + pmt_a1 - repeated
        return totalP


    def MuToMu(self,E):
        sinsq_2th12 = self.sinSq2Theta12
        sinsq_2th13 = self.sin213*self.sin213
        sinsq_2th23 = self.sin223*self.sin223
        #  std::cout << E << "\t" << L  << "\t" << self.deltamSq32 << "\t" << self.deltamSq21 << "\t" << sinsq_2th23 << "\t" << sinsq_2th13 << "\n" 
                                          
        cos_th23 = self.cos23                                                                            
        cos_th13 = self.cos13
        cos_th12 = self.cos12
        sin_th12 = self.sin12
        sin_th13 = self.sin13

        #  std::cout << "th23: " <<  cos_th23 << "\t" << sinsq_2th23 << "\n"
        sin_2th23 = self.sin223
        sin_2th12 = self.sin212                                                                     
        cos_2th23 = self.cos223
        cos_2th13 = self.cos213
        cos_2th12 = self.cos212


        sinsq_th23 = self.sin23*self.sin23
        sinsq_th12 = self.sinSqTheta12

        #Building the more complicated terms                                                                              
        self.buildTerms(E)
        
    

        #This bit is the mu-to-tau part
        #First we calculate the terms for the alpha expansion (good to all orders in th13)
        # this is the equivalent of Eq 49 & 50 corrected for Mu to E instead of E to Mu  
        pMuToTau=0
  
        # Leading order term
        pmt_0 = 0.5*sinsq_2th23
        pmt_0 *= (1 - (cos_2th13-self.APrime)/self.C13)*self.sin1pAmCDelta*self.sin1pAmCDelta +  (1 + (cos_2th13-self.APrime)/self.C13)*self.sin1pApCDelta*self.sin1pApCDelta- 0.5*sinsq_2th13*self.sinC13Delta*self.sinC13Delta/(self.C13*self.C13)

        #    if(E>1.6 && E<1.606)
    
        # terms that appear at order alpha
        t0 = (cos_th12*cos_th12-sin_th12*sin_th12*sin_th13*sin_th13*(1+2*sin_th13*sin_th13*self.APrime+self.APrime*self.APrime)/(self.C13*self.C13))*self.cosC13Delta*self.sin1pADelta*2
        t1 = 2*(cos_th12*cos_th12*cos_th13*cos_th13-cos_th12*cos_th12*sin_th13*sin_th13+sin_th12*sin_th12*sin_th13*sin_th13+(sin_th12*sin_th12*sin_th13*sin_th13-cos_th12*cos_th12)*self.APrime)
        t1 *= self.sinC13Delta*self.cos1pADelta/self.C13

        t2 =  sin_th12*sin_th12*sinsq_2th13*self.sinC13Delta/(self.C13*self.C13*self.C13)
        t2 *= self.APrime/self.Delta*self.sin1pADelta+self.APrime/self.Delta*(cos_2th13-self.APrime)/self.C13*self.sinC13Delta- (1-self.APrime*cos_2th13)*self.cosC13Delta

        pmt_1 = -0.5*sinsq_2th23*self.Delta*(t0+t1+t2)   

        t0 = t1 = t2 = t3 = 0.0

        t0 = self.cosC13Delta-self.cos1pADelta
        t1 = 2*cos_th13*cos_th13*self.sin__dcp*self.sinC13Delta/self.C13*t0
        t2 = -cos_2th23*self.cos_dcp*(1+self.APrime)*t0*t0

        t3  = cos_2th23*self.cos_dcp*(self.sin1pADelta+(cos_2th13-self.APrime)/self.C13*self.sinC13Delta)
        t3 *= (1+2*sin_th13*sin_th13*self.APrime + self.APrime*self.APrime)*self.sinC13Delta/self.C13 - (1+self.APrime)*self.sin1pADelta


        if(math.fabs(self.APrime) > 1e-9): 
            pmt_1 = pmt_1 + (t1+t2+t3)*sin_th13*sin_2th12*sin_2th23/(2*self.APrime*cos_th13*cos_th13)
        else:
            pmt_1 = pmt_1 + sin_th13*sin_2th12*sin_2th23*cos_th13*cos_th13*self.Delta*(2*self.sin__dcp*self.sinC13Delta*self.sinC13Delta+self.cos_dcp*cos_2th23*2*self.sinC13Delta*self.cosC13Delta)

        pmt_1 *= self.alpha

        #  pmt_0 + pmt_1 is the complete contribution for this expansion
                                                                                                                       
        # Now for the expansion in orders of math.sin(th13) (good to all order alpha)
        #  this is the equivalent of Eq 67 and 68
                                                                                                                       
        # leading order term
        pmt_a0 =  0.5*sinsq_2th23

        pmt_a0 *= 1 - 0.5*sinsq_2th12*self.sinaC12Delta*self.sinaC12Delta/(self.C12*self.C12)- self.cosaC12pApam2Delta- (1 - (cos_2th12 - self.APrime/self.alpha)/self.C12)*self.sinaC12Delta*self.sinApam2Delta
            
        denom = (1-self.APrime-self.alpha+self.APrime*self.alpha*cos_th12*cos_th12)*self.C12

        t0 = (self.cosaC12Delta-self.cosApam2Delta)*(self.cosaC12Delta-self.cosApam2Delta)         
        t1 = (cos_2th12 - self.APrime/self.alpha)/self.C12*self.sinaC12Delta+self.sinApam2Delta   
        t2 = ((cos_2th12 - self.APrime/self.alpha)/self.C12+2*(1-self.alpha)/(self.alpha*self.APrime*self.C12))*self.sinaC12Delta+ self.sinApam2Delta
        t3 = (self.alpha*self.APrime*self.C12)/2.0*cos_2th23*self.cos_dcp*(t0 + t1*t2)
        t3 += self.sin__dcp*(1-self.alpha)*(self.cosaC12Delta-self.cosApam2Delta)*self.sinaC12Delta

        pmt_a1 = sin_th13*sin_2th12*sin_2th23/denom*t3

        # pmt_a1+pmt_a2 is the complete contribution from this expansion
                                                                                                                       
        # In order to combine the information correctly we need to add the two
        #  expansions and subtract off the terms that are in both (alpha^1, s13^1)
        #  and lower order terms
        #  these may be taken from the expansion to second order in both parameters
        #  Equation 34


        # Now for the term of order alpha * s13 or lower order!
        t0 = t1 = t2 = t3 = 0.0

        t1 = +self.sin__dcp*self.sinDelta*self.sinADelta*self.sinAm1Delta/(self.APrime*(self.APrime-1))
        t2 = -1/(self.APrime-1)*self.cos_dcp*self.sinDelta*(self.APrime*self.sinDelta-self.sinADelta*self.cosAm1Delta/self.A)*cos_2th23
        t0 =  2*self.alpha*sin_2th12*sin_2th23*sin_th13*(t1+t2)

        t1 = sinsq_2th23*self.sinDelta*self.sinDelta - self.alpha*sinsq_2th23*cos_th12*cos_th12*self.Delta*self.sin2Delta

        repeated = t0+t1

        #  Calculate the total probability
        pMuToTau = pmt_0 + pmt_1 + pmt_a0 + pmt_a1 - repeated
   


        pMuToElec=0
        #Now for the MuToElec part
  
        # this is the equivalent of Eq 47 & 48 corrected for Mu to E instead of E to Mu

        # Leading order term 
        p1 = sinsq_th23*sinsq_2th13*self.sinC13Delta*self.sinC13Delta/(self.C13*self.C13)

        # terms that appear at order alpha
        #first work out the vacuum case since we get 0/0 otherwise.......
        p2Inner = self.Delta*self.cosC13Delta

        if(math.fabs(self.APrime) > 1e-9):
            p2Inner = self.Delta*self.cosC13Delta*(1-self.APrime*cos_2th13)/self.C13 -self.APrime*self.sinC13Delta*(cos_2th13-self.APrime)/(self.C13*self.C13)

        p2 = -2*sinsq_th12*sinsq_th23*sinsq_2th13*self.sinC13Delta/(self.C13*self.C13)*p2Inner*self.alpha


        #again working out vacuum first.....
        p3Inner = self.Delta* cos_th13* cos_th13*(-2*self.sin__dcp*self.sinC13Delta*self.sinC13Delta+2*self.cos_dcp*self.sinC13Delta*self.cosC13Delta)

        if(math.fabs(self.APrime) > 1e-9):
            p3Inner = (self.sinC13Delta/(self.APrime*self.C13*self.C13))*(- self.sin__dcp*(self.cosC13Delta - self.cos1pADelta)*self.C13+ self.cos_dcp*(self.C13*self.sin1pADelta - (1-self.APrime*cos_2th13)*self.sinC13Delta))

        p3 = sin_2th12*sin_2th23*sin_th13*p3Inner*self.alpha

        #  p1 + p2 + p3 is the complete contribution for this expansion
  
        # Now for the expansion in orders of math.sin(th13) (good to all order alpha) 
        #  this is the equivalent of Eq 65 and 66

        # leading order term
        pa1 = 0.0 
        pa2 = 0.0

        # no problems here when A -> 0
        if(math.fabs(self.alpha) > 1e-10):
            # leading order term
            pa1 = cos_th23*cos_th23*sinsq_2th12*self.sinaC12Delta*self.sinaC12Delta/(self.C12*self.C12)

            # and now to calculate the first order in s13 term
            t1 = (cos_2th12 - self.APrime/self.alpha)/self.C12- self.alpha*self.APrime*self.C12*sinsq_2th12/(2*(1-self.alpha)*self.C12*self.C12)
            t2 = -self.cos_dcp*(self.sinApam2Delta-self.sinaC12Delta*t1)
            t3 = -(self.cosaC12Delta-self.cosApam2Delta)*self.sin__dcp
            denom = (1-self.APrime-self.alpha+self.APrime*self.alpha*cos_th12*cos_th12)*self.C12
            t4 = sin_2th12*sin_2th23*(1-self.alpha)*self.sinaC12Delta/denom
            pa2 = t4*(t3+t2)*sin_th13
    
        #pa1+pa2 is the complete contribution from this expansion

        # In order to combine the information correctly we need to add the two
        #  expansions and subtract off the terms that are in both (alpha^1, s13^1) 
        #  these may be taken from the expansion to second order in both parameters
        #  Equation 31 

        t1 = self.Delta*self.sinC13Delta*self.cosdpDelta
        if(math.fabs(self.APrime) > 1e-9):
            t1 = self.sinADelta*self.cosdpDelta*self.sinAm1Delta/(self.APrime*(self.APrime-1))

        repeated = 2*self.alpha*sin_2th12*sin_2th23*sin_th13*t1

        #  Calculate the total probability
        pMuToElec = p1+p2+p3 + (pa1+pa2) - repeated

        p1 = 1. - pMuToTau - pMuToElec
        if(p1 < 1e-6): 
            #  cout<<"P(mu->mu) less than zero Damnation! "<<x<<" "<<p1<<endl 
            p1 = 0
        return p1


    def ElecToTau(self,E):
        #  EtoTau is the same as E->Mu wich sinsq_th23 <-> cossq_th23, math.sin(2th23) <->-sin(2th23)
        origCos = self.cos23
        origSin = self.sin23
        orig2Sin = self.sin223

        self.cos23 = origSin
        self.sin23 = origCos
        self.sin223 = -orig2Sin

        prob = ElecToMu(E)

        #restore the world
        self.cos23 = origCos
        self.sin23 = origSin
        self.sin223 = orig2Sin
        return prob


    def ElecToMu(self, E):
        # Flip delta to reverse direction
        oldSinDelta = self.sin_dcp
        oldDelta =  self.dcp

        self.dcp = -oldDelta
        self.sin_dcp = -oldSinDelta
                                                                                                                       
        prob = self.MuToElec(E)
                                                                                                                       
        #restore the world
        self.dcp = oldDelta
        self.sin_dcp = oldSinDelta
        return prob   


    def ElecToElec(self,E):
        sinsq_2th12 = self.sinSq2Theta12
        sinsq_2th13 = self.sin213*self.sin213
                                                                                                                       
        sin_th12 = self.sin12
 
        #  cos_2th23 = self.cos223
        cos_2th13 = self.cos213
        cos_2th12 = self.cos212
                                                                                                           
        #Building the more complicated term
        self.buildTerms(E)
                                                                                                                       
        #First we calculate the terms for the alpha expansion (good to all orders in th13)
        # this is the equivalent of Eq 45 & 46 corrected for Mu to E instead of E to Mu
                                                                                                                       
        # Leading order term
        p1 = 1 - sinsq_2th13*self.sinC13Delta*self.sinC13Delta/(self.C13*self.C13)
                                                                                                                       
        # terms that appear at order alpha
        p2Inner = self.Delta*self.cosC13Delta*(1-self.APrime*cos_2th13)/self.C13 -self.APrime*self.sinC13Delta*(cos_2th13-self.APrime)/(self.C13*self.C13)
                                                                                                                       
        p2 = +2*sin_th12*sin_th12*sinsq_2th13*self.sinC13Delta/(self.C13*self.C13)*p2Inner*self.alpha
        #  p1 + p2 is the complete contribution for this expansion
                                                                                                                       
        # Now for the expansion in orders of math.sin(th13) (good to all order alpha)
        #  this is the equivalent of Eq 63 and 64
                                                                                                                       
        # leading order term
        pa1 = 1.0
        pa2 = 0.0
                                                                                                                       
        if(math.fabs(self.alpha) > 1e-10):
            # leading order term
            pa1 = 1.0 - sinsq_2th12*self.sinaC12Delta*self.sinaC12Delta/(self.C12*self.C12)
  
        #pa1 is the complete contribution from this expansion, there is no order s13^1 term
                                                                                                                       
        # In order to combine the information correctly we need to add the two
        #  expansions and subtract off the terms that are in both (alpha^1, s13^1)
        #  these may be taken from the expansion to second order in both parameters
        #  Equation 30
                                                                                                                       
        repeated = 1
                                                                                                                       
        #  Calculate the total probability
        totalP = p1+p2 + (pa1+pa2) - repeated
                                                                                                                       
        return totalP


    def TauToTau(self,E):
        #  TautoTau is the same as Mu->Mu wich sinsq_th23 <-> cossq_th23, sin(2th23) <->-sin(2th23)
        origCos = self.cos23
        origSin = self.sin23
        orig2Sin = self.sin223
        origCosSq=self.cosSqTheta23
        origSinSq=self.sinSqTheta23
        

        self.cos23 = origSin
        self.sin23 = origCos
        self.cosSqTheta23=origSinSq
        self.sinSqTheta23= origCosSq
        self.sin223 = -orig2Sin

        prob = self.MuToMu(E)
  
        #restore the world
        self.cos23 = origCos
        self.sin23 = origSin
        self.sin223 = orig2Sin
        self.cosSqTheta23=origCosSq
        self.sinSqTheta23= origSinSq
  
        return prob

  
    def TauToMu(self,E):
        # Flip delta to reverse direction
        oldSinDelta = self.sin_dcp
        oldDelta =  self.dcp
  
        self.dcp = -oldDelta
        self.sin_dcp = -oldSinDelta
                              
        prob = MuToTau(E)
                                                                                      
        #restore the world
        self.dcp = oldDelta
        self.sin_dcp = oldSinDelta
  
        return prob

  
    def TauToElec(self, E):
        # Flip delta to reverse direction
        oldSinDelta = self.sin_dcp
        oldDelta =  self.dcp
   
        self.dcp = -oldDelta
        self.sin_dcp = -oldSinDelta 
                               
        prob = ElecToTau(E)
                                 
        #restore the world 
        self.dcp = oldDelta
        self.sin_dcp = oldSinDelta 
        return prob 

if __name__ == '__main__':
    calcy = OscCalc()
    print(calcy.MuToMu(2))    
