# This is a class to fit toy MC data sets in neutrino oscillation contexts
# Ryan Nichol <r.nichol@ucl.ac.uk>
# 3rd June 2020
import math
import numpy as np

class FitTools:
    # create getPoissonLL as static method
    @staticmethod
    def getPoissonLL(modelVals,dataVals):
        ll=0
        for vi,ni in zip(modelVals,dataVals):
            if vi>0:
                if ni>0:
                    ll+=(vi-ni)+ni*math.log(ni/vi)
                else:
                    ll+=vi
        return 2*ll

    @staticmethod
    def profile3Dto2D(inputArr,whichAxes=[0,1]):
        axOrder=whichAxes+[(3-sum(whichAxes))]
        #print(axOrder)
        inArr=np.array(inputArr)
        midArr=np.transpose(inArr,tuple(axOrder))
        #print(midArr)
        outArr=[[ np.min(arRow) for arRow in arMat] for arMat in midArr]
        return outArr

    @staticmethod
    def profile3Dto1D(inputArr,whichAxis=0):
        axOrder=(whichAxis,(whichAxis+1)%3,(whichAxis+2)%3)
        print(axOrder)
        inArr=np.array(inputArr)
        midArr=np.transpose(inArr,tuple(axOrder))
        #print(midArr)
        outArr=[ np.min(arMat) for arMat in midArr]
        return outArr


