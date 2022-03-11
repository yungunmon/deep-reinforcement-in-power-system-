import csv
import pandas as pd
import numpy as np
import math
'''
LOAD = pd.read_csv('./LOAD811.csv')
LOAD['date'] = pd.to_datetime(LOAD['date'])
LOAD_= pd.DataFrame()
LOAD_['date'] = pd.date_range(start='2021-08-11 00:00:00', end='2021-08-11 23:59:00', freq='MIN')
LOAD_ = pd.merge(LOAD_, LOAD, on='date', how='outer')
LOAD_ = LOAD_.interpolate()
LOAD_.to_csv('LOAD0811.csv', index=False)
'''
LOAD = pd.read_csv('./LOAD0811.csv')

LOAD = LOAD['load']
LOAD = np.array(LOAD[0:1440])
LOAD = np.transpose(LOAD)

LOAD = LOAD/75000

'''
def normal_pdf(x, mu=0, sigma=1):
    sqrt_two_pi = math.sqrt(2 * math.pi)
    return (math.exp(-(x-mu)**2 / 2 / sigma**2) / (sqrt_two_pi * sigma))

xs1 = [x/144 for x in range(-12*60, 12*60)]
pv1 = np.array([normal_pdf(x,sigma=1) for x in xs1]) 

PV = pd.read_csv('./PV.csv')
PV = np.array(PV[0:1440])
PV = np.transpose(PV)
PV = PV[0]/50000000

# 1440 일때

WT = pd.read_csv('./WT.csv')
        WT = np.array(WT[0:1440])
        WT = np.transpose(WT)
        self.Wind = WT[0]/8

        LOAD = pd.read_csv('./LOAD.csv')
        LOAD = LOAD['load']
        LOAD = np.array(LOAD[0:1440])
        LOAD = np.transpose(LOAD)
        self.LOAD = LOAD/52000

        PV = pd.read_csv('./PV.csv')
        PV = np.array(PV[0:1440])
        PV = np.transpose(PV)
        PV = PV[0]/50000000
        self.PV = PV
'''