import csv
import pandas as pd
import numpy as np
import math
'''
LOAD = pd.read_csv('./LOAD811.csv')
LOAD['date'] = pd.to_datetime(LOAD['date'])
LOAD_= pd.DataFrame()
LOAD_['date'] = pd.date_range(start='2021-08-11 00:00:00', end='2021-08-11 23:59:59', freq= '1S')
LOAD_ = pd.merge(LOAD_, LOAD, on='date', how='outer')
LOAD_ = LOAD_.interpolate()
LOAD_.to_csv('LOAD2.csv', index=False)

LOAD = pd.read_csv('./LOAD2.csv')

LOAD = LOAD['load']
LOAD = np.array(LOAD[0:86400])
LOAD = np.transpose(LOAD)

LOAD = LOAD/75000/1.01095125
print(max(LOAD))

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
'''
f = [50, 50.2, 50.5, 50.8, 51]
f0 = 50

kp = 0.1
ki = 0.01
ref = f0
x0 = 0
xi = None
y = None
update = 0
 
plot = []

for i in range (0, 5)  :

    u = f[i]
        
    """
    dot{x_i} &= k_i * (u - ref)
    y &= x_i + k_p * (u - ref)

    """
    #controller internal timestep
    
    for j in range (0, 5)  :
        
        #xi.v_str = x0
        #y.v_str = kp * u - ref + x0
        
        u = u - update
        xi = ki * (u - ref)
        y = kp * (u - ref) +  xi

        update = y

        print(update)


gameperiod = 5                     #5분동안
totaltime = gameperiod*60     #초단위
duetime = 4                        #4초마다     
Maxsignalnumber = round(totaltime/duetime)
print(Maxsignalnumber)


LOAD = pd.read_csv('./LOAD0811.csv')
LOAD = LOAD['load']
LOAD = np.array(LOAD[(60*x + 5*y)+0:(60*x + 5*y)+5])
LOAD = np.transpose(LOAD)
print(LOAD/75000)
x = 10
y = 1
r = round((60*x + 5*y)*60)
LOAD = pd.read_csv('./LOAD2.csv')
LOAD = LOAD['load']
LOAD = np.array(LOAD[r+0:r+300])
LOAD = np.transpose(LOAD)
print(LOAD)'''
print(round(1.005%1,6))