from pandas import read_json
from scipy.stats import beta
import numpy as np
import networkx as nx
import math
import collections
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv
import os
import sys
import pdb
import struct
import array
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl import load_workbook
import openpyxl
import xlsxwriter
from scipy.stats import beta

sys_path_PSSE=r'C:\Program Files\PTI\PSSE35\35.1\PSSPY37'  #or where else you find the psspy.pyc
sys.path.append(sys_path_PSSE)

os_path_PSSE=r'C:\Program Files\PTI\PSSE35\35.1\PSSBIN'  # or where else you find the psse.exe
os.environ['PATH'] += ';' + os_path_PSSE
os.environ['PATH'] += ';' + sys_path_PSSE

import psse35
import pssarrays
import redirect
import dyntools
import bsntools
import psspy

def C39(lamb):
    P39 = (lamb-8.71)/(0.0062)
    if P39 > 1090:
        P39 = 1090
    elif P39<0:
        P39 = 0
    else :
        P39 = P39
    return P39

def C31(lamb):
    P1 = (lamb-3.53)/(0.0148)
    if P1 > 760:
        P1 = 760
    elif P1<0:
        P1 = 0
    else :
        P1 = P1
    return P1
            
def C32(lamb):
    P1 = (lamb-7.58)/(0.0132)
    if P1 > 767:
        P1 = 767
    elif P1<0:
        P1 = 0
    else :
        P1 = P1
    return P1
            
def C33(lamb):
    P1 = (lamb-2.24)/(0.0126)
    if P1 > 1068:
        P1 = 1068
    elif P1<0:
        P1 = 0
    else :
        P1 = P1
    return P1
            
def C34(lamb):
    P1 = (lamb-8.53)/(0.0138)
    if P1 > 982:
        P1 = 982
    elif P1<0:
        P1 = 0
    else :
        P1 = P1
    return P1
            
def C36(lamb):
    P1 = (lamb-7.85)/(0.0038)
    if P1 > 932:
        P1 = 932
    elif P1<0:
        P1 = 0
    else :
        P1 = P1
    return P1
            
def C35(lamb):
    P1 = (lamb-2.25)/(0.0028)
    if P1 > 987:
        P1 = 987
    elif P1<0:
        P1 = 0
    else :
        P1 = P1
    return P1
            
def C37(lamb):
    P1 = (lamb-6.29)/(0.0082)
    if P1 > 882:
        P1 = 882
    elif P1<0:
        P1 = 0
    else :
        P1 = P1
    return P1
            
def C38(lamb):
    P1 = (lamb-4.3)/(0.0102)
    if P1 > 1531:
        P1 = 1531
    elif P1<0:
        P1 = 0
    else :
        P1 = P1
    return P1
            
def C30(lamb):
    P1 = (lamb-8.26)/(0.0064)
    if P1 > 250:
        P1 = 250
    elif P1<0:
        P1 = 0
    else :
        P1 = P1
    return P1
            
def Pgen(lamb):
    Pgen = C30(lamb) + C31(lamb) + C32(lamb) + C33(lamb) +C34(lamb) + C35(lamb) +C36(lamb) + C37(lamb) +C38(lamb) + C39(lamb) 
    return Pgen

def ED(lam, Pdemand):
    global lamb
    lamb = lam
    while Pgen(lamb) - Pdemand<0.1:
        lamb += 0.001
    return lamb

def Cost(gen):
    macgen = []
    macgen = gen
    p30 = macgen[0]
    p31 = macgen[1]
    p32 = macgen[2]
    p33 = macgen[3]
    p34 = macgen[4]
    p35 = macgen[5]
    p36 = macgen[6]
    p37 = macgen[7]
    p38 = macgen[8]
    p39 = macgen[9]
    cost30 = 143.41 + 8.26*p30 + 0.032*(p30)**2
    cost31 = 179.10 + 3.53*p31 + 0.074*(p31)**2
    cost32 = 90.03  + 7.58*p32 + 0.066*(p32)**2
    cost33 = 106.41 + 2.24*p33 + 0.063*(p33)**2
    cost34 = 193.8  + 8.53*p34 + 0.069*(p34)**2
    cost35 = 37.19  + 2.25*p35 + 0.014*(p35)**2
    cost36 = 200    + 7.85*p36 + 0.019*(p36)**2
    cost37 = 195.4  + 6.29*p37 + 0.041*(p37)**2
    cost38 = 62.17  + 4.30*p38 + 0.051*(p38)**2
    cost39 = 113.23 + 8.71*p39 + 0.031*(p39)**2
    
    totalcost = cost30 + cost31 + cost32 + cost33 + cost34 + cost35 + cost36 + cost37 + cost38 + cost39 

    return totalcost

def CostOPF(gen):
    macgen = []
    macgen = gen
    p30 = macgen[0]
    p31 = macgen[1]
    p32 = macgen[2]
    p33 = macgen[3]
    p34 = macgen[4]
    p35 = macgen[5]
    p36 = macgen[6]
    p37 = macgen[7]
    p38 = macgen[8]
    p39 = macgen[9]
    cost30 = 143.41 + 8.26*p30 + 0.0032*(p30)**2    #7
    cost31 = 179.10 + 3.53*p31 + 0.0074*(p31)**2    
    cost32 = 90.03  + 7.58*p32 + 0.0066*(p32)**2    #2
    cost33 = 106.41 + 2.24*p33 + 0.0063*(p33)**2    #2
    cost34 = 193.8  + 8.53*p34 + 0.0069*(p34)**2    #2
    cost35 = 37.19  + 2.25*p35 + 0.0014*(p35)**2    #8
    cost36 = 200    + 7.85*p36 + 0.0019*(p36)**2    #8
    cost37 = 195.4  + 6.29*p37 + 0.0041*(p37)**2    #4
    cost38 = 62.17  + 4.30*p38 + 0.0051*(p38)**2    #3
    cost39 = 113.23 + 8.71*p39 + 0.0031*(p39)**2    #7
    
    totalcost = cost30 + cost31 + cost32 + cost33 + cost34 + cost35 + cost36 + cost37 + cost38 + cost39
    return totalcost

def AGC_droop(x,Pder):
    generator = x
    change_mw= Pder
    z=[int(change_mw)]
    x = np.eye(len(generator))    
    for i in range(len(generator)):
        x[i][0]=0.6*int(generator[0][1])/int(generator[0][2])
        x[i][i]=(-0.6*int(generator[i][1]))/int(generator[i][2])
        x[0][i]=1        
        z.append(0)        
    z.remove(0)     
    X= np.array(x)
    Z= np.array(z)
    Y= np.linalg.solve(X,Z)
    return Y


def normal_pdf(x, mu=0, sigma=1):
    sqrt_two_pi = math.sqrt(2 * math.pi)
    return (math.exp(-(x-mu)**2 / 2 / sigma**2) / (sqrt_two_pi * sigma))

#------------------------------------------------------------
#작업폴더설적
##Set the Excel file name, working directory and path
working_dir=r"C:\Users\yungun\Desktop\labsil\윤민한교수님+강화학습\연구" + os.sep
#-------------------------------------------------------------
#PSSE 실행
redirect.psse2py()
psspy.psseinit(15000)
_i,_f = psspy.getbatdefaults()
#--------------------------------------------------------------
# Progress 실행
psspy.lines_per_page_one_device(1,60)
psspy.progress_output(2,os.path.join(working_dir,'PROGRESS.txt'),[0,0])
psspy.lines_per_page_one_device(1,60)
psspy.prompt_output(2,os.path.join(working_dir,'log.txt'),[0,0])
psspy.lines_per_page_one_device(1,60)
psspy.report_output(2,os.path.join(working_dir,'REPORT.txt'),[0,0])
psspy.lines_per_page_one_device(1,60)
psspy.alert_output(2,os.path.join(working_dir,'log.txt'),[0,0])
#----------------------------------------------------------------------------------------
# Open Source

sav_file = os.path.join(working_dir,'IEEE39.sav')

class env():
    def __init__(self, numAgent, load_model):
        self.numAgent = numAgent
        self.load_model = load_model
        self.time = 0
        self.gameperiod = 5                     #5분동안
        self.totaltime = self.gameperiod*60     #초단위
        self.duetime = 4                        #4초마다     
        self.MaxTime = round(self.totaltime/self.duetime)  #5분 동안 4초마다 = 75번 타임스탭
        self.dyr_timestep = 0.005
        self.state_size = 5
        self.action_size = 3
        self.batch_size = 10

    def reset(self,x,y):
        r = round((60*x + 5*y)*60) # x = 8 ~ 18시 , y = 0 ~ 55분 (0,5,10, ... , 55) 처음에는 하나로만 해보기 
        WT = pd.read_csv('./WT.csv')
        WT = np.array(WT[r+0:r+300])
        WT = np.transpose(WT)
        self.Wind = WT[0]/286

        PV = pd.read_csv('./PV.csv')
        PV = np.array(PV[r+0:r+300])
        PV = np.transpose(PV)
        PV = PV[0]/5140
        self.PV = PV

        LOAD = pd.read_csv('./LOAD2.csv')
        LOAD = LOAD['load']
        LOAD = np.array(LOAD[r+0:r+300])
        LOAD = np.transpose(LOAD)
        self.LOAD = LOAD/75000/1.01095125
 
        
        generator=[['C30',np.random.randint(2,9),250], ['C32',np.random.randint(2,9),767], ['C33',np.random.randint(2,9),1068], ['C34',np.random.randint(2,9),982], ['C35',np.random.randint(2,9),987], ['C36',np.random.randint(2,9),932], ['C37',np.random.randint(2,9),882], ['C38',np.random.randint(2,9),1531], ['C39',np.random.randint(2,9),1090]]
        
        
        self.rwd = np.zeros(shape=(9,))
        self.time = 0        
        self.Pdemand = 6097.1*self.LOAD[self.time]-1500*(self.PV[self.time]+self.Wind[self.time]) 
        
        lam = 8
        lamb = ED(lam, self.Pdemand)
        Ptotal = C30(lamb) + C32(lamb) + C33(lamb) + C34(lamb) + C35(lamb) + C36(lamb) + C37(lamb) + C38(lamb) + C39(lamb)
        self.fixed_pf = [C30(lamb)/Ptotal, C32(lamb)/Ptotal, C33(lamb)/Ptotal, C34(lamb)/Ptotal, C35(lamb)/Ptotal, C36(lamb)/Ptotal, C37(lamb)/Ptotal, C38(lamb)/Ptotal, C39(lamb)/Ptotal]

        self.C30 = C30(lamb)
        self.C31 = C31(lamb)
        self.C32 = C32(lamb)
        self.C33 = C33(lamb)
        self.C34 = C34(lamb)
        self.C35 = C35(lamb)
        self.C36 = C36(lamb)
        self.C37 = C37(lamb)
        self.C38 = C38(lamb)
        self.C39 = C39(lamb)

        Load  = self.LOAD[self.time]
        PVcor = self.PV[self.time]
        WTcor = self.Wind[self.time]

        psspy.case(sav_file)

        psspy.machine_chng_3(30,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ min(250, self.C30) ,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
        psspy.machine_chng_3(31,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ min(760, self.C31) ,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
        psspy.machine_chng_3(32,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ min(767, self.C32) ,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
        psspy.machine_chng_3(33,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ min(1068,self.C33) ,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
        psspy.machine_chng_3(34,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ min(982, self.C34) ,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
        psspy.machine_chng_3(35,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ min(987, self.C35) ,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
        psspy.machine_chng_3(36,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ min(932, self.C36) ,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
        psspy.machine_chng_3(37,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ min(882, self.C37) ,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
        psspy.machine_chng_3(38,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ min(1531,self.C38) ,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
        psspy.machine_chng_3(39,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ min(1090,self.C39) ,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])

        psspy.load_chng_6(3 ,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 322.0*Load , 2.40*Load,_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(4 ,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 500.0*Load ,184.0*Load,_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(7 ,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 233.8*Load , 84.0*Load,_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(8 ,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 522.0*Load ,176.0*Load,_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(12,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 7.500*Load , 88.0*Load,_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(15,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 320.0*Load ,153.0*Load,_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(16,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 329.0*Load , 32.3*Load,_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(18,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 158.0*Load , 30.0*Load,_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(20,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 628.0*Load ,103.0*Load,_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(21,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 274.0*Load ,115.0*Load,_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(23,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 247.5*Load , 84.6*Load,_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(24,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 308.6*Load ,(-92)*Load,_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(25,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 224.0*Load , 47.2*Load,_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(26,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 139.0*Load , 17.0*Load,_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(27,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 281.0*Load , 75.5*Load,_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(28,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 206.0*Load , 27.6*Load,_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(29,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 283.5*Load , 26.9*Load,_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(31,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 9.200*Load , 4.60*Load,_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(39,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[1104.0*Load ,250.0*Load,_f,_f,_f,_f,_f,_f],"")


        #신재생에너지 
        psspy.load_data_6(3 ,r"""2""",[_i,_i,_i,_i,_i,_i,_i], [-500.0*PVcor,_f,_f,_f,_f,_f,_f,_f],"")
        psspy.load_data_6(8 ,r"""2""",[_i,_i,_i,_i,_i,_i,_i], [-500.0*PVcor,_f,_f,_f,_f,_f,_f,_f],"")
        psspy.load_data_6(16,r"""2""",[_i,_i,_i,_i,_i,_i,_i], [-500.0*PVcor,_f,_f,_f,_f,_f,_f,_f],"")
        psspy.load_data_6(18,r"""2""",[_i,_i,_i,_i,_i,_i,_i], [-500.0*WTcor,_f,_f,_f,_f,_f,_f,_f],"")
        psspy.load_data_6(23,r"""2""",[_i,_i,_i,_i,_i,_i,_i], [-500.0*WTcor,_f,_f,_f,_f,_f,_f,_f],"")
        psspy.load_data_6(27,r"""2""",[_i,_i,_i,_i,_i,_i,_i], [-500.0*WTcor,_f,_f,_f,_f,_f,_f,_f],"")

        psspy.fdns([0,0,0,0,0,0,99,0])
        psspy.cong(0)
        psspy.conl(0,1,1,[0,0],[ 100.0,0.0,0.0, 100.0])
        psspy.conl(0,1,2,[0,0],[ 100.0,0.0,0.0, 100.0])
        psspy.conl(0,1,3,[0,0],[ 100.0,0.0,0.0, 100.0])
        psspy.ordr(1)
        psspy.fact()
        psspy.tysl(0)
        psspy.run(0, 1 ,0,0,0) 
        psspy.save(r"""C:\Users\yungun\Desktop\labsil\윤민한교수님+강화학습\연구\IEEE39_dyn.sav""")
        self.sav_dyr = os.path.join(working_dir,'IEEE39_dyn.sav')
        psspy.dyre_new([1,1,1,1],r"""C:\Users\yungun\Desktop\labsil\윤민한교수님+강화학습\연구\IEEE39_IEESGOD.dyr""","","","")
        psspy.dynamics_solution_param_2([_i,_i,_i,_i,_i,_i,_i,_i],[_f,_f, self.time_step,_f,_f,_f,_f,_f])
        psspy.change_channel_out_file(r"""C:\Users\yungun\Desktop\labsil\윤민한교수님+강화학습\연구\example.out""")
        psspy.chsb(0,1,[-1,-1,-1,1,12,0])
        psspy.strt_2([0,0],r"""C:\Users\yungun\Desktop\labsil\윤민한교수님+강화학습\연구\example.out""")

        psspy.run(0, 1.0,1,1,0)
        P_C30 = self.C30
        P_C31 = self.C31
        P_C32 = self.C32
        P_C33 = self.C33
        P_C34 = self.C34
        P_C35 = self.C35
        P_C36 = self.C36
        P_C37 = self.C37
        P_C38 = self.C38
        P_C39 = self.C39
        Pdif = 0
        for i in range(1,round(300/self.dyr_timestep)):
            psspy.run(0, 1.0+self.dyr_timestep*i,1,1,0)            
            ''
            if round(self.dyr_timestep*i,6) % 1 == 0:
                self.Loadfix = self.LOAD[round(self.dyr_timestep*i,6)]
                psspy.load_chng_6(3 ,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 322.0*self.Loadfix , 2.40*self.Loadfix,_f,_f,_f,_f,_f,_f],"")
                psspy.load_chng_6(4 ,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 500.0*self.Loadfix ,184.0*self.Loadfix,_f,_f,_f,_f,_f,_f],"")
                psspy.load_chng_6(7 ,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 233.8*self.Loadfix , 84.0*self.Loadfix,_f,_f,_f,_f,_f,_f],"")
                psspy.load_chng_6(8 ,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 522.0*self.Loadfix ,176.0*self.Loadfix,_f,_f,_f,_f,_f,_f],"")
                psspy.load_chng_6(12,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 7.500*self.Loadfix , 88.0*self.Loadfix,_f,_f,_f,_f,_f,_f],"")
                psspy.load_chng_6(15,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 320.0*self.Loadfix ,153.0*self.Loadfix,_f,_f,_f,_f,_f,_f],"")
                psspy.load_chng_6(16,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 329.0*self.Loadfix , 32.3*self.Loadfix,_f,_f,_f,_f,_f,_f],"")
                psspy.load_chng_6(18,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 158.0*self.Loadfix , 30.0*self.Loadfix,_f,_f,_f,_f,_f,_f],"")
                psspy.load_chng_6(20,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 628.0*self.Loadfix ,103.0*self.Loadfix,_f,_f,_f,_f,_f,_f],"")
                psspy.load_chng_6(21,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 274.0*self.Loadfix ,115.0*self.Loadfix,_f,_f,_f,_f,_f,_f],"")
                psspy.load_chng_6(23,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 247.5*self.Loadfix , 84.6*self.Loadfix,_f,_f,_f,_f,_f,_f],"")
                psspy.load_chng_6(24,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 308.6*self.Loadfix ,(-92)*self.Loadfix,_f,_f,_f,_f,_f,_f],"")
                psspy.load_chng_6(25,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 224.0*self.Loadfix , 47.2*self.Loadfix,_f,_f,_f,_f,_f,_f],"")
                psspy.load_chng_6(26,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 139.0*self.Loadfix , 17.0*self.Loadfix,_f,_f,_f,_f,_f,_f],"")
                psspy.load_chng_6(27,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 281.0*self.Loadfix , 75.5*self.Loadfix,_f,_f,_f,_f,_f,_f],"")
                psspy.load_chng_6(28,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 206.0*self.Loadfix , 27.6*self.Loadfix,_f,_f,_f,_f,_f,_f],"")
                psspy.load_chng_6(29,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 283.5*self.Loadfix , 26.9*self.Loadfix,_f,_f,_f,_f,_f,_f],"")
                psspy.load_chng_6(31,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 9.200*self.Loadfix , 4.60*self.Loadfix,_f,_f,_f,_f,_f,_f],"")
                psspy.load_chng_6(39,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[1104.0*self.Loadfix ,250.0*self.Loadfix,_f,_f,_f,_f,_f,_f],"")
                PVcor = self.PV[round(self.dyr_timestep*i,6)]
                WTcor = self.Wind[round(self.dyr_timestep*i,6)]

                    #신재생에너지 
                psspy.load_data_6(3 ,r"""2""",[_i,_i,_i,_i,_i,_i,_i], [-500.0*PVcor,_f,_f,_f,_f,_f,_f,_f],"")
                psspy.load_data_6(8 ,r"""2""",[_i,_i,_i,_i,_i,_i,_i], [-500.0*PVcor,_f,_f,_f,_f,_f,_f,_f],"")
                psspy.load_data_6(16,r"""2""",[_i,_i,_i,_i,_i,_i,_i], [-500.0*PVcor,_f,_f,_f,_f,_f,_f,_f],"")
                psspy.load_data_6(18,r"""2""",[_i,_i,_i,_i,_i,_i,_i], [-500.0*WTcor,_f,_f,_f,_f,_f,_f,_f],"")
                psspy.load_data_6(23,r"""2""",[_i,_i,_i,_i,_i,_i,_i], [-500.0*WTcor,_f,_f,_f,_f,_f,_f,_f],"")
                psspy.load_data_6(27,r"""2""",[_i,_i,_i,_i,_i,_i,_i], [-500.0*WTcor,_f,_f,_f,_f,_f,_f,_f],"")
            if round(self.dyr_timestep*i,6) % 4 == 0:
                
                P_C30 += Pdif*self.fixed_pf[0]
                P_C32 += Pdif*self.fixed_pf[1]
                P_C33 += Pdif*self.fixed_pf[2]
                P_C34 += Pdif*self.fixed_pf[3]
                P_C35 += Pdif*self.fixed_pf[4]
                P_C36 += Pdif*self.fixed_pf[5]
                P_C37 += Pdif*self.fixed_pf[6]
                P_C38 += Pdif*self.fixed_pf[7]
                P_C39 += Pdif*self.fixed_pf[8]
                psspy.increment_gref(30,r"""1""",P_C30)
                psspy.increment_gref(32,r"""1""",P_C32)
                psspy.increment_gref(33,r"""1""",P_C33)
                psspy.increment_gref(34,r"""1""",P_C34)
                psspy.increment_gref(35,r"""1""",P_C35)
                psspy.increment_gref(36,r"""1""",P_C36)
                psspy.increment_gref(37,r"""1""",P_C37)
                psspy.increment_gref(38,r"""1""",P_C38)
                psspy.increment_gref(39,r"""1""",P_C39)

            ierr, fre = psspy.chnval(34)
            fre = 60*(1-fre)
            Pdif 


        state30 = np.hstack([self.C30, Pdroop[0], self.LOAD[1], self.PV[1], self.Wind[1]]).flatten()
        state32 = np.hstack([self.C32, Pdroop[1], self.LOAD[1], self.PV[1], self.Wind[1]]).flatten()
        state33 = np.hstack([self.C33, Pdroop[2], self.LOAD[1], self.PV[1], self.Wind[1]]).flatten()
        state34 = np.hstack([self.C34, Pdroop[3], self.LOAD[1], self.PV[1], self.Wind[1]]).flatten()
        state35 = np.hstack([self.C35, Pdroop[4], self.LOAD[1], self.PV[1], self.Wind[1]]).flatten()
        state36 = np.hstack([self.C36, Pdroop[5], self.LOAD[1], self.PV[1], self.Wind[1]]).flatten()
        state37 = np.hstack([self.C37, Pdroop[6], self.LOAD[1], self.PV[1], self.Wind[1]]).flatten()
        state38 = np.hstack([self.C38, Pdroop[7], self.LOAD[1], self.PV[1], self.Wind[1]]).flatten()
        state39 = np.hstack([self.C39, Pdroop[8], self.LOAD[1], self.PV[1], self.Wind[1]]).flatten()

        states = [state30, state32, state33, state34, state35, state36, state37, state38, state39 ]

        return states ,PVcor,WTcor

    def step(self, acts, PV_,WT_):        
        a = acts

        self.rwd = np.zeros(shape=(9,))
        terminal = False     
             
        n_state, rwd, PVcor, Wtcor, ifo , ifoGEN, ifoPVWT = self._step(a,PV_,WT_)

        self.rwd[0] = rwd[0]
        self.rwd[1] = rwd[1]
        self.rwd[2] = rwd[2]
        self.rwd[3] = rwd[3]
        self.rwd[4] = rwd[4]
        self.rwd[5] = rwd[5]
        self.rwd[6] = rwd[6]
        self.rwd[7] = rwd[7]
        self.rwd[8] = rwd[8]

             
        next_state = n_state
        total_reward = [self.rwd[0], self.rwd[1], self.rwd[2], self.rwd[3], self.rwd[4], self.rwd[5], self.rwd[6], self.rwd[7], self.rwd[8]]
        terminals = [False,False,False,False,False,False,False,False,False]
        info = ifo

        self.time = self.time + 1
        if self.time == self.MaxTime:
            self.time = 0
            
            terminals = [True,True,True,True,True,True,True,True,True]

        return next_state, total_reward, terminals, PVcor, Wtcor, ifo , ifoGEN, ifoPVWT 

    def noact(self, PV_,WT_):
        self.rwd = np.zeros(shape=(9,))
        self.time = 0        
        self.Pdemand = 6097.1*self.LOAD[self.time]-1500*(self.PV[self.time]+self.Wind[self.time]) 
        
        lam = 8
        lamb = ED(lam, self.Pdemand)
        Ptotal = C30(lamb) + C32(lamb) + C33(lamb) + C34(lamb) + C35(lamb) + C36(lamb) + C37(lamb) + C38(lamb) + C39(lamb)
        fixed_pf = [C30(lamb)/Ptotal, C32(lamb)/Ptotal, C33(lamb)/Ptotal, C34(lamb)/Ptotal, C35(lamb)/Ptotal, C36(lamb)/Ptotal, C37(lamb)/Ptotal, C38(lamb)/Ptotal, C39(lamb)/Ptotal]


        Pdif = self.Pdemand - Pgen(lamb)  # 주파수 PI 지난값 
        Pdroop = AGC_droop(generator, Pdif)

        Load  = self.LOAD[self.time]
        PVcor = self.PV[self.time]
        WTcor = self.Wind[self.time]

        psspy.case(self.sav_dyr)
        for i in range(round(5/time_step)):

        psspy.machine_chng_3(30,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ min(250, C30(lamb) + Pdroop[0]) ,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
        psspy.machine_chng_3(31,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ min(760, C31(lamb)            ) ,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
        psspy.machine_chng_3(32,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ min(767, C32(lamb) + Pdroop[1]) ,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
        psspy.machine_chng_3(33,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ min(1068,C33(lamb) + Pdroop[2]) ,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
        psspy.machine_chng_3(34,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ min(982, C34(lamb) + Pdroop[3]) ,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
        psspy.machine_chng_3(35,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ min(987, C35(lamb) + Pdroop[4]) ,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
        psspy.machine_chng_3(36,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ min(932, C36(lamb) + Pdroop[5]) ,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
        psspy.machine_chng_3(37,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ min(882, C37(lamb) + Pdroop[6]) ,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
        psspy.machine_chng_3(38,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ min(1531,C38(lamb) + Pdroop[7]) ,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
        psspy.machine_chng_3(39,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ min(1090,C39(lamb) + Pdroop[8]) ,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])

        psspy.load_chng_6(3 ,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 322.0*Load , 2.40*Load,_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(4 ,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 500.0*Load ,184.0*Load,_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(7 ,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 233.8*Load , 84.0*Load,_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(8 ,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 522.0*Load ,176.0*Load,_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(12,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 7.500*Load , 88.0*Load,_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(15,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 320.0*Load ,153.0*Load,_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(16,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 329.0*Load , 32.3*Load,_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(18,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 158.0*Load , 30.0*Load,_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(20,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 628.0*Load ,103.0*Load,_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(21,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 274.0*Load ,115.0*Load,_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(23,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 247.5*Load , 84.6*Load,_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(24,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 308.6*Load ,(-92)*Load,_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(25,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 224.0*Load , 47.2*Load,_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(26,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 139.0*Load , 17.0*Load,_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(27,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 281.0*Load , 75.5*Load,_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(28,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 206.0*Load , 27.6*Load,_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(29,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 283.5*Load , 26.9*Load,_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(31,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 9.200*Load , 4.60*Load,_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(39,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[1104.0*Load ,250.0*Load,_f,_f,_f,_f,_f,_f],"")


        #신재생에너지 
        psspy.load_data_6(3 ,r"""2""",[_i,_i,_i,_i,_i,_i,_i], [-500.0*PVcor,_f,_f,_f,_f,_f,_f,_f],"")
        psspy.load_data_6(8 ,r"""2""",[_i,_i,_i,_i,_i,_i,_i], [-500.0*PVcor,_f,_f,_f,_f,_f,_f,_f],"")
        psspy.load_data_6(16,r"""2""",[_i,_i,_i,_i,_i,_i,_i], [-500.0*PVcor,_f,_f,_f,_f,_f,_f,_f],"")
        psspy.load_data_6(18,r"""2""",[_i,_i,_i,_i,_i,_i,_i], [-500.0*WTcor,_f,_f,_f,_f,_f,_f,_f],"")
        psspy.load_data_6(23,r"""2""",[_i,_i,_i,_i,_i,_i,_i], [-500.0*WTcor,_f,_f,_f,_f,_f,_f,_f],"")
        psspy.load_data_6(27,r"""2""",[_i,_i,_i,_i,_i,_i,_i], [-500.0*WTcor,_f,_f,_f,_f,_f,_f,_f],"")


        psspy.fdns([0,0,0,0,0,0,99,0])
        psspy.cong(0)
        psspy.conl(0,1,1,[0,0],[ 100.0,0.0,0.0, 100.0])
        psspy.conl(0,1,2,[0,0],[ 100.0,0.0,0.0, 100.0])
        psspy.conl(0,1,3,[0,0],[ 100.0,0.0,0.0, 100.0])
        psspy.ordr(1)
        psspy.fact()
        psspy.tysl(0)
        psspy.dyre_new([1,1,1,1],r"""C:\Users\yungun\Desktop\labsil\윤민한교수님+강화학습\연구\IEEE39_IEESGOD.dyr""","","","")
        

        psspy.dynamics_solution_param_2([_i,_i,_i,_i,_i,_i,_i,_i],[_f,_f, self.dyr_timestep,_f,_f,_f,_f,_f])
        psspy.change_channel_out_file(r"""C:\Users\yungun\Desktop\labsil\윤민한교수님+강화학습\연구\example.out""")
        psspy.chsb(0,1,[-1,-1,-1,1,12,0])
        psspy.strt_2([0,0],r"""C:\Users\yungun\Desktop\labsil\윤민한교수님+강화학습\연구\example.out""")



        GEN = []
        for j in range(10):
            [xx,x]=psspy.macdat(30+j,'1','P')
            GEN.append(x)
        OPFGEN = []
        #OPFGEN = [C30(lam) , C31(lam) , C32(lam) , C33(lam) ,C34(lam) , C35(lam) ,C36(lam) , C37(lam) ,C38(lam) , C39(lam)] 
        cost = Cost(GEN) 

        return cost
    def _step(self,acts, PV_,WT_):
        a1 = acts[0]+6 # 6-8
        a2 = acts[1]+6 # 6-8
        a3 = acts[2]+2 # 2-4
        a4 = acts[3]+6 # 6-8
        a5 = acts[4]+5 # 5-7
        a6 = acts[5]+6 # 6-8
        a7 = acts[6]+3 # 3-5
        a8 = acts[7]+2 # 2-4
        a9 = acts[8]+6 # 6-8
        for i in range(4/self.dyr_timestep):
            if (5/self.dyr_timestep)%(1/self.dyr_timestep) == 

        total_reward = 0
        generator=[['C30',a1,250], ['C32',a2,767], ['C33',a3,1068], ['C34',a4,982], ['C35',a5,987], ['C36',a6,932], ['C37',a7,882], ['C38',a8,1531], ['C39',a9,1090]]
        cost=0
        self.Pdemand = 6097.1*self.LOAD[self.time]-1500*(self.PV[self.time]+self.Wind[self.time])

        if self.time%5==0:
            lam = 10
            ED(lam, self.Pdemand)

        Pdif = self.Pdemand - Pgen(lamb)
        
        Pdroop = AGC(generator, Pdif)

        psspy.case(sav_file)
        psspy.machine_chng_3(30,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ min(250, C30(lamb) + Pdroop[0]) ,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
        psspy.machine_chng_3(31,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ min(760, C31(lamb)            ) ,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
        psspy.machine_chng_3(32,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ min(767, C32(lamb) + Pdroop[1]) ,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
        psspy.machine_chng_3(33,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ min(1068,C33(lamb) + Pdroop[2]) ,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
        psspy.machine_chng_3(34,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ min(982, C34(lamb) + Pdroop[3]) ,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
        psspy.machine_chng_3(35,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ min(987, C35(lamb) + Pdroop[4]) ,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
        psspy.machine_chng_3(36,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ min(932, C36(lamb) + Pdroop[5]) ,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
        psspy.machine_chng_3(37,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ min(882, C37(lamb) + Pdroop[6]) ,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
        psspy.machine_chng_3(38,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ min(1531,C38(lamb) + Pdroop[7]) ,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
        psspy.machine_chng_3(39,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ min(1090,C39(lamb) + Pdroop[8]) ,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])

        psspy.load_chng_6(3 ,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 322.0*self.LOAD[self.time] , 2.40*self.LOAD[self.time],_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(4 ,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 500.0*self.LOAD[self.time] ,184.0*self.LOAD[self.time],_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(7 ,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 233.8*self.LOAD[self.time] , 84.0*self.LOAD[self.time],_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(8 ,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 522.0*self.LOAD[self.time] ,176.0*self.LOAD[self.time],_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(12,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 7.500*self.LOAD[self.time] , 88.0*self.LOAD[self.time],_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(15,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 320.0*self.LOAD[self.time] ,153.0*self.LOAD[self.time],_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(16,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 329.0*self.LOAD[self.time] , 32.3*self.LOAD[self.time],_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(18,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 158.0*self.LOAD[self.time] , 30.0*self.LOAD[self.time],_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(20,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 628.0*self.LOAD[self.time] ,103.0*self.LOAD[self.time],_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(21,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 274.0*self.LOAD[self.time] ,115.0*self.LOAD[self.time],_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(23,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 247.5*self.LOAD[self.time] , 84.6*self.LOAD[self.time],_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(24,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 308.6*self.LOAD[self.time] ,(-92)*self.LOAD[self.time],_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(25,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 224.0*self.LOAD[self.time] , 47.2*self.LOAD[self.time],_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(26,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 139.0*self.LOAD[self.time] , 17.0*self.LOAD[self.time],_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(27,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 281.0*self.LOAD[self.time] , 75.5*self.LOAD[self.time],_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(28,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 206.0*self.LOAD[self.time] , 27.6*self.LOAD[self.time],_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(29,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 283.5*self.LOAD[self.time] , 26.9*self.LOAD[self.time],_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(31,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ 9.200*self.LOAD[self.time] , 4.60*self.LOAD[self.time],_f,_f,_f,_f,_f,_f],"")
        psspy.load_chng_6(39,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[1104.0*self.LOAD[self.time] ,250.0*self.LOAD[self.time],_f,_f,_f,_f,_f,_f],"")


        #신재생에너지 
        psspy.load_data_6(3 ,r"""2""",[_i,_i,_i,_i,_i,_i,_i], [-500.0*PV_,_f,_f,_f,_f,_f,_f,_f],"")
        psspy.load_data_6(8 ,r"""2""",[_i,_i,_i,_i,_i,_i,_i], [-500.0*PV_,_f,_f,_f,_f,_f,_f,_f],"")
        psspy.load_data_6(16,r"""2""",[_i,_i,_i,_i,_i,_i,_i], [-500.0*PV_,_f,_f,_f,_f,_f,_f,_f],"")
        psspy.load_data_6(18,r"""2""",[_i,_i,_i,_i,_i,_i,_i], [-500.0*WT_,_f,_f,_f,_f,_f,_f,_f],"")
        psspy.load_data_6(23,r"""2""",[_i,_i,_i,_i,_i,_i,_i], [-500.0*WT_,_f,_f,_f,_f,_f,_f,_f],"")
        psspy.load_data_6(27,r"""2""",[_i,_i,_i,_i,_i,_i,_i], [-500.0*WT_,_f,_f,_f,_f,_f,_f,_f],"")

        PVcor = self.PV[(self.time+1)%self.MaxTime]*(1+((np.random.rand()-0.5)*2)*0.1)
        WTcor = self.Wind[(self.time+1)%self.MaxTime]*(1+((np.random.rand()-0.5)*2)*0.3)


        psspy.fnsl([0,0,0,1,0,0,0,0])
        psspy.fnsl([0,0,0,1,0,0,0,0])

        GEN = []
        for j in range(10):
            [xx,x]=psspy.macdat(30+j,'1','P')
            GEN.append(x)
        OPFGEN = []
        #OPFGEN = [C30(lam) , C31(lam) , C32(lam) , C33(lam) ,C34(lam) , C35(lam) ,C36(lam) , C37(lam) ,C38(lam) , C39(lam)] 

        TotalCost = Cost(GEN)
        noactcost = self.noact(PV_,WT_)
        cost = TotalCost - noactcost    


        state30 = np.hstack([C30(lamb), Pdroop[0], self.LOAD[(self.time+1)%self.MaxTime], PVcor, WTcor]).flatten()
        state32 = np.hstack([C32(lamb), Pdroop[1], self.LOAD[(self.time+1)%self.MaxTime], PVcor, WTcor]).flatten()
        state33 = np.hstack([C33(lamb), Pdroop[2], self.LOAD[(self.time+1)%self.MaxTime], PVcor, WTcor]).flatten()
        state34 = np.hstack([C34(lamb), Pdroop[3], self.LOAD[(self.time+1)%self.MaxTime], PVcor, WTcor]).flatten()
        state35 = np.hstack([C35(lamb), Pdroop[4], self.LOAD[(self.time+1)%self.MaxTime], PVcor, WTcor]).flatten()
        state36 = np.hstack([C36(lamb), Pdroop[5], self.LOAD[(self.time+1)%self.MaxTime], PVcor, WTcor]).flatten()
        state37 = np.hstack([C37(lamb), Pdroop[6], self.LOAD[(self.time+1)%self.MaxTime], PVcor, WTcor]).flatten()
        state38 = np.hstack([C38(lamb), Pdroop[7], self.LOAD[(self.time+1)%self.MaxTime], PVcor, WTcor]).flatten()
        state39 = np.hstack([C39(lamb), Pdroop[8], self.LOAD[(self.time+1)%self.MaxTime], PVcor, WTcor]).flatten()

        next_states = [state30, state32, state33, state34, state35, state36, state37, state38, state39 ]

        self.rwd = np.zeros(shape=(9,))

        self.rwd[0] = -cost
        self.rwd[1] = -cost
        self.rwd[2] = -cost
        self.rwd[3] = -cost
        self.rwd[4] = -cost
        self.rwd[5] = -cost
        self.rwd[6] = -cost
        self.rwd[7] = -cost
        self.rwd[8] = -cost

        total_rewards = self.rwd
        info = [a1,a2,a3,a4,a5,a6,a7,a8,a9, TotalCost ,noactcost, PV_, WT_]
        infoGEN = GEN
        infoPVWT = [PV_, WT_]

        return next_states , total_rewards , PVcor, WTcor, info, infoGEN, infoPVWT