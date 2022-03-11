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

sys_path_PSSE=r'C:\Program Files\PTI\PSSE35\35.2\PSSPY37'  #or where else you find the psspy.pyc
sys.path.append(sys_path_PSSE)

os_path_PSSE=r'C:\Program Files\PTI\PSSE35\35.2\PSSBIN'  # or where else you find the psse.exe
os.environ['PATH'] += ';' + os_path_PSSE
os.environ['PATH'] += ';' + sys_path_PSSE

import psse35
import pssarrays
import redirect
import dyntools
import bsntools
import psspy

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

def AGC(x,Pder):
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


class env():
    def __init__(self, numAgent, load_model):
        self.numAgent = numAgent
        self.load_model = load_model
        self.time = 0
        self.MaxTime = 60 
        self.state_size = 5
        self.action_size = 7
        self.batch_size = 10
        self.interval = 60 

    def reset(self):
        '''
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
        self.PV = [0.7521766495441423 ,0.7535741696396218 ,0.7548454769185965 ,0.7571540240713919 ,0.7571342055299571 ,0.7572084729062807 ,0.7619505283310222 ,0.7644447439246261 ,
0.765628015156386 ,0.7658595791668327 ,0.767694150255002 ,0.7692005680202592 ,0.7725649218909558 ,0.7732320765805131 ,0.7743894794002951 ,0.7754094041273886 ,0.7760988807530879 ,
0.7781994375289325 ,0.777032229746546 ,0.7788640888237821 ,0.7804572909389003 ,0.7822532680253286 ,0.7818523076396723 ,0.7836816633222009 ,0.7827051307700379 ,0.78386253358982 ,
0.7848006807564675 ,0.7844652258656583 ,0.7859804055123917 ,0.7878890353606589 ,0.7885190563620543 ,0.7941080936628441 ,0.7948424227770533 ,0.7974073592711449 ,0.7966625993456543 ,
0.7989513279570151 ,0.797806233494545 ,0.798515528661679 ,0.798734575698588 ,0.7993856669387718 ,0.7993066013892589 ,0.8013266323020112 ,0.8019405898540333 ,0.8026486333238135 ,
0.8034420008298754 ,0.8050487629996593 ,0.804537444630646 ,0.8035744721331489 ,0.8063459386906119 ,0.8075231600518287 ,0.8089663671007206 ,0.8087051795862348 ,0.8089999543130466 ,
0.8086730526874881 ,0.8097808048455708 ,0.8111895902172341 ,0.8120572250996194 ,0.8141888656930825 ,0.8158586299630061 ,0.8189136059710971,0.8289136059710971 ]
        self.LOAD = [1.0 ,0.9979161995715697 ,0.9973039321013215 ,0.9947960804863331 ,0.9904808490423999 ,0.9861952341030674 ,0.9861421309704715 ,0.9851562699031806 ,
0.9833717315042246 ,0.9827173943489078 ,0.9784239075313886 ,0.9774206893823454 ,0.9776178100922863 ,0.9803403971817356 ,0.9812303112267905 ,0.9777049172741447 ,
0.9797363887745505 ,0.9796463778713005 ,0.9752355220041345 ,0.9738115434096201 ,0.9706741318285041 ,0.9713521492042545 ,0.973877357735814 ,0.9703441564908545 ,
0.9696890449657849 ,0.9681563435695633 ,0.966079576228875 ,0.9630768261932849 ,0.9648010992001567 ,0.968648015834951 ,0.9671027322516964 ,0.9654737610690542 ,
0.9622209807833656 ,0.9588387008665658 ,0.9588585741544707 ,0.9572532187715701 ,0.9536814321750672 ,0.9506477750736823 ,0.9469129364023738 ,0.9440976282807211 ,
0.9425325358302533 ,0.9386022541950746 ,0.9373221640061589 ,0.9358200469115476 ,0.9333600075081756 ,0.9316163774048402 ,0.9303338996859741 ,0.9328198776679836 ,
0.928231516530833 ,0.9266112560168223 ,0.927417805226674 ,0.927171904508107 ,0.9255660329337649 ,0.9229740421138205 ,0.9204219270414327 ,0.923071344283124 ,
0.9188536084357511 ,0.9175478376813281 ,0.9190304685077294 ,0.9140001184901348  ,0.910001184901348 ]
        self.Wind = [0.7457627118644068 ,0.7966101694915254 ,0.6440677966101694 ,0.6779661016949152 ,0.8644067796610169 ,0.5084745762711864 ,0.847457627118644 ,
0.7457627118644068 ,0.5254237288135593 ,0.8135593220338982 ,0.7288135593220338 ,1.0 ,0.8135593220338982 ,0.8813559322033898 ,0.7796610169491525 ,0.7288135593220338 ,
0.5762711864406779 ,0.6949152542372881 ,0.7796610169491525 ,0.7966101694915254 ,0.7457627118644068 ,0.5932203389830508 ,0.6271186440677966 ,0.5254237288135593 ,
0.5423728813559322 ,0.711864406779661 ,0.5254237288135593 ,0.711864406779661 ,0.6271186440677966 ,0.7288135593220338 ,0.6949152542372881 ,0.5254237288135593 ,
0.5254237288135593 ,0.47457627118644063 ,0.5084745762711864 ,0.7627118644067796 ,0.5423728813559322 ,0.30508474576271183 ,0.3220338983050847 ,0.5254237288135593 ,
0.6779661016949152 ,0.5932203389830508 ,0.6949152542372881 ,0.6271186440677966 ,0.5762711864406779 ,0.5084745762711864 ,0.423728813559322 ,0.5423728813559322 ,0.5762711864406779 ,
0.6271186440677966 ,0.5423728813559322 ,0.38983050847457623 ,0.38983050847457623 ,0.6101694915254237 ,0.7796610169491525 ,0.7627118644067796 ,0.8644067796610169 ,
0.5254237288135593 ,0.5254237288135593 ,0.5084745762711864,0.38983050847457623 ] 

        generator=[['C30',np.random.randint(2,9),250], ['C32',np.random.randint(2,9),767], ['C33',np.random.randint(2,9),1068], ['C34',np.random.randint(2,9),982], ['C35',np.random.randint(2,9),987], ['C36',np.random.randint(2,9),932], ['C37',np.random.randint(2,9),882], ['C38',np.random.randint(2,9),1531], ['C39',np.random.randint(2,9),1090]]
        
        
        self.rwd = np.zeros(shape=(9,))
        self.time = 0
        
        self.Pdemand = 6097.1*self.LOAD[self.time]-1500*(self.PV[self.time]+self.Wind[self.time])
        lam = 9
        lamb = ED(lam, self.Pdemand)
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

        PVcor = self.PV[self.time]*(1+((np.random.rand()-0.5)*2)*0.2)
        WTcor = self.Wind[self.time]*(1+((np.random.rand()-0.5)*2)*0.3)

        #신재생에너지 
        psspy.load_data_6(3 ,r"""2""",[_i,_i,_i,_i,_i,_i,_i], [-500.0*PVcor,_f,_f,_f,_f,_f,_f,_f],"")
        psspy.load_data_6(8 ,r"""2""",[_i,_i,_i,_i,_i,_i,_i], [-500.0*PVcor,_f,_f,_f,_f,_f,_f,_f],"")
        psspy.load_data_6(16,r"""2""",[_i,_i,_i,_i,_i,_i,_i], [-500.0*PVcor,_f,_f,_f,_f,_f,_f,_f],"")
        psspy.load_data_6(18,r"""2""",[_i,_i,_i,_i,_i,_i,_i], [-500.0*WTcor,_f,_f,_f,_f,_f,_f,_f],"")
        psspy.load_data_6(23,r"""2""",[_i,_i,_i,_i,_i,_i,_i], [-500.0*WTcor,_f,_f,_f,_f,_f,_f,_f],"")
        psspy.load_data_6(27,r"""2""",[_i,_i,_i,_i,_i,_i,_i], [-500.0*WTcor,_f,_f,_f,_f,_f,_f,_f],"")


        psspy.fnsl([0,0,0,1,0,0,0,0])
        psspy.fnsl([0,0,0,1,0,0,0,0])

        GEN = []
        for j in range(10):
            [xx,x]=psspy.macdat(30+j,'1','P')
            GEN.append(x)
        cost = Cost(GEN)

        state30 = np.hstack([C30(lamb), Pdroop[0], self.LOAD[1], self.PV[1], self.Wind[1]]).flatten()
        state32 = np.hstack([C32(lamb), Pdroop[1], self.LOAD[1], self.PV[1], self.Wind[1]]).flatten()
        state33 = np.hstack([C33(lamb), Pdroop[2], self.LOAD[1], self.PV[1], self.Wind[1]]).flatten()
        state34 = np.hstack([C34(lamb), Pdroop[3], self.LOAD[1], self.PV[1], self.Wind[1]]).flatten()
        state35 = np.hstack([C35(lamb), Pdroop[4], self.LOAD[1], self.PV[1], self.Wind[1]]).flatten()
        state36 = np.hstack([C36(lamb), Pdroop[5], self.LOAD[1], self.PV[1], self.Wind[1]]).flatten()
        state37 = np.hstack([C37(lamb), Pdroop[6], self.LOAD[1], self.PV[1], self.Wind[1]]).flatten()
        state38 = np.hstack([C38(lamb), Pdroop[7], self.LOAD[1], self.PV[1], self.Wind[1]]).flatten()
        state39 = np.hstack([C39(lamb), Pdroop[8], self.LOAD[1], self.PV[1], self.Wind[1]]).flatten()

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
        generator=[['C30',7,250], ['C32',2,767], ['C33',2,1068], ['C34',2,982], ['C35',8,987], ['C36',8,932], ['C37',4,882], ['C38',3,1531], ['C39',7,1090]]
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
        cost = Cost(GEN) 

        return cost
    def _step(self,acts, PV_,WT_):
        a1 = acts[0]+2
        a2 = acts[1]+2
        a3 = acts[2]+2
        a4 = acts[3]+2
        a5 = acts[4]+2
        a6 = acts[5]+2
        a7 = acts[6]+2
        a8 = acts[7]+2
        a9 = acts[8]+2
        

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