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

import win32com.client
from pylab import *
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ColorConverter
import matplotlib.text as text

path = 'C:/Users/yungun/Desktop/labsil/Distributed_unbalance_mitigate/QVDroop_ppo_act36_3600sim_revision34_2/MajorRevision/'
matA =  np.zeros((3,3), dtype=complex)
matA = np.array([[1 , 1, 1],
    [1, -0.5 - np.sqrt(3)*0.5j,-0.5 + np.sqrt(3)*0.5j],
    [1, -0.5 + np.sqrt(3)*0.5j,-0.5 - np.sqrt(3)*0.5j]])
invA = np.linalg.inv(matA)
        
class env():
    def __init__(self, numAgent, load_model):        
        self.numAgent = numAgent
        self.load_model = load_model
        self.time = 0
        self.MaxTime = 720
        self.totaltime = 4320
        self.state_size = 2
        self.action_size = 36
        self.batch_size = 10
        self.interval = 60 
        self.action = [[ 0.92 , 0.94 , 1.02 , 1.04],
                       [ 0.92 , 0.94 , 1.02 , 1.06],
                       [ 0.92 , 0.94 , 1.02 , 1.08],
                       [ 0.92 , 0.94 , 1.04 , 1.06],
                       [ 0.92 , 0.94 , 1.04 , 1.08],
                       [ 0.92 , 0.94 , 1.06 , 1.08],
                       [ 0.92 , 0.96 , 1.02 , 1.04],
                       [ 0.92 , 0.96 , 1.02 , 1.06],
                       [ 0.92 , 0.96 , 1.02 , 1.08],
                       [ 0.92 , 0.96 , 1.04 , 1.06],
                       [ 0.92 , 0.96 , 1.04 , 1.08],
                       [ 0.92 , 0.96 , 1.06 , 1.08],
                       [ 0.92 , 0.98 , 1.02 , 1.04],
                       [ 0.92 , 0.98 , 1.02 , 1.06],
                       [ 0.92 , 0.98 , 1.02 , 1.08],
                       [ 0.92 , 0.98 , 1.04 , 1.06],
                       [ 0.92 , 0.98 , 1.04 , 1.08],
                       [ 0.92 , 0.98 , 1.06 , 1.08],
                       [ 0.94 , 0.96 , 1.02 , 1.04],
                       [ 0.94 , 0.96 , 1.02 , 1.06],
                       [ 0.94 , 0.96 , 1.02 , 1.08],
                       [ 0.94 , 0.96 , 1.04 , 1.06],
                       [ 0.94 , 0.96 , 1.04 , 1.08],
                       [ 0.94 , 0.96 , 1.06 , 1.08],
                       [ 0.94 , 0.98 , 1.02 , 1.04],
                       [ 0.94 , 0.98 , 1.02 , 1.06],
                       [ 0.94 , 0.98 , 1.02 , 1.08],
                       [ 0.94 , 0.98 , 1.04 , 1.06],
                       [ 0.94 , 0.98 , 1.04 , 1.08],
                       [ 0.94 , 0.98 , 1.06 , 1.08],
                       [ 0.96 , 0.98 , 1.02 , 1.04],
                       [ 0.96 , 0.98 , 1.02 , 1.06],
                       [ 0.96 , 0.98 , 1.02 , 1.08],
                       [ 0.96 , 0.98 , 1.04 , 1.06],
                       [ 0.96 , 0.98 , 1.04 , 1.08],
                       [ 0.96 , 0.98 , 1.06 , 1.08]]  

    def reset(self,ra):
        train = pd.read_csv('C:/Users/yungun/Desktop/labsil/Distributed_unbalance_mitigate/QVDroop_ppo_act36_3600sim_revision34/fileout.csv')
        pv = np.array(train)
        pv = np.transpose(pv)
        pv = pv[ra]/6000*(1+((np.random.rand()-0.5)*2)*0.1) #0.9~1.1
        self.PV = pv[1960:7000]                             # P 15번째
        '''
        with open('C:/Users/yungun/Desktop/labsil/Distributed_unbalance_mitigate/QVDroop_ppo_act36_8640sim/pvtest.csv', 'a', newline='') as mycsvfile:
            wr = csv.writer(mycsvfile)         
            wr.writerow(self.PV)'''

        LOAD = pd.read_csv('C:/Users/yungun/Desktop/labsil/Distributed_unbalance_mitigate/QVDroop_ppo_act36_3600sim_revision34/LOAD.csv')
        LOAD = LOAD['load']
        LOAD = np.array(LOAD[2160:7200])
        LOAD = np.transpose(LOAD)/60000
        self.Load = LOAD 

        self.rwd = np.zeros(shape=(12,))
        self.time = 0
        dssObj = win32com.client.Dispatch('OpenDSSEngine.DSS')
        dssText = dssObj.Text 
        dssCircuit = dssObj.ActiveCircuit 
        dssSolution = dssCircuit.Solution 
        dssElem = dssCircuit.ActiveCktElement 
        dssBus = dssCircuit.ActiveBus
        dssText.Command= 'Clear'
        dssText.Command= 'new circuit.IEEE34'
        dssText.Command= '~ basekv=69 pu=1.05 angle=30 mvasc3=200000'
        dssText.Command= 'New Transformer.SubXF Phases=3 Windings=2 Xhl=0.01    ! normally 8'
        dssText.Command= '~ wdg=1 bus=sourcebus conn=Delta kv=69    kva=25000   %r=0.0005   !reduce %r, too'
        dssText.Command= '~ wdg=2 bus=800       conn=wye   kv=24.9  kva=25000   %r=0.0005'
        dssText.Command= 'Redirect        IEEELineCodes.dss   ! revised according to Later test feeder doc'
        dssText.Command= 'New Line.L1     Phases=3 Bus1=800.1.2.3  Bus2=802.1.2.3  LineCode=300  Length=2.58   units=kft'
        dssText.Command= 'New Line.L2     Phases=3 Bus1=802.1.2.3  Bus2=806.1.2.3  LineCode=300  Length=1.73   units=kft'
        dssText.Command= 'New Line.L3     Phases=3 Bus1=806.1.2.3  Bus2=808.1.2.3  LineCode=300  Length=32.23   units=kft'
        dssText.Command= 'New Line.L4     Phases=1 Bus1=808.2      Bus2=810.2      LineCode=303  Length=5.804   units=kft'
        dssText.Command= 'New Line.L5     Phases=3 Bus1=808.1.2.3  Bus2=812.1.2.3  LineCode=300  Length=37.5   units=kft'
        dssText.Command= 'New Line.L6     Phases=3 Bus1=812.1.2.3  Bus2=814.1.2.3  LineCode=300  Length=29.73   units=kft'
        dssText.Command= 'New Line.L7     Phases=3 Bus1=814r.1.2.3 Bus2=850.1.2.3  LineCode=301  Length=0.01   units=kft'
        dssText.Command= 'New Line.L8     Phases=1 Bus1=816.1      Bus2=818.1      LineCode=302  Length=1.71   units=kft'
        dssText.Command= 'New Line.L9     Phases=3 Bus1=816.1.2.3  Bus2=824.1.2.3  LineCode=301  Length=10.21   units=kft'
        dssText.Command= 'New Line.L10    Phases=1 Bus1=818.1      Bus2=820.1      LineCode=302  Length=48.15   units=kft'
        dssText.Command= 'New Line.L11    Phases=1 Bus1=820.1      Bus2=822.1      LineCode=302  Length=13.74   units=kft'
        dssText.Command= 'New Line.L12    Phases=1 Bus1=824.2      Bus2=826.2      LineCode=303  Length=3.03   units=kft'
        dssText.Command= 'New Line.L13    Phases=3 Bus1=824.1.2.3  Bus2=828.1.2.3  LineCode=301  Length=0.84   units=kft'
        dssText.Command= 'New Line.L14    Phases=3 Bus1=828.1.2.3  Bus2=830.1.2.3  LineCode=301  Length=20.44   units=kft'
        dssText.Command= 'New Line.L15    Phases=3 Bus1=830.1.2.3  Bus2=854.1.2.3  LineCode=301  Length=0.52   units=kft'
        dssText.Command= 'New Line.L16    Phases=3 Bus1=832.1.2.3  Bus2=858.1.2.3  LineCode=301  Length=4.9   units=kft'
        dssText.Command= 'New Line.L17    Phases=3 Bus1=834.1.2.3  Bus2=860.1.2.3  LineCode=301  Length=2.02   units=kft'
        dssText.Command= 'New Line.L18    Phases=3 Bus1=834.1.2.3  Bus2=842.1.2.3  LineCode=301  Length=0.28   units=kft'
        dssText.Command= 'New Line.L19    Phases=3 Bus1=836.1.2.3  Bus2=840.1.2.3  LineCode=301  Length=0.86   units=kft'
        dssText.Command= 'New Line.L20    Phases=3 Bus1=836.1.2.3  Bus2=862.1.2.3  LineCode=301  Length=0.28   units=kft'
        dssText.Command= 'New Line.L21    Phases=3 Bus1=842.1.2.3  Bus2=844.1.2.3  LineCode=301  Length=1.35   units=kft'
        dssText.Command= 'New Line.L22    Phases=3 Bus1=844.1.2.3  Bus2=846.1.2.3  LineCode=301  Length=3.64   units=kft'
        dssText.Command= 'New Line.L23    Phases=3 Bus1=846.1.2.3  Bus2=848.1.2.3  LineCode=301  Length=0.53   units=kft'
        dssText.Command= 'New Line.L24    Phases=3 Bus1=850.1.2.3  Bus2=816.1.2.3  LineCode=301  Length=0.31   units=kft'
        dssText.Command= 'New Line.L25    Phases=3 Bus1=852r.1.2.3 Bus2=832.1.2.3  LineCode=301  Length=0.01   units=kft'

        dssText.Command= '! 24.9/4.16 kV  Transformer'
        dssText.Command= 'New Transformer.XFM1  Phases=3 Windings=2 Xhl=4.08'
        dssText.Command= '~ wdg=1 bus=832       conn=wye   kv=24.9  kva=500    %r=0.95'
        dssText.Command= '~ wdg=2 bus=888       conn=Wye   kv=4.16  kva=500    %r=0.95'

        dssText.Command= 'New Line.L26    Phases=1 Bus1=854.2      Bus2=856.2      LineCode=303  Length=23.33   units=kft'
        dssText.Command= 'New Line.L27    Phases=3 Bus1=854.1.2.3  Bus2=852.1.2.3  LineCode=301  Length=36.83   units=kft'
        dssText.Command= '! 9-17-10 858-864 changed to phase A per error report'
        dssText.Command= 'New Line.L28    Phases=1 Bus1=858.1      Bus2=864.1      LineCode=303  Length=1.62   units=kft'
        dssText.Command= 'New Line.L29    Phases=3 Bus1=858.1.2.3  Bus2=834.1.2.3  LineCode=301  Length=5.83   units=kft'
        dssText.Command= 'New Line.L30    Phases=3 Bus1=860.1.2.3  Bus2=836.1.2.3  LineCode=301  Length=2.68   units=kft'
        dssText.Command= 'New Line.L31    Phases=1 Bus1=862.2      Bus2=838.2      LineCode=304  Length=4.86   units=kft'
        dssText.Command= 'New Line.L32    Phases=3 Bus1=888.1.2.3  Bus2=890.1.2.3  LineCode=300  Length=10.56   units=kft'
        dssText.Command= '! Capacitors'

        cap844  =  300 *self.Load[self.time]
        cap848  =  450 *self.Load[self.time]
        dssText.Command= 'New Capacitor.C844      Bus1=844        Phases=3        kVAR= %s        kV=24.9'%(cap844)
        dssText.Command= 'New Capacitor.C848      Bus1=848        Phases=3        kVAR= %s        kV=24.9'%(cap848)
        dssText.Command= "new transformer.reg1a phases=1 windings=2 buses=(814.1 814r.1) conns='wye wye' "
        dssText.Command= '~ kvs="14.376 14.376" kvas="20000 20000" XHL=1'
        dssText.Command= 'new regcontrol.creg1a transformer=reg1a winding=2 vreg=122 band=2 ptratio=120 ctprim=100 R=2.7 X=1.6'
        dssText.Command= "new transformer.reg1b phases=1 windings=2 buses=(814.2 814r.2) conns='wye wye'"
        dssText.Command= '~ kvs="14.376 14.376" kvas="20000 20000" XHL=1'
        dssText.Command= 'new regcontrol.creg1b transformer=reg1b winding=2 vreg=122 band=2 ptratio=120 ctprim=100 R=2.7 X=1.6'
        dssText.Command= "new transformer.reg1c phases=1 windings=2 buses=(814.3 814r.3) conns='wye wye'"
        dssText.Command= '~ kvs="14.376 14.376" kvas="20000 20000" XHL=1'
        dssText.Command= 'new regcontrol.creg1c transformer=reg1c winding=2 vreg=122 band=2 ptratio=120 ctprim=100 R=2.7 X=1.6'

        dssText.Command= "new transformer.reg2a phases=1 windings=2 buses=(852.1 852r.1) conns='wye wye'"
        dssText.Command= '~ kvs="14.376 14.376" kvas="20000 20000" XHL=1'
        dssText.Command= 'new regcontrol.creg2a transformer=reg2a winding=2 vreg=124 band=2 ptratio=120 ctprim=100 R=2.5 X=1.5'
        dssText.Command= "new transformer.reg2b phases=1 windings=2 buses=(852.2 852r.2) conns='wye wye'"
        dssText.Command= '~ kvs="14.376 14.376" kvas="20000 20000" XHL=1'
        dssText.Command= 'new regcontrol.creg2b transformer=reg2b winding=2 vreg=124 band=2 ptratio=120 ctprim=100 R=2.5 X=1.5'
        dssText.Command= "new transformer.reg2c phases=1 windings=2 buses=(852.3 852r.3) conns='wye wye'"
        dssText.Command= '~ kvs="14.376 14.376" kvas="20000 20000" XHL=1'
        dssText.Command= 'new regcontrol.creg2c transformer=reg2c winding=2 vreg=124 band=2 ptratio=120 ctprim=100 R=2.5 X=1.5'

        LoadS860kw   =  60 *self.Load[self.time]
        LoadS860kvar =  48 *self.Load[self.time]
        LoadS840kw   =  27 *self.Load[self.time]
        LoadS840kvar =  21 *self.Load[self.time]
        LoadS844kw   =  405 *self.Load[self.time]
        LoadS844kvar =  315 *self.Load[self.time]
        LoadS848kw   =  60 *self.Load[self.time]
        LoadS848var  =  48 *self.Load[self.time]
        LoadS830akw   = 10 *self.Load[self.time]
        LoadS830akvar = 5  *self.Load[self.time]
        LoadS830bkw   = 10 *self.Load[self.time]
        LoadS830bkvar = 5  *self.Load[self.time]
        LoadS830ckw   = 25 *self.Load[self.time]
        LoadS830ckvar = 10 *self.Load[self.time]
        LoadS890kw   =  450 *self.Load[self.time]
        LoadS890kvar =  225 *self.Load[self.time]
        # spot loads
        dssText.Command= 'New Load.S860       Bus1=860   Phases=3 Conn=Wye   Model=1 kV= 24.900   kW= %s kVAR= %s'%(LoadS860kw,LoadS860kvar)
        dssText.Command= 'New Load.S840       Bus1=840   Phases=3 Conn=Wye   Model=5 kV= 24.900   kW= %s kVAR= %s'%(LoadS840kw,LoadS840kvar)
        dssText.Command= 'New Load.S844       Bus1=844   Phases=3 Conn=Wye   Model=2 kV= 24.900   kW= %s kVAR= %s'%(LoadS844kw,LoadS844kvar)
        dssText.Command= 'New Load.S848       Bus1=848   Phases=3 Conn=Delta Model=1 kV= 24.900   kW= %s kVAR= %s'%(LoadS848kw,LoadS848var)
        dssText.Command= 'New Load.S830a      Bus1=830.1.2 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadS830akw,LoadS830akvar)
        dssText.Command= 'New Load.S830b      Bus1=830.2.3 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadS830bkw,LoadS830bkvar)
        dssText.Command= 'New Load.S830c      Bus1=830.3.1 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadS830ckw,LoadS830ckvar)
        dssText.Command= 'New Load.S890       Bus1=890   Phases=3 Conn=Delta Model=5 kV=  4.160   kW= %s kVAR= %s'%(LoadS890kw,LoadS890kvar)


        LoadD802_806bkw   =  15 *self.Load[self.time]
        LoadD802_806bkvar = 7.5 *self.Load[self.time]        
        LoadD802_806ckw   = 12.5*self.Load[self.time]
        LoadD802_806ckvar = 7.0 *self.Load[self.time]

        LoadD808_810bkw   = 8 *self.Load[self.time]
        LoadD808_810bkvar = 4 *self.Load[self.time]

        LoadD818_820akw   = 17  *self.Load[self.time]
        LoadD818_820akvar = 8.5 *self.Load[self.time]

        LoadD820_822akw   = 67.5*self.Load[self.time]
        LoadD820_822akvar = 35  *self.Load[self.time]   

        LoadD816_824bkw   = 2.5*self.Load[self.time]
        LoadD816_824bkvar = 1  *self.Load[self.time]  

        LoadD824_826bkw   = 20 *self.Load[self.time]
        LoadD824_826bkvar = 10 *self.Load[self.time]   
        
        LoadD824_828ckw   = 2  *self.Load[self.time]
        LoadD824_828ckvar = 1  *self.Load[self.time]   
        
        LoadD828_830akw   = 3.5 *self.Load[self.time]
        LoadD828_830akvar = 1.5 *self.Load[self.time]   
        
        LoadD854_856bkw   = 2  *self.Load[self.time]
        LoadD854_856bkvar = 1  *self.Load[self.time]   
        
        LoadD832_858akw   = 3.5 *self.Load[self.time]
        LoadD832_858akvar = 1.5 *self.Load[self.time]           
        LoadD832_858bkw   = 1.0 *self.Load[self.time]
        LoadD832_858bkvar = 0.5 *self.Load[self.time]           
        LoadD832_858ckw   = 3   *self.Load[self.time]
        LoadD832_858ckvar = 1.5 *self.Load[self.time]   
        
        # distributed loads
        dssText.Command= 'New Load.D802_806sb Bus1=802.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD802_806bkw,LoadD802_806bkvar)
        dssText.Command= 'New Load.D802_806rb Bus1=806.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD802_806bkw,LoadD802_806bkvar)
        dssText.Command= 'New Load.D802_806sc Bus1=802.3 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD802_806ckw,LoadD802_806ckvar)
        dssText.Command= 'New Load.D802_806rc Bus1=806.3 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD802_806ckw,LoadD802_806ckvar)

        dssText.Command= 'New Load.D808_810sb Bus1=808.2 Phases=1 Conn=Wye   Model=4 kV= 14.376 kW= %s  kVAR= %s '%(LoadD808_810bkw,LoadD808_810bkvar)
        dssText.Command= 'New Load.D808_810rb Bus1=810.2 Phases=1 Conn=Wye   Model=4 kV= 14.376 kW= %s  kVAR= %s '%(LoadD808_810bkw,LoadD808_810bkvar)

        dssText.Command= 'New Load.D818_820sa Bus1=818.1 Phases=1 Conn=Wye   Model=2 kV= 14.376 kW= %s  kVAR= %s '%(LoadD818_820akw,LoadD818_820akvar)
        dssText.Command= 'New Load.D818_820ra Bus1=820.1 Phases=1 Conn=Wye   Model=2 kV= 14.376 kW= %s  kVAR= %s '%(LoadD818_820akw,LoadD818_820akvar)

        dssText.Command= 'New Load.D820_822sa Bus1=820.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD820_822akw,LoadD820_822akvar)
        dssText.Command= 'New Load.D820_822ra Bus1=822.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD820_822akw,LoadD820_822akvar)

        dssText.Command= 'New Load.D816_824sb Bus1=816.2 Phases=1 Conn=Delta Model=5 kV= 24.900 kW= %s  kVAR= %s '%(LoadD816_824bkw,LoadD816_824bkvar)
        dssText.Command= 'New Load.D816_824rb Bus1=824.2 Phases=1 Conn=Delta Model=5 kV= 24.900 kW= %s  kVAR= %s '%(LoadD816_824bkw,LoadD816_824bkvar)

        dssText.Command= 'New Load.D824_826sb Bus1=824.2 Phases=1 Conn=Wye   Model=5 kV= 14.376 kW= %s  kVAR= %s '%(LoadD824_826bkw,LoadD824_826bkvar)
        dssText.Command= 'New Load.D824_826rb Bus1=826.2 Phases=1 Conn=Wye   Model=5 kV= 14.376 kW= %s  kVAR= %s '%(LoadD824_826bkw,LoadD824_826bkvar)

        dssText.Command= 'New Load.D824_828sc Bus1=824.3 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD824_828ckw,LoadD824_828ckvar)
        dssText.Command= 'New Load.D824_828rc Bus1=828.3 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD824_828ckw,LoadD824_828ckvar)

        dssText.Command= 'New Load.D828_830sa Bus1=828.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD828_830akw,LoadD828_830akvar)
        dssText.Command= 'New Load.D828_830ra Bus1=830.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD828_830akw,LoadD828_830akvar)
        dssText.Command= 'New Load.D854_856sb Bus1=854.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD854_856bkw,LoadD854_856bkvar)
        dssText.Command= 'New Load.D854_856rb Bus1=856.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD854_856bkw,LoadD854_856bkvar)

        dssText.Command= 'New Load.D832_858sa Bus1=832.1 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s  kVAR= %s '%(LoadD832_858akw,LoadD832_858akvar)
        dssText.Command= 'New Load.D832_858ra Bus1=858.1 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s  kVAR= %s '%(LoadD832_858akw,LoadD832_858akvar)
        dssText.Command= 'New Load.D832_858sb Bus1=832.2 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s  kVAR= %s '%(LoadD832_858bkw,LoadD832_858bkvar)
        dssText.Command= 'New Load.D832_858rb Bus1=858.2 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s  kVAR= %s '%(LoadD832_858bkw,LoadD832_858bkvar)
        dssText.Command= 'New Load.D832_858sc Bus1=832.3 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s  kVAR= %s '%(LoadD832_858ckw,LoadD832_858ckvar)
        dssText.Command= 'New Load.D832_858rc Bus1=858.3 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s  kVAR= %s '%(LoadD832_858ckw,LoadD832_858ckvar)
        
        LoadD858_864bkw   = 1   *self.Load[self.time]
        LoadD858_864bkvar = 0.5 *self.Load[self.time]   
        
        LoadD858_834akw   = 2   *self.Load[self.time]
        LoadD858_834akvar = 1   *self.Load[self.time]           
        LoadD858_834bkw   = 7.5 *self.Load[self.time]
        LoadD858_834bkvar = 4   *self.Load[self.time]          
        LoadD858_834ckw   = 6.5 *self.Load[self.time]
        LoadD858_834ckvar = 3.5 *self.Load[self.time]   
        
        LoadD834_860akw   = 8   *self.Load[self.time]
        LoadD834_860akvar = 4   *self.Load[self.time]           
        LoadD834_860bkw   = 10  *self.Load[self.time]
        LoadD834_860bkvar =  5  *self.Load[self.time]            
        LoadD834_860ckw   = 55  *self.Load[self.time]
        LoadD834_860ckvar = 27.5*self.Load[self.time]   
        
        LoadD860_836akw   = 15  *self.Load[self.time]
        LoadD860_836akvar = 7.5 *self.Load[self.time]            
        LoadD860_836bkw   = 5   *self.Load[self.time]
        LoadD860_836bkvar = 3   *self.Load[self.time]           
        LoadD860_836ckw   = 21  *self.Load[self.time]
        LoadD860_836ckvar = 11  *self.Load[self.time]    
        
        LoadD836_840akw   = 9   *self.Load[self.time]
        LoadD836_840akvar = 4.5 *self.Load[self.time]           
        LoadD836_840bkw   = 11  *self.Load[self.time]
        LoadD836_840bkvar = 5.5 *self.Load[self.time]    
        
        LoadD862_838bkw   = 14  *self.Load[self.time]
        LoadD862_838bkvar = 7   *self.Load[self.time]   
        
        LoadD842_844akw   = 4.5 *self.Load[self.time]
        LoadD842_844akvar = 2.5 *self.Load[self.time]    
        
        LoadD844_846bkw   = 12.5*self.Load[self.time]
        LoadD844_846bkvar = 6   *self.Load[self.time]          
        LoadD844_846ckw   = 10  *self.Load[self.time]
        LoadD844_846ckvar = 5.5 *self.Load[self.time]    
        
        LoadD846_848bkw   = 11.5*self.Load[self.time]
        LoadD846_848bkvar = 5.5 *self.Load[self.time]  

        dssText.Command= 'New Load.D858_864sb Bus1=858.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD858_864bkw,LoadD858_864bkvar)
        dssText.Command= 'New Load.D858_864rb Bus1=864.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD858_864bkw,LoadD858_864bkvar)

        dssText.Command= 'New Load.D858_834sa Bus1=858.1.2 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD858_834akw,LoadD858_834akvar)
        dssText.Command= 'New Load.D858_834ra Bus1=834.1.2 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD858_834akw,LoadD858_834akvar)
        dssText.Command= 'New Load.D858_834sb Bus1=858.2.3 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD858_834bkw,LoadD858_834bkvar)
        dssText.Command= 'New Load.D858_834rb Bus1=834.2.3 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD858_834bkw,LoadD858_834bkvar)
        dssText.Command= 'New Load.D858_834sc Bus1=858.3.1 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD858_834ckw,LoadD858_834ckvar)
        dssText.Command= 'New Load.D858_834rc Bus1=834.3.1 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD858_834ckw,LoadD858_834ckvar)

        dssText.Command= 'New Load.D834_860sa Bus1=834.1.2 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadD834_860akw,LoadD834_860akvar)
        dssText.Command= 'New Load.D834_860ra Bus1=860.1.2 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadD834_860akw,LoadD834_860akvar)
        dssText.Command= 'New Load.D834_860sb Bus1=834.2.3 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadD834_860bkw,LoadD834_860bkvar)
        dssText.Command= 'New Load.D834_860rb Bus1=860.2.3 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadD834_860bkw,LoadD834_860bkvar)
        dssText.Command= 'New Load.D834_860sc Bus1=834.3.1 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadD834_860ckw,LoadD834_860ckvar)
        dssText.Command= 'New Load.D834_860rc Bus1=860.3.1 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadD834_860ckw,LoadD834_860ckvar)

        dssText.Command= 'New Load.D860_836sa Bus1=860.1.2 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD860_836akw,LoadD860_836akvar)
        dssText.Command= 'New Load.D860_836ra Bus1=836.1.2 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD860_836akw,LoadD860_836akvar)
        dssText.Command= 'New Load.D860_836sb Bus1=860.2.3 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD860_836bkw,LoadD860_836bkvar)
        dssText.Command= 'New Load.D860_836rb Bus1=836.2.3 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD860_836bkw,LoadD860_836bkvar)
        dssText.Command= 'New Load.D860_836sc Bus1=860.3.1 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD860_836ckw,LoadD860_836ckvar)
        dssText.Command= 'New Load.D860_836rc Bus1=836.3.1 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD860_836ckw,LoadD860_836ckvar)

        dssText.Command= 'New Load.D836_840sa Bus1=836.1.2 Phases=1 Conn=Delta Model=5 kV= 24.900 kW= %s kVAR= %s'%(LoadD836_840akw,LoadD836_840akvar)
        dssText.Command= 'New Load.D836_840ra Bus1=840.1.2 Phases=1 Conn=Delta Model=5 kV= 24.900 kW= %s kVAR= %s'%(LoadD836_840akw,LoadD836_840akvar)
        dssText.Command= 'New Load.D836_840sb Bus1=836.2.3 Phases=1 Conn=Delta Model=5 kV= 24.900 kW= %s kVAR= %s'%(LoadD836_840bkw,LoadD836_840bkvar)
        dssText.Command= 'New Load.D836_840rb Bus1=840.2.3 Phases=1 Conn=Delta Model=5 kV= 24.900 kW= %s kVAR= %s'%(LoadD836_840bkw,LoadD836_840bkvar)

        dssText.Command= 'New Load.D862_838sb Bus1=862.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD862_838bkw,LoadD862_838bkvar)
        dssText.Command= 'New Load.D862_838rb Bus1=838.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD862_838bkw,LoadD862_838bkvar)

        dssText.Command= 'New Load.D842_844sa Bus1=842.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD842_844akw,LoadD842_844akvar)
        dssText.Command= 'New Load.D842_844ra Bus1=844.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD842_844akw,LoadD842_844akvar)

        dssText.Command= 'New Load.D844_846sb Bus1=844.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD844_846bkw,LoadD844_846bkvar)
        dssText.Command= 'New Load.D844_846rb Bus1=846.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD844_846bkw,LoadD844_846bkvar)
        dssText.Command= 'New Load.D844_846sc Bus1=844.3 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD844_846ckw,LoadD844_846ckvar)
        dssText.Command= 'New Load.D844_846rc Bus1=846.3 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD844_846ckw,LoadD844_846ckvar)

        dssText.Command= 'New Load.D846_848sb Bus1=846.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD846_848bkw,LoadD846_848bkvar)
        dssText.Command= 'New Load.D846_848rb Bus1=848.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD846_848bkw,LoadD846_848bkvar)

        dssText.Command= 'Load.s860.vminpu=.85'
        dssText.Command= 'Load.s840.vminpu=.85'
        dssText.Command= 'Load.s844.vminpu=.85'
        dssText.Command= 'Load.s848.vminpu=.85'
        dssText.Command= 'Load.s830a.vminpu=.85'
        dssText.Command= 'Load.s830b.vminpu=.85'
        dssText.Command= 'Load.s830c.vminpu=.85'
        dssText.Command= 'Load.s890.vminpu=.85'
        dssText.Command= 'Load.d802_806sb.vminpu=.85'
        dssText.Command= 'Load.d802_806rb.vminpu=.85'
        dssText.Command= 'Load.d802_806sc.vminpu=.85'
        dssText.Command= 'Load.d802_806rc.vminpu=.85'
        dssText.Command= 'Load.d808_810sb.vminpu=.85'
        dssText.Command= 'Load.d808_810rb.vminpu=.85'
        dssText.Command= 'Load.d818_820sa.vminpu=.85'
        dssText.Command= 'Load.d818_820ra.vminpu=.85'
        dssText.Command= 'Load.d820_822sa.vminpu=.85'
        dssText.Command= 'Load.d820_822ra.vminpu=.85'
        dssText.Command= 'Load.d816_824sb.vminpu=.85'
        dssText.Command= 'Load.d816_824rb.vminpu=.85'
        dssText.Command= 'Load.d824_826sb.vminpu=.85'
        dssText.Command= 'Load.d824_826rb.vminpu=.85'
        dssText.Command= 'Load.d824_828sc.vminpu=.85'
        dssText.Command= 'Load.d824_828rc.vminpu=.85'
        dssText.Command= 'Load.d828_830sa.vminpu=.85'
        dssText.Command= 'Load.d828_830ra.vminpu=.85'
        dssText.Command= 'Load.d854_856sb.vminpu=.85'
        dssText.Command= 'Load.d854_856rb.vminpu=.85'
        dssText.Command= 'Load.d832_858sa.vminpu=.85'
        dssText.Command= 'Load.d832_858ra.vminpu=.85'
        dssText.Command= 'Load.d832_858sb.vminpu=.85'
        dssText.Command= 'Load.d832_858rb.vminpu=.85'
        dssText.Command= 'Load.d832_858sc.vminpu=.85'
        dssText.Command= 'Load.d832_858rc.vminpu=.85'
        dssText.Command= 'Load.d858_864sb.vminpu=.85'
        dssText.Command= 'Load.d858_864rb.vminpu=.85'
        dssText.Command= 'Load.d858_834sa.vminpu=.85'
        dssText.Command= 'Load.d858_834ra.vminpu=.85'
        dssText.Command= 'Load.d858_834sb.vminpu=.85'
        dssText.Command= 'Load.d858_834rb.vminpu=.85'
        dssText.Command= 'Load.d858_834sc.vminpu=.85'
        dssText.Command= 'Load.d858_834rc.vminpu=.85'
        dssText.Command= 'Load.d834_860sa.vminpu=.85'
        dssText.Command= 'Load.d834_860ra.vminpu=.85'
        dssText.Command= 'Load.d834_860sb.vminpu=.85'
        dssText.Command= 'Load.d834_860rb.vminpu=.85'
        dssText.Command= 'Load.d834_860sc.vminpu=.85'
        dssText.Command= 'Load.d834_860rc.vminpu=.85'
        dssText.Command= 'Load.d860_836sa.vminpu=.85'
        dssText.Command= 'Load.d860_836ra.vminpu=.85'
        dssText.Command= 'Load.d860_836sb.vminpu=.85'
        dssText.Command= 'Load.d860_836rb.vminpu=.85'
        dssText.Command= 'Load.d860_836sc.vminpu=.85'
        dssText.Command= 'Load.d860_836rc.vminpu=.85'
        dssText.Command= 'Load.d836_840sa.vminpu=.85'
        dssText.Command= 'Load.d836_840ra.vminpu=.85'
        dssText.Command= 'Load.d836_840sb.vminpu=.85'
        dssText.Command= 'Load.d836_840rb.vminpu=.85'
        dssText.Command= 'Load.d862_838sb.vminpu=.85'
        dssText.Command= 'Load.d862_838rb.vminpu=.85'
        dssText.Command= 'Load.d842_844sa.vminpu=.85'
        dssText.Command= 'Load.d842_844ra.vminpu=.85'
        dssText.Command= 'Load.d844_846sb.vminpu=.85'
        dssText.Command= 'Load.d844_846rb.vminpu=.85'
        dssText.Command= 'Load.d844_846sc.vminpu=.85'
        dssText.Command= 'Load.d844_846rc.vminpu=.85'
        dssText.Command= 'Load.d846_848sb.vminpu=.85'
        dssText.Command= 'Load.d846_848rb.vminpu=.85'


        dssText.Command='new XYCurve.Eff npts=4 xarray=[.1 .2 .4 1.0] yarray=[1 1 1 1]'

        a = self.PV[self.time]
        if a>0.2 :
            Qpu = 1
        else :
            Qpu = a/0.2
        act1 = 14
        act2 = 14
        act3 = 14
        act4 = 14
        act5 = 14
        act6 = 14
        act7 = 14
        act8 = 14
        act9 = 14
        act10 = 14
        act11 = 14
        act12 = 14

        #830
        dssText.Command="New PVSystem.PVgen830a Phases=1 Bus1=830.1 Pmpp= 100 Irradiance=%s kV=14.376 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=wye EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
        dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
        dssText.Command='New XYcurve.vvc830a npts=6 yarray=[%f %f 0 0 -%f -%f] xarray=[ 0.1 %f %f %f %f 1.5]'%(Qpu,Qpu,Qpu,Qpu,self.action[act1][0],self.action[act1][1],self.action[act1][2],self.action[act1][3])
        dssText.Command='New InvControl.VoltVar830a DERList=PVSystem.PVgen830a mode=VOLTVAR voltage_curvex_ref=rated vvc_curve1=vvc830a VoltageChangeTolerance=0.01 VarChangeTolerance=1 RefReactivePower=VAraval'
        #830
        dssText.Command="New PVSystem.PVgen830b Phases=1 Bus1=830.2 Pmpp= 100 Irradiance=%s kV=14.376 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=wye EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
        dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
        dssText.Command='New XYcurve.vvc830b npts=6 yarray=[%f %f 0 0 -%f -%f] xarray=[ 0.1 %f %f %f %f 1.5]'%(Qpu,Qpu,Qpu,Qpu,self.action[act2][0],self.action[act2][1],self.action[act2][2],self.action[act2][3])
        dssText.Command='New InvControl.VoltVar830b DERList=PVSystem.PVgen830b mode=VOLTVAR voltage_curvex_ref=rated vvc_curve1=vvc830b VoltageChangeTolerance=0.01 VarChangeTolerance=1 RefReactivePower=VAraval'
        #830
        dssText.Command="New PVSystem.PVgen830c Phases=1 Bus1=830.3 Pmpp= 100 Irradiance=%s kV=14.376 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=wye EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
        dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
        dssText.Command='New XYcurve.vvc830c npts=6 yarray=[%f %f 0 0 -%f -%f] xarray=[ 0.1 %f %f %f %f 1.5]'%(Qpu,Qpu,Qpu,Qpu,self.action[act3][0],self.action[act3][1],self.action[act3][2],self.action[act3][3])
        dssText.Command='New InvControl.VoltVar830c DERList=PVSystem.PVgen830c mode=VOLTVAR voltage_curvex_ref=rated vvc_curve1=vvc830c VoltageChangeTolerance=0.01 VarChangeTolerance=1 RefReactivePower=VAraval'

        #840
        dssText.Command="New PVSystem.PVgen840a Phases=1 Bus1=840.1.2 Pmpp= 100 Irradiance=%s kV=24.9 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=delta EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
        dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
        dssText.Command='New XYcurve.vvc840a npts=6 yarray=[%f %f 0 0 -%f -%f] xarray=[ 0.1 %f %f %f %f 1.5]'%(Qpu,Qpu,Qpu,Qpu,self.action[act4][0],self.action[act4][1],self.action[act4][2],self.action[act4][3])
        dssText.Command='New InvControl.VoltVar840a DERList=PVSystem.PVgen840a mode=VOLTVAR voltage_curvex_ref=rated vvc_curve1=vvc840a VoltageChangeTolerance=0.01 VarChangeTolerance=1 RefReactivePower=VAraval'
        #840
        dssText.Command="New PVSystem.PVgen840b Phases=1 Bus1=840.2.3 Pmpp= 100 Irradiance=%s kV=24.9 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=delta EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
        dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
        dssText.Command='New XYcurve.vvc840b npts=6 yarray=[%f %f 0 0 -%f -%f] xarray=[ 0.1 %f %f %f %f 1.5]'%(Qpu,Qpu,Qpu,Qpu,self.action[act5][0],self.action[act5][1],self.action[act5][2],self.action[act5][3])
        dssText.Command='New InvControl.VoltVar840b DERList=PVSystem.PVgen840b mode=VOLTVAR voltage_curvex_ref=rated vvc_curve1=vvc840b VoltageChangeTolerance=0.01 VarChangeTolerance=1 RefReactivePower=VAraval'
        #840
        dssText.Command="New PVSystem.PVgen840c Phases=1 Bus1=840.3.1 Pmpp= 100 Irradiance=%s kV=24.9 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=delta EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
        dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
        dssText.Command='New XYcurve.vvc840c npts=6 yarray=[%f %f 0 0 -%f -%f] xarray=[ 0.1 %f %f %f %f 1.5]'%(Qpu,Qpu,Qpu,Qpu,self.action[act6][0],self.action[act6][1],self.action[act6][2],self.action[act6][3])
        dssText.Command='New InvControl.VoltVar840c DERList=PVSystem.PVgen840c mode=VOLTVAR voltage_curvex_ref=rated vvc_curve1=vvc840c VoltageChangeTolerance=0.01 VarChangeTolerance=1 RefReactivePower=VAraval'        
 
        #890
        dssText.Command="New PVSystem.PVgen890a Phases=1 Bus1=890.1.2 Pmpp= 100 Irradiance=%s kV=4.16 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=delta EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
        dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
        dssText.Command='New XYcurve.vvc890a npts=6 yarray=[%f %f 0 0 -%f -%f] xarray=[ 0.1 %f %f %f %f 1.5]'%(Qpu,Qpu,Qpu,Qpu,self.action[act7][0],self.action[act7][1],self.action[act7][2],self.action[act7][3])
        dssText.Command='New InvControl.VoltVar890a DERList=PVSystem.PVgen890a mode=VOLTVAR voltage_curvex_ref=rated vvc_curve1=vvc890a VoltageChangeTolerance=0.01 VarChangeTolerance=1 RefReactivePower=VAraval'

        dssText.Command="New PVSystem.PVgen890b Phases=1 Bus1=890.2.3 Pmpp= 100 Irradiance=%s kV=4.16 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=delta EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
        dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
        dssText.Command='New XYcurve.vvc890b npts=6 yarray=[%f %f 0 0 -%f -%f] xarray=[ 0.1 %f %f %f %f 1.5]'%(Qpu,Qpu,Qpu,Qpu,self.action[act8][0],self.action[act8][1],self.action[act8][2],self.action[act8][3])
        dssText.Command='New InvControl.VoltVar890b DERList=PVSystem.PVgen890b mode=VOLTVAR voltage_curvex_ref=rated vvc_curve1=vvc890b VoltageChangeTolerance=0.01 VarChangeTolerance=1 RefReactivePower=VAraval'

        dssText.Command="New PVSystem.PVgen890c Phases=1 Bus1=890.3.1 Pmpp= 100 Irradiance=%s kV=4.16 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=delta EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
        dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
        dssText.Command='New XYcurve.vvc890c npts=6 yarray=[%f %f 0 0 -%f -%f] xarray=[ 0.1 %f %f %f %f 1.5]'%(Qpu,Qpu,Qpu,Qpu,self.action[act9][0],self.action[act9][1],self.action[act9][2],self.action[act9][3])
        dssText.Command='New InvControl.VoltVar890c DERList=PVSystem.PVgen890c mode=VOLTVAR voltage_curvex_ref=rated vvc_curve1=vvc890c VoltageChangeTolerance=0.01 VarChangeTolerance=1 RefReactivePower=VAraval'

        dssText.Command= 'Set Voltagebases= "69,24.9,4.16, .48"'

        dssText.Command= 'calcv'

        dssText.Command= 'Set maxiterations = 300'
        dssText.Command= 'Solve'

        Vabc_pu=dssCircuit.AllBusVolts
        Va_pu=dssCircuit.AllNodeVmagPUByPhase(1)    # phase a voltage pu
        Vb_pu=dssCircuit.AllNodeVmagPUByPhase(2)    # phase a voltage pu
        Vc_pu=dssCircuit.AllNodeVmagPUByPhase(3)

        state830a = np.hstack([a, self.Load[self.time] ]).flatten()
        state830b = np.hstack([a, self.Load[self.time] ]).flatten()
        state830c = np.hstack([a, self.Load[self.time] ]).flatten()
        state840a = np.hstack([a, self.Load[self.time] ]).flatten()
        state840b = np.hstack([a, self.Load[self.time] ]).flatten()
        state840c = np.hstack([a, self.Load[self.time] ]).flatten()
        state890a = np.hstack([a, self.Load[self.time] ]).flatten()
        state890b = np.hstack([a, self.Load[self.time] ]).flatten()
        state890c = np.hstack([a, self.Load[self.time] ]).flatten()

        states = [state830a, state830b, state830c, state840a, state840b, state840c, state890a, state890b, state890c]

        return states 

    def step(self, acts):        
        a = acts

        self.rwd = np.zeros(shape=(9,))
        terminal = False     
             
        n_state, rwd, ifo ,infoV = self._step(a)

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

        return next_state, total_reward, terminals, ifo ,infoV

    def fixeddroop(self,i,pv):
        dssObj = win32com.client.Dispatch('OpenDSSEngine.DSS')
        dssText = dssObj.Text 
        dssCircuit = dssObj.ActiveCircuit 
        dssSolution = dssCircuit.Solution 
        dssElem = dssCircuit.ActiveCktElement 
        dssBus = dssCircuit.ActiveBus
        dssText.Command= 'Clear'
        dssText.Command= 'new circuit.IEEE34'
        dssText.Command= '~ basekv=69 pu=1.05 angle=30 mvasc3=200000'

        dssText.Command= 'New Transformer.SubXF Phases=3 Windings=2 Xhl=0.01    ! normally 8'
        dssText.Command= '~ wdg=1 bus=sourcebus conn=Delta kv=69    kva=25000   %r=0.0005   !reduce %r, too'
        dssText.Command= '~ wdg=2 bus=800       conn=wye   kv=24.9  kva=25000   %r=0.0005'

        dssText.Command= 'Redirect        IEEELineCodes.dss   ! revised according to Later test feeder doc'
        dssText.Command= 'New Line.L1     Phases=3 Bus1=800.1.2.3  Bus2=802.1.2.3  LineCode=300  Length=2.58   units=kft'
        dssText.Command= 'New Line.L2     Phases=3 Bus1=802.1.2.3  Bus2=806.1.2.3  LineCode=300  Length=1.73   units=kft'
        dssText.Command= 'New Line.L3     Phases=3 Bus1=806.1.2.3  Bus2=808.1.2.3  LineCode=300  Length=32.23   units=kft'
        dssText.Command= 'New Line.L4     Phases=1 Bus1=808.2      Bus2=810.2      LineCode=303  Length=5.804   units=kft'
        dssText.Command= 'New Line.L5     Phases=3 Bus1=808.1.2.3  Bus2=812.1.2.3  LineCode=300  Length=37.5   units=kft'
        dssText.Command= 'New Line.L6     Phases=3 Bus1=812.1.2.3  Bus2=814.1.2.3  LineCode=300  Length=29.73   units=kft'
        dssText.Command= 'New Line.L7     Phases=3 Bus1=814r.1.2.3 Bus2=850.1.2.3  LineCode=301  Length=0.01   units=kft'
        dssText.Command= 'New Line.L8     Phases=1 Bus1=816.1      Bus2=818.1      LineCode=302  Length=1.71   units=kft'
        dssText.Command= 'New Line.L9     Phases=3 Bus1=816.1.2.3  Bus2=824.1.2.3  LineCode=301  Length=10.21   units=kft'
        dssText.Command= 'New Line.L10    Phases=1 Bus1=818.1      Bus2=820.1      LineCode=302  Length=48.15   units=kft'
        dssText.Command= 'New Line.L11    Phases=1 Bus1=820.1      Bus2=822.1      LineCode=302  Length=13.74   units=kft'
        dssText.Command= 'New Line.L12    Phases=1 Bus1=824.2      Bus2=826.2      LineCode=303  Length=3.03   units=kft'
        dssText.Command= 'New Line.L13    Phases=3 Bus1=824.1.2.3  Bus2=828.1.2.3  LineCode=301  Length=0.84   units=kft'
        dssText.Command= 'New Line.L14    Phases=3 Bus1=828.1.2.3  Bus2=830.1.2.3  LineCode=301  Length=20.44   units=kft'
        dssText.Command= 'New Line.L15    Phases=3 Bus1=830.1.2.3  Bus2=854.1.2.3  LineCode=301  Length=0.52   units=kft'
        dssText.Command= 'New Line.L16    Phases=3 Bus1=832.1.2.3  Bus2=858.1.2.3  LineCode=301  Length=4.9   units=kft'
        dssText.Command= 'New Line.L17    Phases=3 Bus1=834.1.2.3  Bus2=860.1.2.3  LineCode=301  Length=2.02   units=kft'
        dssText.Command= 'New Line.L18    Phases=3 Bus1=834.1.2.3  Bus2=842.1.2.3  LineCode=301  Length=0.28   units=kft'
        dssText.Command= 'New Line.L19    Phases=3 Bus1=836.1.2.3  Bus2=840.1.2.3  LineCode=301  Length=0.86   units=kft'
        dssText.Command= 'New Line.L20    Phases=3 Bus1=836.1.2.3  Bus2=862.1.2.3  LineCode=301  Length=0.28   units=kft'
        dssText.Command= 'New Line.L21    Phases=3 Bus1=842.1.2.3  Bus2=844.1.2.3  LineCode=301  Length=1.35   units=kft'
        dssText.Command= 'New Line.L22    Phases=3 Bus1=844.1.2.3  Bus2=846.1.2.3  LineCode=301  Length=3.64   units=kft'
        dssText.Command= 'New Line.L23    Phases=3 Bus1=846.1.2.3  Bus2=848.1.2.3  LineCode=301  Length=0.53   units=kft'
        dssText.Command= 'New Line.L24    Phases=3 Bus1=850.1.2.3  Bus2=816.1.2.3  LineCode=301  Length=0.31   units=kft'
        dssText.Command= 'New Line.L25    Phases=3 Bus1=852r.1.2.3 Bus2=832.1.2.3  LineCode=301  Length=0.01   units=kft'

        dssText.Command= '! 24.9/4.16 kV  Transformer'
        dssText.Command= 'New Transformer.XFM1  Phases=3 Windings=2 Xhl=4.08'
        dssText.Command= '~ wdg=1 bus=832       conn=wye   kv=24.9  kva=500    %r=0.95'
        dssText.Command= '~ wdg=2 bus=888       conn=Wye   kv=4.16  kva=500    %r=0.95'
        dssText.Command= '~ Tap = %s'%(self.tap7)

        dssText.Command= 'New Line.L26    Phases=1 Bus1=854.2      Bus2=856.2      LineCode=303  Length=23.33   units=kft'
        dssText.Command= 'New Line.L27    Phases=3 Bus1=854.1.2.3  Bus2=852.1.2.3  LineCode=301  Length=36.83   units=kft'
        dssText.Command= '! 9-17-10 858-864 changed to phase A per error report'
        dssText.Command= 'New Line.L28    Phases=1 Bus1=858.1      Bus2=864.1      LineCode=303  Length=1.62   units=kft'
        dssText.Command= 'New Line.L29    Phases=3 Bus1=858.1.2.3  Bus2=834.1.2.3  LineCode=301  Length=5.83   units=kft'
        dssText.Command= 'New Line.L30    Phases=3 Bus1=860.1.2.3  Bus2=836.1.2.3  LineCode=301  Length=2.68   units=kft'
        dssText.Command= 'New Line.L31    Phases=1 Bus1=862.2      Bus2=838.2      LineCode=304  Length=4.86   units=kft'
        dssText.Command= 'New Line.L32    Phases=3 Bus1=888.1.2.3  Bus2=890.1.2.3  LineCode=300  Length=10.56   units=kft'
        dssText.Command= '! Capacitors'

        cap844  =  300 *self.Load[self.time*6+i]
        cap848  =  450 *self.Load[self.time*6+i]
        dssText.Command= 'New Capacitor.C844      Bus1=844        Phases=3        kVAR= %s        kV=24.9'%(cap844)
        dssText.Command= 'New Capacitor.C848      Bus1=848        Phases=3        kVAR= %s        kV=24.9'%(cap848)

        dssText.Command= "new transformer.reg1a phases=1 windings=2 buses=(814.1 814r.1) conns='wye wye' "
        dssText.Command= '~ kvs="14.376 14.376" kvas="20000 20000" XHL=1'
        dssText.Command= 'new regcontrol.creg1a transformer=reg1a winding=2 vreg=122 band=2 ptratio=120 ctprim=100 R=2.7 X=1.6 TapNum = %s'%(self.tap1)

        dssText.Command= "new transformer.reg1b phases=1 windings=2 buses=(814.2 814r.2) conns='wye wye'"
        dssText.Command= '~ kvs="14.376 14.376" kvas="20000 20000" XHL=1'
        dssText.Command= 'new regcontrol.creg1b transformer=reg1b winding=2 vreg=122 band=2 ptratio=120 ctprim=100 R=2.7 X=1.6 TapNum = %s'%(self.tap2)

        dssText.Command= "new transformer.reg1c phases=1 windings=2 buses=(814.3 814r.3) conns='wye wye'"
        dssText.Command= '~ kvs="14.376 14.376" kvas="20000 20000" XHL=1'
        dssText.Command= 'new regcontrol.creg1c transformer=reg1c winding=2 vreg=122 band=2 ptratio=120 ctprim=100 R=2.7 X=1.6 TapNum = %s'%(self.tap3)

        dssText.Command= "new transformer.reg2a phases=1 windings=2 buses=(852.1 852r.1) conns='wye wye'"
        dssText.Command= '~ kvs="14.376 14.376" kvas="20000 20000" XHL=1'
        dssText.Command= 'new regcontrol.creg2a transformer=reg2a winding=2 vreg=124 band=2 ptratio=120 ctprim=100 R=2.5 X=1.5 TapNum = %s'%(self.tap4)
        dssText.Command= "new transformer.reg2b phases=1 windings=2 buses=(852.2 852r.2) conns='wye wye'"
        dssText.Command= '~ kvs="14.376 14.376" kvas="20000 20000" XHL=1'
        dssText.Command= 'new regcontrol.creg2b transformer=reg2b winding=2 vreg=124 band=2 ptratio=120 ctprim=100 R=2.5 X=1.5 TapNum = %s'%(self.tap5)
        dssText.Command= "new transformer.reg2c phases=1 windings=2 buses=(852.3 852r.3) conns='wye wye'"
        dssText.Command= '~ kvs="14.376 14.376" kvas="20000 20000" XHL=1'
        dssText.Command= 'new regcontrol.creg2c transformer=reg2c winding=2 vreg=124 band=2 ptratio=120 ctprim=100 R=2.5 X=1.5 TapNum = %s'%(self.tap6)

        LoadS860kw   =  60 *self.Load[self.time*6+i]
        LoadS860kvar =  48 *self.Load[self.time*6+i]
        LoadS840kw   =  27 *self.Load[self.time*6+i]
        LoadS840kvar =  21 *self.Load[self.time*6+i]
        LoadS844kw   =  405 *self.Load[self.time*6+i]
        LoadS844kvar =  315 *self.Load[self.time*6+i]
        LoadS848kw   =  60 *self.Load[self.time*6+i]
        LoadS848var  =  48 *self.Load[self.time*6+i]
        LoadS830akw   = 10 *self.Load[self.time*6+i]
        LoadS830akvar = 5  *self.Load[self.time*6+i]
        LoadS830bkw   = 10 *self.Load[self.time*6+i]
        LoadS830bkvar = 5  *self.Load[self.time*6+i]
        LoadS830ckw   = 25 *self.Load[self.time*6+i]
        LoadS830ckvar = 10 *self.Load[self.time*6+i]
        LoadS890kw   =  450 *self.Load[self.time*6+i]
        LoadS890kvar =  225 *self.Load[self.time*6+i]
        # spot loads
        dssText.Command= 'New Load.S860       Bus1=860   Phases=3 Conn=Wye   Model=1 kV= 24.900   kW= %s kVAR= %s'%(LoadS860kw,LoadS860kvar)
        dssText.Command= 'New Load.S840       Bus1=840   Phases=3 Conn=Wye   Model=5 kV= 24.900   kW= %s kVAR= %s'%(LoadS840kw,LoadS840kvar)
        dssText.Command= 'New Load.S844       Bus1=844   Phases=3 Conn=Wye   Model=2 kV= 24.900   kW= %s kVAR= %s'%(LoadS844kw,LoadS844kvar)
        dssText.Command= 'New Load.S848       Bus1=848   Phases=3 Conn=Delta Model=1 kV= 24.900   kW= %s kVAR= %s'%(LoadS848kw,LoadS848var)
        dssText.Command= 'New Load.S830a      Bus1=830.1.2 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadS830akw,LoadS830akvar)
        dssText.Command= 'New Load.S830b      Bus1=830.2.3 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadS830bkw,LoadS830bkvar)
        dssText.Command= 'New Load.S830c      Bus1=830.3.1 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadS830ckw,LoadS830ckvar)
        dssText.Command= 'New Load.S890       Bus1=890   Phases=3 Conn=Delta Model=5 kV=  4.160   kW= %s kVAR= %s'%(LoadS890kw,LoadS890kvar)


        LoadD802_806bkw   =  15 *self.Load[self.time*6+i]
        LoadD802_806bkvar = 7.5 *self.Load[self.time*6+i]        
        LoadD802_806ckw   = 12.5*self.Load[self.time*6+i]
        LoadD802_806ckvar = 7.0 *self.Load[self.time*6+i]

        LoadD808_810bkw   = 8 *self.Load[self.time*6+i]
        LoadD808_810bkvar = 4 *self.Load[self.time*6+i]

        LoadD818_820akw   = 17  *self.Load[self.time*6+i]
        LoadD818_820akvar = 8.5 *self.Load[self.time*6+i]

        LoadD820_822akw   = 67.5*self.Load[self.time*6+i]
        LoadD820_822akvar = 35  *self.Load[self.time*6+i]   

        LoadD816_824bkw   = 2.5*self.Load[self.time*6+i]
        LoadD816_824bkvar = 1  *self.Load[self.time*6+i]  

        LoadD824_826bkw   = 20 *self.Load[self.time*6+i]
        LoadD824_826bkvar = 10 *self.Load[self.time*6+i]   
        
        LoadD824_828ckw   = 2  *self.Load[self.time*6+i]
        LoadD824_828ckvar = 1  *self.Load[self.time*6+i]   
        
        LoadD828_830akw   = 3.5 *self.Load[self.time*6+i]
        LoadD828_830akvar = 1.5 *self.Load[self.time*6+i]   
        
        LoadD854_856bkw   = 2  *self.Load[self.time*6+i]
        LoadD854_856bkvar = 1  *self.Load[self.time*6+i]   
        
        LoadD832_858akw   = 3.5 *self.Load[self.time*6+i]
        LoadD832_858akvar = 1.5 *self.Load[self.time*6+i]           
        LoadD832_858bkw   = 1.0 *self.Load[self.time*6+i]
        LoadD832_858bkvar = 0.5 *self.Load[self.time*6+i]           
        LoadD832_858ckw   = 3   *self.Load[self.time*6+i]
        LoadD832_858ckvar = 1.5 *self.Load[self.time*6+i]   
        
        # distributed loads
        dssText.Command= 'New Load.D802_806sb Bus1=802.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD802_806bkw,LoadD802_806bkvar)
        dssText.Command= 'New Load.D802_806rb Bus1=806.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD802_806bkw,LoadD802_806bkvar)
        dssText.Command= 'New Load.D802_806sc Bus1=802.3 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD802_806ckw,LoadD802_806ckvar)
        dssText.Command= 'New Load.D802_806rc Bus1=806.3 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD802_806ckw,LoadD802_806ckvar)

        dssText.Command= 'New Load.D808_810sb Bus1=808.2 Phases=1 Conn=Wye   Model=4 kV= 14.376 kW= %s  kVAR= %s '%(LoadD808_810bkw,LoadD808_810bkvar)
        dssText.Command= 'New Load.D808_810rb Bus1=810.2 Phases=1 Conn=Wye   Model=4 kV= 14.376 kW= %s  kVAR= %s '%(LoadD808_810bkw,LoadD808_810bkvar)

        dssText.Command= 'New Load.D818_820sa Bus1=818.1 Phases=1 Conn=Wye   Model=2 kV= 14.376 kW= %s  kVAR= %s '%(LoadD818_820akw,LoadD818_820akvar)
        dssText.Command= 'New Load.D818_820ra Bus1=820.1 Phases=1 Conn=Wye   Model=2 kV= 14.376 kW= %s  kVAR= %s '%(LoadD818_820akw,LoadD818_820akvar)

        dssText.Command= 'New Load.D820_822sa Bus1=820.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD820_822akw,LoadD820_822akvar)
        dssText.Command= 'New Load.D820_822ra Bus1=822.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD820_822akw,LoadD820_822akvar)

        dssText.Command= 'New Load.D816_824sb Bus1=816.2 Phases=1 Conn=Delta Model=5 kV= 24.900 kW= %s  kVAR= %s '%(LoadD816_824bkw,LoadD816_824bkvar)
        dssText.Command= 'New Load.D816_824rb Bus1=824.2 Phases=1 Conn=Delta Model=5 kV= 24.900 kW= %s  kVAR= %s '%(LoadD816_824bkw,LoadD816_824bkvar)

        dssText.Command= 'New Load.D824_826sb Bus1=824.2 Phases=1 Conn=Wye   Model=5 kV= 14.376 kW= %s  kVAR= %s '%(LoadD824_826bkw,LoadD824_826bkvar)
        dssText.Command= 'New Load.D824_826rb Bus1=826.2 Phases=1 Conn=Wye   Model=5 kV= 14.376 kW= %s  kVAR= %s '%(LoadD824_826bkw,LoadD824_826bkvar)

        dssText.Command= 'New Load.D824_828sc Bus1=824.3 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD824_828ckw,LoadD824_828ckvar)
        dssText.Command= 'New Load.D824_828rc Bus1=828.3 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD824_828ckw,LoadD824_828ckvar)

        dssText.Command= 'New Load.D828_830sa Bus1=828.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD828_830akw,LoadD828_830akvar)
        dssText.Command= 'New Load.D828_830ra Bus1=830.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD828_830akw,LoadD828_830akvar)
        dssText.Command= 'New Load.D854_856sb Bus1=854.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD854_856bkw,LoadD854_856bkvar)
        dssText.Command= 'New Load.D854_856rb Bus1=856.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD854_856bkw,LoadD854_856bkvar)

        dssText.Command= 'New Load.D832_858sa Bus1=832.1 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s  kVAR= %s '%(LoadD832_858akw,LoadD832_858akvar)
        dssText.Command= 'New Load.D832_858ra Bus1=858.1 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s  kVAR= %s '%(LoadD832_858akw,LoadD832_858akvar)
        dssText.Command= 'New Load.D832_858sb Bus1=832.2 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s  kVAR= %s '%(LoadD832_858bkw,LoadD832_858bkvar)
        dssText.Command= 'New Load.D832_858rb Bus1=858.2 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s  kVAR= %s '%(LoadD832_858bkw,LoadD832_858bkvar)
        dssText.Command= 'New Load.D832_858sc Bus1=832.3 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s  kVAR= %s '%(LoadD832_858ckw,LoadD832_858ckvar)
        dssText.Command= 'New Load.D832_858rc Bus1=858.3 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s  kVAR= %s '%(LoadD832_858ckw,LoadD832_858ckvar)
        
        LoadD858_864bkw   = 1   *self.Load[self.time*6+i]
        LoadD858_864bkvar = 0.5 *self.Load[self.time*6+i]   
        
        LoadD858_834akw   = 2   *self.Load[self.time*6+i]
        LoadD858_834akvar = 1   *self.Load[self.time*6+i]           
        LoadD858_834bkw   = 7.5 *self.Load[self.time*6+i]
        LoadD858_834bkvar = 4   *self.Load[self.time*6+i]          
        LoadD858_834ckw   = 6.5 *self.Load[self.time*6+i]
        LoadD858_834ckvar = 3.5 *self.Load[self.time*6+i]   
        
        LoadD834_860akw   = 8   *self.Load[self.time*6+i]
        LoadD834_860akvar = 4   *self.Load[self.time*6+i]           
        LoadD834_860bkw   = 10  *self.Load[self.time*6+i]
        LoadD834_860bkvar =  5  *self.Load[self.time*6+i]            
        LoadD834_860ckw   = 55  *self.Load[self.time*6+i]
        LoadD834_860ckvar = 27.5*self.Load[self.time*6+i]   
        
        LoadD860_836akw   = 15  *self.Load[self.time*6+i]
        LoadD860_836akvar = 7.5 *self.Load[self.time*6+i]            
        LoadD860_836bkw   = 5   *self.Load[self.time*6+i]
        LoadD860_836bkvar = 3   *self.Load[self.time*6+i]           
        LoadD860_836ckw   = 21  *self.Load[self.time*6+i]
        LoadD860_836ckvar = 11  *self.Load[self.time*6+i]    
        
        LoadD836_840akw   = 9   *self.Load[self.time*6+i]
        LoadD836_840akvar = 4.5 *self.Load[self.time*6+i]           
        LoadD836_840bkw   = 11  *self.Load[self.time*6+i]
        LoadD836_840bkvar = 5.5 *self.Load[self.time*6+i]    
        
        LoadD862_838bkw   = 14  *self.Load[self.time*6+i]
        LoadD862_838bkvar = 7   *self.Load[self.time*6+i]   
        
        LoadD842_844akw   = 4.5 *self.Load[self.time*6+i]
        LoadD842_844akvar = 2.5 *self.Load[self.time*6+i]    
        
        LoadD844_846bkw   = 12.5*self.Load[self.time*6+i]
        LoadD844_846bkvar = 6   *self.Load[self.time*6+i]          
        LoadD844_846ckw   = 10  *self.Load[self.time*6+i]
        LoadD844_846ckvar = 5.5 *self.Load[self.time*6+i]    
        
        LoadD846_848bkw   = 11.5*self.Load[self.time*6+i]
        LoadD846_848bkvar = 5.5 *self.Load[self.time*6+i]  

        dssText.Command= 'New Load.D858_864sb Bus1=858.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD858_864bkw,LoadD858_864bkvar)
        dssText.Command= 'New Load.D858_864rb Bus1=864.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD858_864bkw,LoadD858_864bkvar)

        dssText.Command= 'New Load.D858_834sa Bus1=858.1.2 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD858_834akw,LoadD858_834akvar)
        dssText.Command= 'New Load.D858_834ra Bus1=834.1.2 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD858_834akw,LoadD858_834akvar)
        dssText.Command= 'New Load.D858_834sb Bus1=858.2.3 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD858_834bkw,LoadD858_834bkvar)
        dssText.Command= 'New Load.D858_834rb Bus1=834.2.3 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD858_834bkw,LoadD858_834bkvar)
        dssText.Command= 'New Load.D858_834sc Bus1=858.3.1 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD858_834ckw,LoadD858_834ckvar)
        dssText.Command= 'New Load.D858_834rc Bus1=834.3.1 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD858_834ckw,LoadD858_834ckvar)

        dssText.Command= 'New Load.D834_860sa Bus1=834.1.2 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadD834_860akw,LoadD834_860akvar)
        dssText.Command= 'New Load.D834_860ra Bus1=860.1.2 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadD834_860akw,LoadD834_860akvar)
        dssText.Command= 'New Load.D834_860sb Bus1=834.2.3 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadD834_860bkw,LoadD834_860bkvar)
        dssText.Command= 'New Load.D834_860rb Bus1=860.2.3 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadD834_860bkw,LoadD834_860bkvar)
        dssText.Command= 'New Load.D834_860sc Bus1=834.3.1 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadD834_860ckw,LoadD834_860ckvar)
        dssText.Command= 'New Load.D834_860rc Bus1=860.3.1 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadD834_860ckw,LoadD834_860ckvar)

        dssText.Command= 'New Load.D860_836sa Bus1=860.1.2 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD860_836akw,LoadD860_836akvar)
        dssText.Command= 'New Load.D860_836ra Bus1=836.1.2 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD860_836akw,LoadD860_836akvar)
        dssText.Command= 'New Load.D860_836sb Bus1=860.2.3 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD860_836bkw,LoadD860_836bkvar)
        dssText.Command= 'New Load.D860_836rb Bus1=836.2.3 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD860_836bkw,LoadD860_836bkvar)
        dssText.Command= 'New Load.D860_836sc Bus1=860.3.1 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD860_836ckw,LoadD860_836ckvar)
        dssText.Command= 'New Load.D860_836rc Bus1=836.3.1 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD860_836ckw,LoadD860_836ckvar)

        dssText.Command= 'New Load.D836_840sa Bus1=836.1.2 Phases=1 Conn=Delta Model=5 kV= 24.900 kW= %s kVAR= %s'%(LoadD836_840akw,LoadD836_840akvar)
        dssText.Command= 'New Load.D836_840ra Bus1=840.1.2 Phases=1 Conn=Delta Model=5 kV= 24.900 kW= %s kVAR= %s'%(LoadD836_840akw,LoadD836_840akvar)
        dssText.Command= 'New Load.D836_840sb Bus1=836.2.3 Phases=1 Conn=Delta Model=5 kV= 24.900 kW= %s kVAR= %s'%(LoadD836_840bkw,LoadD836_840bkvar)
        dssText.Command= 'New Load.D836_840rb Bus1=840.2.3 Phases=1 Conn=Delta Model=5 kV= 24.900 kW= %s kVAR= %s'%(LoadD836_840bkw,LoadD836_840bkvar)

        dssText.Command= 'New Load.D862_838sb Bus1=862.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD862_838bkw,LoadD862_838bkvar)
        dssText.Command= 'New Load.D862_838rb Bus1=838.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD862_838bkw,LoadD862_838bkvar)

        dssText.Command= 'New Load.D842_844sa Bus1=842.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD842_844akw,LoadD842_844akvar)
        dssText.Command= 'New Load.D842_844ra Bus1=844.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD842_844akw,LoadD842_844akvar)

        dssText.Command= 'New Load.D844_846sb Bus1=844.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD844_846bkw,LoadD844_846bkvar)
        dssText.Command= 'New Load.D844_846rb Bus1=846.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD844_846bkw,LoadD844_846bkvar)
        dssText.Command= 'New Load.D844_846sc Bus1=844.3 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD844_846ckw,LoadD844_846ckvar)
        dssText.Command= 'New Load.D844_846rc Bus1=846.3 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD844_846ckw,LoadD844_846ckvar)

        dssText.Command= 'New Load.D846_848sb Bus1=846.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD846_848bkw,LoadD846_848bkvar)
        dssText.Command= 'New Load.D846_848rb Bus1=848.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD846_848bkw,LoadD846_848bkvar)

        dssText.Command= 'Load.s860.vminpu=.85'
        dssText.Command= 'Load.s840.vminpu=.85'
        dssText.Command= 'Load.s844.vminpu=.85'
        dssText.Command= 'Load.s848.vminpu=.85'
        dssText.Command= 'Load.s830a.vminpu=.85'
        dssText.Command= 'Load.s830b.vminpu=.85'
        dssText.Command= 'Load.s830c.vminpu=.85'
        dssText.Command= 'Load.s890.vminpu=.85'
        dssText.Command= 'Load.d802_806sb.vminpu=.85'
        dssText.Command= 'Load.d802_806rb.vminpu=.85'
        dssText.Command= 'Load.d802_806sc.vminpu=.85'
        dssText.Command= 'Load.d802_806rc.vminpu=.85'
        dssText.Command= 'Load.d808_810sb.vminpu=.85'
        dssText.Command= 'Load.d808_810rb.vminpu=.85'
        dssText.Command= 'Load.d818_820sa.vminpu=.85'
        dssText.Command= 'Load.d818_820ra.vminpu=.85'
        dssText.Command= 'Load.d820_822sa.vminpu=.85'
        dssText.Command= 'Load.d820_822ra.vminpu=.85'
        dssText.Command= 'Load.d816_824sb.vminpu=.85'
        dssText.Command= 'Load.d816_824rb.vminpu=.85'
        dssText.Command= 'Load.d824_826sb.vminpu=.85'
        dssText.Command= 'Load.d824_826rb.vminpu=.85'
        dssText.Command= 'Load.d824_828sc.vminpu=.85'
        dssText.Command= 'Load.d824_828rc.vminpu=.85'
        dssText.Command= 'Load.d828_830sa.vminpu=.85'
        dssText.Command= 'Load.d828_830ra.vminpu=.85'
        dssText.Command= 'Load.d854_856sb.vminpu=.85'
        dssText.Command= 'Load.d854_856rb.vminpu=.85'
        dssText.Command= 'Load.d832_858sa.vminpu=.85'
        dssText.Command= 'Load.d832_858ra.vminpu=.85'
        dssText.Command= 'Load.d832_858sb.vminpu=.85'
        dssText.Command= 'Load.d832_858rb.vminpu=.85'
        dssText.Command= 'Load.d832_858sc.vminpu=.85'
        dssText.Command= 'Load.d832_858rc.vminpu=.85'
        dssText.Command= 'Load.d858_864sb.vminpu=.85'
        dssText.Command= 'Load.d858_864rb.vminpu=.85'
        dssText.Command= 'Load.d858_834sa.vminpu=.85'
        dssText.Command= 'Load.d858_834ra.vminpu=.85'
        dssText.Command= 'Load.d858_834sb.vminpu=.85'
        dssText.Command= 'Load.d858_834rb.vminpu=.85'
        dssText.Command= 'Load.d858_834sc.vminpu=.85'
        dssText.Command= 'Load.d858_834rc.vminpu=.85'
        dssText.Command= 'Load.d834_860sa.vminpu=.85'
        dssText.Command= 'Load.d834_860ra.vminpu=.85'
        dssText.Command= 'Load.d834_860sb.vminpu=.85'
        dssText.Command= 'Load.d834_860rb.vminpu=.85'
        dssText.Command= 'Load.d834_860sc.vminpu=.85'
        dssText.Command= 'Load.d834_860rc.vminpu=.85'
        dssText.Command= 'Load.d860_836sa.vminpu=.85'
        dssText.Command= 'Load.d860_836ra.vminpu=.85'
        dssText.Command= 'Load.d860_836sb.vminpu=.85'
        dssText.Command= 'Load.d860_836rb.vminpu=.85'
        dssText.Command= 'Load.d860_836sc.vminpu=.85'
        dssText.Command= 'Load.d860_836rc.vminpu=.85'
        dssText.Command= 'Load.d836_840sa.vminpu=.85'
        dssText.Command= 'Load.d836_840ra.vminpu=.85'
        dssText.Command= 'Load.d836_840sb.vminpu=.85'
        dssText.Command= 'Load.d836_840rb.vminpu=.85'
        dssText.Command= 'Load.d862_838sb.vminpu=.85'
        dssText.Command= 'Load.d862_838rb.vminpu=.85'
        dssText.Command= 'Load.d842_844sa.vminpu=.85'
        dssText.Command= 'Load.d842_844ra.vminpu=.85'
        dssText.Command= 'Load.d844_846sb.vminpu=.85'
        dssText.Command= 'Load.d844_846rb.vminpu=.85'
        dssText.Command= 'Load.d844_846sc.vminpu=.85'
        dssText.Command= 'Load.d844_846rc.vminpu=.85'
        dssText.Command= 'Load.d846_848sb.vminpu=.85'
        dssText.Command= 'Load.d846_848rb.vminpu=.85'


        dssText.Command='new XYCurve.Eff npts=4 xarray=[.1 .2 .4 1.0] yarray=[1 1 1 1]'

        a = pv
        if a>0.2 :
            Qpu = 1
        else :
            Qpu = a/0.2
        act1 = 14
        act2 = 14
        act3 = 14
        act4 = 14
        act5 = 14
        act6 = 14
        act7 = 14
        act8 = 14
        act9 = 14
        act10 = 14
        act11 = 14
        act12 = 14

        
        #830
        dssText.Command="New PVSystem.PVgen830a Phases=1 Bus1=830.1 Pmpp= 100 Irradiance=%s kV=14.376 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=wye EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
        dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
        dssText.Command='New XYcurve.vvc830a npts=6 yarray=[%f %f 0 0 -%f -%f] xarray=[ 0.1 %f %f %f %f 1.5]'%(Qpu,Qpu,Qpu,Qpu,self.action[act1][0],self.action[act1][1],self.action[act1][2],self.action[act1][3])
        dssText.Command='New InvControl.VoltVar830a DERList=PVSystem.PVgen830a mode=VOLTVAR voltage_curvex_ref=rated vvc_curve1=vvc830a VoltageChangeTolerance=0.01 VarChangeTolerance=1 RefReactivePower=VAraval'
        #830
        dssText.Command="New PVSystem.PVgen830b Phases=1 Bus1=830.2 Pmpp= 100 Irradiance=%s kV=14.376 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=wye EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
        dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
        dssText.Command='New XYcurve.vvc830b npts=6 yarray=[%f %f 0 0 -%f -%f] xarray=[ 0.1 %f %f %f %f 1.5]'%(Qpu,Qpu,Qpu,Qpu,self.action[act2][0],self.action[act2][1],self.action[act2][2],self.action[act2][3])
        dssText.Command='New InvControl.VoltVar830b DERList=PVSystem.PVgen830b mode=VOLTVAR voltage_curvex_ref=rated vvc_curve1=vvc830b VoltageChangeTolerance=0.01 VarChangeTolerance=1 RefReactivePower=VAraval'
        #830
        dssText.Command="New PVSystem.PVgen830c Phases=1 Bus1=830.3 Pmpp= 100 Irradiance=%s kV=14.376 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=wye EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
        dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
        dssText.Command='New XYcurve.vvc830c npts=6 yarray=[%f %f 0 0 -%f -%f] xarray=[ 0.1 %f %f %f %f 1.5]'%(Qpu,Qpu,Qpu,Qpu,self.action[act3][0],self.action[act3][1],self.action[act3][2],self.action[act3][3])
        dssText.Command='New InvControl.VoltVar830c DERList=PVSystem.PVgen830c mode=VOLTVAR voltage_curvex_ref=rated vvc_curve1=vvc830c VoltageChangeTolerance=0.01 VarChangeTolerance=1 RefReactivePower=VAraval'

        #840
        dssText.Command="New PVSystem.PVgen840a Phases=1 Bus1=840.1.2 Pmpp= 100 Irradiance=%s kV=24.9 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=delta EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
        dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
        dssText.Command='New XYcurve.vvc840a npts=6 yarray=[%f %f 0 0 -%f -%f] xarray=[ 0.1 %f %f %f %f 1.5]'%(Qpu,Qpu,Qpu,Qpu,self.action[act4][0],self.action[act4][1],self.action[act4][2],self.action[act4][3])
        dssText.Command='New InvControl.VoltVar840a DERList=PVSystem.PVgen840a mode=VOLTVAR voltage_curvex_ref=rated vvc_curve1=vvc840a VoltageChangeTolerance=0.01 VarChangeTolerance=1 RefReactivePower=VAraval'
        #840
        dssText.Command="New PVSystem.PVgen840b Phases=1 Bus1=840.2.3 Pmpp= 100 Irradiance=%s kV=24.9 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=delta EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
        dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
        dssText.Command='New XYcurve.vvc840b npts=6 yarray=[%f %f 0 0 -%f -%f] xarray=[ 0.1 %f %f %f %f 1.5]'%(Qpu,Qpu,Qpu,Qpu,self.action[act5][0],self.action[act5][1],self.action[act5][2],self.action[act5][3])
        dssText.Command='New InvControl.VoltVar840b DERList=PVSystem.PVgen840b mode=VOLTVAR voltage_curvex_ref=rated vvc_curve1=vvc840b VoltageChangeTolerance=0.01 VarChangeTolerance=1 RefReactivePower=VAraval'
        #840
        dssText.Command="New PVSystem.PVgen840c Phases=1 Bus1=840.3.1 Pmpp= 100 Irradiance=%s kV=24.9 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=delta EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
        dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
        dssText.Command='New XYcurve.vvc840c npts=6 yarray=[%f %f 0 0 -%f -%f] xarray=[ 0.1 %f %f %f %f 1.5]'%(Qpu,Qpu,Qpu,Qpu,self.action[act6][0],self.action[act6][1],self.action[act6][2],self.action[act6][3])
        dssText.Command='New InvControl.VoltVar840c DERList=PVSystem.PVgen840c mode=VOLTVAR voltage_curvex_ref=rated vvc_curve1=vvc840c VoltageChangeTolerance=0.01 VarChangeTolerance=1 RefReactivePower=VAraval'        
 
        #890
        dssText.Command="New PVSystem.PVgen890a Phases=1 Bus1=890.1.2 Pmpp= 100 Irradiance=%s kV=4.16 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=delta EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
        dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
        dssText.Command='New XYcurve.vvc890a npts=6 yarray=[%f %f 0 0 -%f -%f] xarray=[ 0.1 %f %f %f %f 1.5]'%(Qpu,Qpu,Qpu,Qpu,self.action[act7][0],self.action[act7][1],self.action[act7][2],self.action[act7][3])
        dssText.Command='New InvControl.VoltVar890a DERList=PVSystem.PVgen890a mode=VOLTVAR voltage_curvex_ref=rated vvc_curve1=vvc890a VoltageChangeTolerance=0.01 VarChangeTolerance=1 RefReactivePower=VAraval'

        dssText.Command="New PVSystem.PVgen890b Phases=1 Bus1=890.2.3 Pmpp= 100 Irradiance=%s kV=4.16 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=delta EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
        dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
        dssText.Command='New XYcurve.vvc890b npts=6 yarray=[%f %f 0 0 -%f -%f] xarray=[ 0.1 %f %f %f %f 1.5]'%(Qpu,Qpu,Qpu,Qpu,self.action[act8][0],self.action[act8][1],self.action[act8][2],self.action[act8][3])
        dssText.Command='New InvControl.VoltVar890b DERList=PVSystem.PVgen890b mode=VOLTVAR voltage_curvex_ref=rated vvc_curve1=vvc890b VoltageChangeTolerance=0.01 VarChangeTolerance=1 RefReactivePower=VAraval'

        dssText.Command="New PVSystem.PVgen890c Phases=1 Bus1=890.3.1 Pmpp= 100 Irradiance=%s kV=4.16 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=delta EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
        dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
        dssText.Command='New XYcurve.vvc890c npts=6 yarray=[%f %f 0 0 -%f -%f] xarray=[ 0.1 %f %f %f %f 1.5]'%(Qpu,Qpu,Qpu,Qpu,self.action[act9][0],self.action[act9][1],self.action[act9][2],self.action[act9][3])
        dssText.Command='New InvControl.VoltVar890c DERList=PVSystem.PVgen890c mode=VOLTVAR voltage_curvex_ref=rated vvc_curve1=vvc890c VoltageChangeTolerance=0.01 VarChangeTolerance=1 RefReactivePower=VAraval'

        dssText.Command= 'Set Voltagebases= "69,24.9,4.16, .48"'

        dssText.Command= 'calcv'

        dssText.Command= 'Set maxiterations = 300'
        dssText.Command= 'Solve'

        Vabc_pu=dssCircuit.AllBusVolts

        V632ar = Vabc_pu[12]
        V632ai = Vabc_pu[13]
        V632br = Vabc_pu[14]
        V632bi = Vabc_pu[15]
        V632cr = Vabc_pu[16]
        V632ci = Vabc_pu[17]

        Vseq = np.zeros((3,1), dtype=complex)
        Vseq = np.array([[V632ar + V632ai*1j],
            [V632br +V632bi*1j],
            [V632cr +V632ci*1j]])
        Vzpn = np.dot(invA,Vseq)
        Vz = Vzpn[0]
        Vp = Vzpn[1]
        Vn = Vzpn[2]
            
        VUFz = abs(Vz)/abs(Vp)*100
        VUFn = abs(Vn)/abs(Vp)*100
        noactVUF = VUFn + VUFz

        return VUFn , VUFz

    def noinvwithouttap(self,i,pv):
        dssObj = win32com.client.Dispatch('OpenDSSEngine.DSS')
        dssText = dssObj.Text 
        dssCircuit = dssObj.ActiveCircuit 
        dssSolution = dssCircuit.Solution 
        dssElem = dssCircuit.ActiveCktElement 
        dssBus = dssCircuit.ActiveBus
        dssText.Command= 'Clear'
        dssText.Command= 'new circuit.IEEE34'
        dssText.Command= '~ basekv=69 pu=1.05 angle=30 mvasc3=200000'

        dssText.Command= 'New Transformer.SubXF Phases=3 Windings=2 Xhl=0.01    ! normally 8'
        dssText.Command= '~ wdg=1 bus=sourcebus conn=Delta kv=69    kva=25000   %r=0.0005   !reduce %r, too'
        dssText.Command= '~ wdg=2 bus=800       conn=wye   kv=24.9  kva=25000   %r=0.0005'

        dssText.Command= 'Redirect        IEEELineCodes.dss   ! revised according to Later test feeder doc'
        dssText.Command= 'New Line.L1     Phases=3 Bus1=800.1.2.3  Bus2=802.1.2.3  LineCode=300  Length=2.58   units=kft'
        dssText.Command= 'New Line.L2     Phases=3 Bus1=802.1.2.3  Bus2=806.1.2.3  LineCode=300  Length=1.73   units=kft'
        dssText.Command= 'New Line.L3     Phases=3 Bus1=806.1.2.3  Bus2=808.1.2.3  LineCode=300  Length=32.23   units=kft'
        dssText.Command= 'New Line.L4     Phases=1 Bus1=808.2      Bus2=810.2      LineCode=303  Length=5.804   units=kft'
        dssText.Command= 'New Line.L5     Phases=3 Bus1=808.1.2.3  Bus2=812.1.2.3  LineCode=300  Length=37.5   units=kft'
        dssText.Command= 'New Line.L6     Phases=3 Bus1=812.1.2.3  Bus2=814.1.2.3  LineCode=300  Length=29.73   units=kft'
        dssText.Command= 'New Line.L7     Phases=3 Bus1=814r.1.2.3 Bus2=850.1.2.3  LineCode=301  Length=0.01   units=kft'
        dssText.Command= 'New Line.L8     Phases=1 Bus1=816.1      Bus2=818.1      LineCode=302  Length=1.71   units=kft'
        dssText.Command= 'New Line.L9     Phases=3 Bus1=816.1.2.3  Bus2=824.1.2.3  LineCode=301  Length=10.21   units=kft'
        dssText.Command= 'New Line.L10    Phases=1 Bus1=818.1      Bus2=820.1      LineCode=302  Length=48.15   units=kft'
        dssText.Command= 'New Line.L11    Phases=1 Bus1=820.1      Bus2=822.1      LineCode=302  Length=13.74   units=kft'
        dssText.Command= 'New Line.L12    Phases=1 Bus1=824.2      Bus2=826.2      LineCode=303  Length=3.03   units=kft'
        dssText.Command= 'New Line.L13    Phases=3 Bus1=824.1.2.3  Bus2=828.1.2.3  LineCode=301  Length=0.84   units=kft'
        dssText.Command= 'New Line.L14    Phases=3 Bus1=828.1.2.3  Bus2=830.1.2.3  LineCode=301  Length=20.44   units=kft'
        dssText.Command= 'New Line.L15    Phases=3 Bus1=830.1.2.3  Bus2=854.1.2.3  LineCode=301  Length=0.52   units=kft'
        dssText.Command= 'New Line.L16    Phases=3 Bus1=832.1.2.3  Bus2=858.1.2.3  LineCode=301  Length=4.9   units=kft'
        dssText.Command= 'New Line.L17    Phases=3 Bus1=834.1.2.3  Bus2=860.1.2.3  LineCode=301  Length=2.02   units=kft'
        dssText.Command= 'New Line.L18    Phases=3 Bus1=834.1.2.3  Bus2=842.1.2.3  LineCode=301  Length=0.28   units=kft'
        dssText.Command= 'New Line.L19    Phases=3 Bus1=836.1.2.3  Bus2=840.1.2.3  LineCode=301  Length=0.86   units=kft'
        dssText.Command= 'New Line.L20    Phases=3 Bus1=836.1.2.3  Bus2=862.1.2.3  LineCode=301  Length=0.28   units=kft'
        dssText.Command= 'New Line.L21    Phases=3 Bus1=842.1.2.3  Bus2=844.1.2.3  LineCode=301  Length=1.35   units=kft'
        dssText.Command= 'New Line.L22    Phases=3 Bus1=844.1.2.3  Bus2=846.1.2.3  LineCode=301  Length=3.64   units=kft'
        dssText.Command= 'New Line.L23    Phases=3 Bus1=846.1.2.3  Bus2=848.1.2.3  LineCode=301  Length=0.53   units=kft'
        dssText.Command= 'New Line.L24    Phases=3 Bus1=850.1.2.3  Bus2=816.1.2.3  LineCode=301  Length=0.31   units=kft'
        dssText.Command= 'New Line.L25    Phases=3 Bus1=852r.1.2.3 Bus2=832.1.2.3  LineCode=301  Length=0.01   units=kft'

        dssText.Command= '! 24.9/4.16 kV  Transformer'
        dssText.Command= 'New Transformer.XFM1  Phases=3 Windings=2 Xhl=4.08'
        dssText.Command= '~ wdg=1 bus=832       conn=wye   kv=24.9  kva=500    %r=0.95'
        dssText.Command= '~ wdg=2 bus=888       conn=Wye   kv=4.16  kva=500    %r=0.95'
        dssText.Command= '~ Tap = %s'%(self.tap7)

        dssText.Command= 'New Line.L26    Phases=1 Bus1=854.2      Bus2=856.2      LineCode=303  Length=23.33   units=kft'
        dssText.Command= 'New Line.L27    Phases=3 Bus1=854.1.2.3  Bus2=852.1.2.3  LineCode=301  Length=36.83   units=kft'
        dssText.Command= '! 9-17-10 858-864 changed to phase A per error report'
        dssText.Command= 'New Line.L28    Phases=1 Bus1=858.1      Bus2=864.1      LineCode=303  Length=1.62   units=kft'
        dssText.Command= 'New Line.L29    Phases=3 Bus1=858.1.2.3  Bus2=834.1.2.3  LineCode=301  Length=5.83   units=kft'
        dssText.Command= 'New Line.L30    Phases=3 Bus1=860.1.2.3  Bus2=836.1.2.3  LineCode=301  Length=2.68   units=kft'
        dssText.Command= 'New Line.L31    Phases=1 Bus1=862.2      Bus2=838.2      LineCode=304  Length=4.86   units=kft'
        dssText.Command= 'New Line.L32    Phases=3 Bus1=888.1.2.3  Bus2=890.1.2.3  LineCode=300  Length=10.56   units=kft'
        dssText.Command= '! Capacitors'

        cap844  =  300 *self.Load[self.time*6+i]
        cap848  =  450 *self.Load[self.time*6+i]
        dssText.Command= 'New Capacitor.C844      Bus1=844        Phases=3        kVAR= %s        kV=24.9'%(cap844)
        dssText.Command= 'New Capacitor.C848      Bus1=848        Phases=3        kVAR= %s        kV=24.9'%(cap848)

        dssText.Command= "new transformer.reg1a phases=1 windings=2 buses=(814.1 814r.1) conns='wye wye' "
        dssText.Command= '~ kvs="14.376 14.376" kvas="20000 20000" XHL=1'
        dssText.Command= 'new regcontrol.creg1a transformer=reg1a winding=2 vreg=122 band=2 ptratio=120 ctprim=100 R=2.7 X=1.6 TapNum = %s'%(self.tap1)

        dssText.Command= "new transformer.reg1b phases=1 windings=2 buses=(814.2 814r.2) conns='wye wye'"
        dssText.Command= '~ kvs="14.376 14.376" kvas="20000 20000" XHL=1'
        dssText.Command= 'new regcontrol.creg1b transformer=reg1b winding=2 vreg=122 band=2 ptratio=120 ctprim=100 R=2.7 X=1.6 TapNum = %s'%(self.tap2)

        dssText.Command= "new transformer.reg1c phases=1 windings=2 buses=(814.3 814r.3) conns='wye wye'"
        dssText.Command= '~ kvs="14.376 14.376" kvas="20000 20000" XHL=1'
        dssText.Command= 'new regcontrol.creg1c transformer=reg1c winding=2 vreg=122 band=2 ptratio=120 ctprim=100 R=2.7 X=1.6 TapNum = %s'%(self.tap3)

        dssText.Command= "new transformer.reg2a phases=1 windings=2 buses=(852.1 852r.1) conns='wye wye'"
        dssText.Command= '~ kvs="14.376 14.376" kvas="20000 20000" XHL=1'
        dssText.Command= 'new regcontrol.creg2a transformer=reg2a winding=2 vreg=124 band=2 ptratio=120 ctprim=100 R=2.5 X=1.5 TapNum = %s'%(self.tap4)
        dssText.Command= "new transformer.reg2b phases=1 windings=2 buses=(852.2 852r.2) conns='wye wye'"
        dssText.Command= '~ kvs="14.376 14.376" kvas="20000 20000" XHL=1'
        dssText.Command= 'new regcontrol.creg2b transformer=reg2b winding=2 vreg=124 band=2 ptratio=120 ctprim=100 R=2.5 X=1.5 TapNum = %s'%(self.tap5)
        dssText.Command= "new transformer.reg2c phases=1 windings=2 buses=(852.3 852r.3) conns='wye wye'"
        dssText.Command= '~ kvs="14.376 14.376" kvas="20000 20000" XHL=1'
        dssText.Command= 'new regcontrol.creg2c transformer=reg2c winding=2 vreg=124 band=2 ptratio=120 ctprim=100 R=2.5 X=1.5 TapNum = %s'%(self.tap6)

        LoadS860kw   =  60 *self.Load[self.time*6+i]
        LoadS860kvar =  48 *self.Load[self.time*6+i]
        LoadS840kw   =  27 *self.Load[self.time*6+i]
        LoadS840kvar =  21 *self.Load[self.time*6+i]
        LoadS844kw   =  405 *self.Load[self.time*6+i]
        LoadS844kvar =  315 *self.Load[self.time*6+i]
        LoadS848kw   =  60 *self.Load[self.time*6+i]
        LoadS848var  =  48 *self.Load[self.time*6+i]
        LoadS830akw   = 10 *self.Load[self.time*6+i]
        LoadS830akvar = 5  *self.Load[self.time*6+i]
        LoadS830bkw   = 10 *self.Load[self.time*6+i]
        LoadS830bkvar = 5  *self.Load[self.time*6+i]
        LoadS830ckw   = 25 *self.Load[self.time*6+i]
        LoadS830ckvar = 10 *self.Load[self.time*6+i]
        LoadS890kw   =  450 *self.Load[self.time*6+i]
        LoadS890kvar =  225 *self.Load[self.time*6+i]
        # spot loads
        dssText.Command= 'New Load.S860       Bus1=860   Phases=3 Conn=Wye   Model=1 kV= 24.900   kW= %s kVAR= %s'%(LoadS860kw,LoadS860kvar)
        dssText.Command= 'New Load.S840       Bus1=840   Phases=3 Conn=Wye   Model=5 kV= 24.900   kW= %s kVAR= %s'%(LoadS840kw,LoadS840kvar)
        dssText.Command= 'New Load.S844       Bus1=844   Phases=3 Conn=Wye   Model=2 kV= 24.900   kW= %s kVAR= %s'%(LoadS844kw,LoadS844kvar)
        dssText.Command= 'New Load.S848       Bus1=848   Phases=3 Conn=Delta Model=1 kV= 24.900   kW= %s kVAR= %s'%(LoadS848kw,LoadS848var)
        dssText.Command= 'New Load.S830a      Bus1=830.1.2 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadS830akw,LoadS830akvar)
        dssText.Command= 'New Load.S830b      Bus1=830.2.3 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadS830bkw,LoadS830bkvar)
        dssText.Command= 'New Load.S830c      Bus1=830.3.1 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadS830ckw,LoadS830ckvar)
        dssText.Command= 'New Load.S890       Bus1=890   Phases=3 Conn=Delta Model=5 kV=  4.160   kW= %s kVAR= %s'%(LoadS890kw,LoadS890kvar)


        LoadD802_806bkw   =  15 *self.Load[self.time*6+i]
        LoadD802_806bkvar = 7.5 *self.Load[self.time*6+i]        
        LoadD802_806ckw   = 12.5*self.Load[self.time*6+i]
        LoadD802_806ckvar = 7.0 *self.Load[self.time*6+i]

        LoadD808_810bkw   = 8 *self.Load[self.time*6+i]
        LoadD808_810bkvar = 4 *self.Load[self.time*6+i]

        LoadD818_820akw   = 17  *self.Load[self.time*6+i]
        LoadD818_820akvar = 8.5 *self.Load[self.time*6+i]

        LoadD820_822akw   = 67.5*self.Load[self.time*6+i]
        LoadD820_822akvar = 35  *self.Load[self.time*6+i]   

        LoadD816_824bkw   = 2.5*self.Load[self.time*6+i]
        LoadD816_824bkvar = 1  *self.Load[self.time*6+i]  

        LoadD824_826bkw   = 20 *self.Load[self.time*6+i]
        LoadD824_826bkvar = 10 *self.Load[self.time*6+i]   
        
        LoadD824_828ckw   = 2  *self.Load[self.time*6+i]
        LoadD824_828ckvar = 1  *self.Load[self.time*6+i]   
        
        LoadD828_830akw   = 3.5 *self.Load[self.time*6+i]
        LoadD828_830akvar = 1.5 *self.Load[self.time*6+i]   
        
        LoadD854_856bkw   = 2  *self.Load[self.time*6+i]
        LoadD854_856bkvar = 1  *self.Load[self.time*6+i]   
        
        LoadD832_858akw   = 3.5 *self.Load[self.time*6+i]
        LoadD832_858akvar = 1.5 *self.Load[self.time*6+i]           
        LoadD832_858bkw   = 1.0 *self.Load[self.time*6+i]
        LoadD832_858bkvar = 0.5 *self.Load[self.time*6+i]           
        LoadD832_858ckw   = 3   *self.Load[self.time*6+i]
        LoadD832_858ckvar = 1.5 *self.Load[self.time*6+i]   
        
        # distributed loads
        dssText.Command= 'New Load.D802_806sb Bus1=802.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD802_806bkw,LoadD802_806bkvar)
        dssText.Command= 'New Load.D802_806rb Bus1=806.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD802_806bkw,LoadD802_806bkvar)
        dssText.Command= 'New Load.D802_806sc Bus1=802.3 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD802_806ckw,LoadD802_806ckvar)
        dssText.Command= 'New Load.D802_806rc Bus1=806.3 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD802_806ckw,LoadD802_806ckvar)

        dssText.Command= 'New Load.D808_810sb Bus1=808.2 Phases=1 Conn=Wye   Model=4 kV= 14.376 kW= %s  kVAR= %s '%(LoadD808_810bkw,LoadD808_810bkvar)
        dssText.Command= 'New Load.D808_810rb Bus1=810.2 Phases=1 Conn=Wye   Model=4 kV= 14.376 kW= %s  kVAR= %s '%(LoadD808_810bkw,LoadD808_810bkvar)

        dssText.Command= 'New Load.D818_820sa Bus1=818.1 Phases=1 Conn=Wye   Model=2 kV= 14.376 kW= %s  kVAR= %s '%(LoadD818_820akw,LoadD818_820akvar)
        dssText.Command= 'New Load.D818_820ra Bus1=820.1 Phases=1 Conn=Wye   Model=2 kV= 14.376 kW= %s  kVAR= %s '%(LoadD818_820akw,LoadD818_820akvar)

        dssText.Command= 'New Load.D820_822sa Bus1=820.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD820_822akw,LoadD820_822akvar)
        dssText.Command= 'New Load.D820_822ra Bus1=822.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD820_822akw,LoadD820_822akvar)

        dssText.Command= 'New Load.D816_824sb Bus1=816.2 Phases=1 Conn=Delta Model=5 kV= 24.900 kW= %s  kVAR= %s '%(LoadD816_824bkw,LoadD816_824bkvar)
        dssText.Command= 'New Load.D816_824rb Bus1=824.2 Phases=1 Conn=Delta Model=5 kV= 24.900 kW= %s  kVAR= %s '%(LoadD816_824bkw,LoadD816_824bkvar)

        dssText.Command= 'New Load.D824_826sb Bus1=824.2 Phases=1 Conn=Wye   Model=5 kV= 14.376 kW= %s  kVAR= %s '%(LoadD824_826bkw,LoadD824_826bkvar)
        dssText.Command= 'New Load.D824_826rb Bus1=826.2 Phases=1 Conn=Wye   Model=5 kV= 14.376 kW= %s  kVAR= %s '%(LoadD824_826bkw,LoadD824_826bkvar)

        dssText.Command= 'New Load.D824_828sc Bus1=824.3 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD824_828ckw,LoadD824_828ckvar)
        dssText.Command= 'New Load.D824_828rc Bus1=828.3 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD824_828ckw,LoadD824_828ckvar)

        dssText.Command= 'New Load.D828_830sa Bus1=828.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD828_830akw,LoadD828_830akvar)
        dssText.Command= 'New Load.D828_830ra Bus1=830.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD828_830akw,LoadD828_830akvar)
        dssText.Command= 'New Load.D854_856sb Bus1=854.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD854_856bkw,LoadD854_856bkvar)
        dssText.Command= 'New Load.D854_856rb Bus1=856.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD854_856bkw,LoadD854_856bkvar)

        dssText.Command= 'New Load.D832_858sa Bus1=832.1 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s  kVAR= %s '%(LoadD832_858akw,LoadD832_858akvar)
        dssText.Command= 'New Load.D832_858ra Bus1=858.1 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s  kVAR= %s '%(LoadD832_858akw,LoadD832_858akvar)
        dssText.Command= 'New Load.D832_858sb Bus1=832.2 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s  kVAR= %s '%(LoadD832_858bkw,LoadD832_858bkvar)
        dssText.Command= 'New Load.D832_858rb Bus1=858.2 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s  kVAR= %s '%(LoadD832_858bkw,LoadD832_858bkvar)
        dssText.Command= 'New Load.D832_858sc Bus1=832.3 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s  kVAR= %s '%(LoadD832_858ckw,LoadD832_858ckvar)
        dssText.Command= 'New Load.D832_858rc Bus1=858.3 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s  kVAR= %s '%(LoadD832_858ckw,LoadD832_858ckvar)
        
        LoadD858_864bkw   = 1   *self.Load[self.time*6+i]
        LoadD858_864bkvar = 0.5 *self.Load[self.time*6+i]   
        
        LoadD858_834akw   = 2   *self.Load[self.time*6+i]
        LoadD858_834akvar = 1   *self.Load[self.time*6+i]           
        LoadD858_834bkw   = 7.5 *self.Load[self.time*6+i]
        LoadD858_834bkvar = 4   *self.Load[self.time*6+i]          
        LoadD858_834ckw   = 6.5 *self.Load[self.time*6+i]
        LoadD858_834ckvar = 3.5 *self.Load[self.time*6+i]   
        
        LoadD834_860akw   = 8   *self.Load[self.time*6+i]
        LoadD834_860akvar = 4   *self.Load[self.time*6+i]           
        LoadD834_860bkw   = 10  *self.Load[self.time*6+i]
        LoadD834_860bkvar =  5  *self.Load[self.time*6+i]            
        LoadD834_860ckw   = 55  *self.Load[self.time*6+i]
        LoadD834_860ckvar = 27.5*self.Load[self.time*6+i]   
        
        LoadD860_836akw   = 15  *self.Load[self.time*6+i]
        LoadD860_836akvar = 7.5 *self.Load[self.time*6+i]            
        LoadD860_836bkw   = 5   *self.Load[self.time*6+i]
        LoadD860_836bkvar = 3   *self.Load[self.time*6+i]           
        LoadD860_836ckw   = 21  *self.Load[self.time*6+i]
        LoadD860_836ckvar = 11  *self.Load[self.time*6+i]    
        
        LoadD836_840akw   = 9   *self.Load[self.time*6+i]
        LoadD836_840akvar = 4.5 *self.Load[self.time*6+i]           
        LoadD836_840bkw   = 11  *self.Load[self.time*6+i]
        LoadD836_840bkvar = 5.5 *self.Load[self.time*6+i]    
        
        LoadD862_838bkw   = 14  *self.Load[self.time*6+i]
        LoadD862_838bkvar = 7   *self.Load[self.time*6+i]   
        
        LoadD842_844akw   = 4.5 *self.Load[self.time*6+i]
        LoadD842_844akvar = 2.5 *self.Load[self.time*6+i]    
        
        LoadD844_846bkw   = 12.5*self.Load[self.time*6+i]
        LoadD844_846bkvar = 6   *self.Load[self.time*6+i]          
        LoadD844_846ckw   = 10  *self.Load[self.time*6+i]
        LoadD844_846ckvar = 5.5 *self.Load[self.time*6+i]    
        
        LoadD846_848bkw   = 11.5*self.Load[self.time*6+i]
        LoadD846_848bkvar = 5.5 *self.Load[self.time*6+i]  

        dssText.Command= 'New Load.D858_864sb Bus1=858.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD858_864bkw,LoadD858_864bkvar)
        dssText.Command= 'New Load.D858_864rb Bus1=864.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD858_864bkw,LoadD858_864bkvar)

        dssText.Command= 'New Load.D858_834sa Bus1=858.1.2 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD858_834akw,LoadD858_834akvar)
        dssText.Command= 'New Load.D858_834ra Bus1=834.1.2 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD858_834akw,LoadD858_834akvar)
        dssText.Command= 'New Load.D858_834sb Bus1=858.2.3 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD858_834bkw,LoadD858_834bkvar)
        dssText.Command= 'New Load.D858_834rb Bus1=834.2.3 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD858_834bkw,LoadD858_834bkvar)
        dssText.Command= 'New Load.D858_834sc Bus1=858.3.1 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD858_834ckw,LoadD858_834ckvar)
        dssText.Command= 'New Load.D858_834rc Bus1=834.3.1 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD858_834ckw,LoadD858_834ckvar)

        dssText.Command= 'New Load.D834_860sa Bus1=834.1.2 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadD834_860akw,LoadD834_860akvar)
        dssText.Command= 'New Load.D834_860ra Bus1=860.1.2 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadD834_860akw,LoadD834_860akvar)
        dssText.Command= 'New Load.D834_860sb Bus1=834.2.3 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadD834_860bkw,LoadD834_860bkvar)
        dssText.Command= 'New Load.D834_860rb Bus1=860.2.3 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadD834_860bkw,LoadD834_860bkvar)
        dssText.Command= 'New Load.D834_860sc Bus1=834.3.1 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadD834_860ckw,LoadD834_860ckvar)
        dssText.Command= 'New Load.D834_860rc Bus1=860.3.1 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadD834_860ckw,LoadD834_860ckvar)

        dssText.Command= 'New Load.D860_836sa Bus1=860.1.2 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD860_836akw,LoadD860_836akvar)
        dssText.Command= 'New Load.D860_836ra Bus1=836.1.2 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD860_836akw,LoadD860_836akvar)
        dssText.Command= 'New Load.D860_836sb Bus1=860.2.3 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD860_836bkw,LoadD860_836bkvar)
        dssText.Command= 'New Load.D860_836rb Bus1=836.2.3 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD860_836bkw,LoadD860_836bkvar)
        dssText.Command= 'New Load.D860_836sc Bus1=860.3.1 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD860_836ckw,LoadD860_836ckvar)
        dssText.Command= 'New Load.D860_836rc Bus1=836.3.1 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD860_836ckw,LoadD860_836ckvar)

        dssText.Command= 'New Load.D836_840sa Bus1=836.1.2 Phases=1 Conn=Delta Model=5 kV= 24.900 kW= %s kVAR= %s'%(LoadD836_840akw,LoadD836_840akvar)
        dssText.Command= 'New Load.D836_840ra Bus1=840.1.2 Phases=1 Conn=Delta Model=5 kV= 24.900 kW= %s kVAR= %s'%(LoadD836_840akw,LoadD836_840akvar)
        dssText.Command= 'New Load.D836_840sb Bus1=836.2.3 Phases=1 Conn=Delta Model=5 kV= 24.900 kW= %s kVAR= %s'%(LoadD836_840bkw,LoadD836_840bkvar)
        dssText.Command= 'New Load.D836_840rb Bus1=840.2.3 Phases=1 Conn=Delta Model=5 kV= 24.900 kW= %s kVAR= %s'%(LoadD836_840bkw,LoadD836_840bkvar)

        dssText.Command= 'New Load.D862_838sb Bus1=862.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD862_838bkw,LoadD862_838bkvar)
        dssText.Command= 'New Load.D862_838rb Bus1=838.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD862_838bkw,LoadD862_838bkvar)

        dssText.Command= 'New Load.D842_844sa Bus1=842.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD842_844akw,LoadD842_844akvar)
        dssText.Command= 'New Load.D842_844ra Bus1=844.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD842_844akw,LoadD842_844akvar)

        dssText.Command= 'New Load.D844_846sb Bus1=844.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD844_846bkw,LoadD844_846bkvar)
        dssText.Command= 'New Load.D844_846rb Bus1=846.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD844_846bkw,LoadD844_846bkvar)
        dssText.Command= 'New Load.D844_846sc Bus1=844.3 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD844_846ckw,LoadD844_846ckvar)
        dssText.Command= 'New Load.D844_846rc Bus1=846.3 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD844_846ckw,LoadD844_846ckvar)

        dssText.Command= 'New Load.D846_848sb Bus1=846.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD846_848bkw,LoadD846_848bkvar)
        dssText.Command= 'New Load.D846_848rb Bus1=848.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD846_848bkw,LoadD846_848bkvar)

        dssText.Command= 'Load.s860.vminpu=.85'
        dssText.Command= 'Load.s840.vminpu=.85'
        dssText.Command= 'Load.s844.vminpu=.85'
        dssText.Command= 'Load.s848.vminpu=.85'
        dssText.Command= 'Load.s830a.vminpu=.85'
        dssText.Command= 'Load.s830b.vminpu=.85'
        dssText.Command= 'Load.s830c.vminpu=.85'
        dssText.Command= 'Load.s890.vminpu=.85'
        dssText.Command= 'Load.d802_806sb.vminpu=.85'
        dssText.Command= 'Load.d802_806rb.vminpu=.85'
        dssText.Command= 'Load.d802_806sc.vminpu=.85'
        dssText.Command= 'Load.d802_806rc.vminpu=.85'
        dssText.Command= 'Load.d808_810sb.vminpu=.85'
        dssText.Command= 'Load.d808_810rb.vminpu=.85'
        dssText.Command= 'Load.d818_820sa.vminpu=.85'
        dssText.Command= 'Load.d818_820ra.vminpu=.85'
        dssText.Command= 'Load.d820_822sa.vminpu=.85'
        dssText.Command= 'Load.d820_822ra.vminpu=.85'
        dssText.Command= 'Load.d816_824sb.vminpu=.85'
        dssText.Command= 'Load.d816_824rb.vminpu=.85'
        dssText.Command= 'Load.d824_826sb.vminpu=.85'
        dssText.Command= 'Load.d824_826rb.vminpu=.85'
        dssText.Command= 'Load.d824_828sc.vminpu=.85'
        dssText.Command= 'Load.d824_828rc.vminpu=.85'
        dssText.Command= 'Load.d828_830sa.vminpu=.85'
        dssText.Command= 'Load.d828_830ra.vminpu=.85'
        dssText.Command= 'Load.d854_856sb.vminpu=.85'
        dssText.Command= 'Load.d854_856rb.vminpu=.85'
        dssText.Command= 'Load.d832_858sa.vminpu=.85'
        dssText.Command= 'Load.d832_858ra.vminpu=.85'
        dssText.Command= 'Load.d832_858sb.vminpu=.85'
        dssText.Command= 'Load.d832_858rb.vminpu=.85'
        dssText.Command= 'Load.d832_858sc.vminpu=.85'
        dssText.Command= 'Load.d832_858rc.vminpu=.85'
        dssText.Command= 'Load.d858_864sb.vminpu=.85'
        dssText.Command= 'Load.d858_864rb.vminpu=.85'
        dssText.Command= 'Load.d858_834sa.vminpu=.85'
        dssText.Command= 'Load.d858_834ra.vminpu=.85'
        dssText.Command= 'Load.d858_834sb.vminpu=.85'
        dssText.Command= 'Load.d858_834rb.vminpu=.85'
        dssText.Command= 'Load.d858_834sc.vminpu=.85'
        dssText.Command= 'Load.d858_834rc.vminpu=.85'
        dssText.Command= 'Load.d834_860sa.vminpu=.85'
        dssText.Command= 'Load.d834_860ra.vminpu=.85'
        dssText.Command= 'Load.d834_860sb.vminpu=.85'
        dssText.Command= 'Load.d834_860rb.vminpu=.85'
        dssText.Command= 'Load.d834_860sc.vminpu=.85'
        dssText.Command= 'Load.d834_860rc.vminpu=.85'
        dssText.Command= 'Load.d860_836sa.vminpu=.85'
        dssText.Command= 'Load.d860_836ra.vminpu=.85'
        dssText.Command= 'Load.d860_836sb.vminpu=.85'
        dssText.Command= 'Load.d860_836rb.vminpu=.85'
        dssText.Command= 'Load.d860_836sc.vminpu=.85'
        dssText.Command= 'Load.d860_836rc.vminpu=.85'
        dssText.Command= 'Load.d836_840sa.vminpu=.85'
        dssText.Command= 'Load.d836_840ra.vminpu=.85'
        dssText.Command= 'Load.d836_840sb.vminpu=.85'
        dssText.Command= 'Load.d836_840rb.vminpu=.85'
        dssText.Command= 'Load.d862_838sb.vminpu=.85'
        dssText.Command= 'Load.d862_838rb.vminpu=.85'
        dssText.Command= 'Load.d842_844sa.vminpu=.85'
        dssText.Command= 'Load.d842_844ra.vminpu=.85'
        dssText.Command= 'Load.d844_846sb.vminpu=.85'
        dssText.Command= 'Load.d844_846rb.vminpu=.85'
        dssText.Command= 'Load.d844_846sc.vminpu=.85'
        dssText.Command= 'Load.d844_846rc.vminpu=.85'
        dssText.Command= 'Load.d846_848sb.vminpu=.85'
        dssText.Command= 'Load.d846_848rb.vminpu=.85'


        dssText.Command='new XYCurve.Eff npts=4 xarray=[.1 .2 .4 1.0] yarray=[1 1 1 1]'

        a = pv
        if a>0.2 :
            Qpu = 1
        else :
            Qpu = a/0.2

        
        #830
        dssText.Command="New PVSystem.PVgen830a Phases=1 Bus1=830.1 Pmpp= 100 Irradiance=%s kV=14.376 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=wye EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
        dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'

        #830
        dssText.Command="New PVSystem.PVgen830b Phases=1 Bus1=830.2 Pmpp= 100 Irradiance=%s kV=14.376 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=wye EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
        dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
        
        #830
        dssText.Command="New PVSystem.PVgen830c Phases=1 Bus1=830.3 Pmpp= 100 Irradiance=%s kV=14.376 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=wye EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
        dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
        
        #840
        dssText.Command="New PVSystem.PVgen840a Phases=1 Bus1=840.1.2 Pmpp= 100 Irradiance=%s kV=24.9 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=delta EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
        dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
        
        #840
        dssText.Command="New PVSystem.PVgen840b Phases=1 Bus1=840.2.3 Pmpp= 100 Irradiance=%s kV=24.9 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=delta EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
        dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
        
        #840
        dssText.Command="New PVSystem.PVgen840c Phases=1 Bus1=840.3.1 Pmpp= 100 Irradiance=%s kV=24.9 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=delta EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
        dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
        
        #890
        dssText.Command="New PVSystem.PVgen890a Phases=1 Bus1=890.1.2 Pmpp= 100 Irradiance=%s kV=4.16 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=delta EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
        dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
        
        dssText.Command="New PVSystem.PVgen890b Phases=1 Bus1=890.2.3 Pmpp= 100 Irradiance=%s kV=4.16 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=delta EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
        dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
        
        dssText.Command="New PVSystem.PVgen890c Phases=1 Bus1=890.3.1 Pmpp= 100 Irradiance=%s kV=4.16 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=delta EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
        dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
        
        dssText.Command= 'Set Voltagebases= "69,24.9,4.16, .48"'

        dssText.Command= 'calcv'

        dssText.Command= 'Set maxiterations = 300'
        dssText.Command= 'Solve'

        Vabc_pu=dssCircuit.AllBusVolts
        V632ar = Vabc_pu[12]
        V632ai = Vabc_pu[13]
        V632br = Vabc_pu[14]
        V632bi = Vabc_pu[15]
        V632cr = Vabc_pu[16]
        V632ci = Vabc_pu[17]

        Vseq = np.zeros((3,1), dtype=complex)
        Vseq = np.array([[V632ar + V632ai*1j],
            [V632br +V632bi*1j],
            [V632cr +V632ci*1j]])
        Vzpn = np.dot(invA,Vseq)
        Vz = Vzpn[0]
        Vp = Vzpn[1]
        Vn = Vzpn[2]
        
        VUFz = abs(Vz)/abs(Vp)*100
        VUFn = abs(Vn)/abs(Vp)*100

        Vabc_pureal=dssCircuit.AllBusVmagPU


        if self.load_model == False :
            with open(path+ 'busvpunoq.csv', 'a', newline='') as mycsvfile:
                wr = csv.writer(mycsvfile)
                wr.writerow(Vabc_pureal)     

        return VUFn , VUFz 

        

    def noinv(self,i,pv):
        dssObj = win32com.client.Dispatch('OpenDSSEngine.DSS')
        dssText = dssObj.Text 
        dssCircuit = dssObj.ActiveCircuit 
        dssSolution = dssCircuit.Solution 
        dssElem = dssCircuit.ActiveCktElement 
        dssBus = dssCircuit.ActiveBus
        dssText.Command= 'Clear'
        dssText.Command= 'new circuit.IEEE34'
        dssText.Command= '~ basekv=69 pu=1.05 angle=30 mvasc3=200000'

        dssText.Command= 'New Transformer.SubXF Phases=3 Windings=2 Xhl=0.01    ! normally 8'
        dssText.Command= '~ wdg=1 bus=sourcebus conn=Delta kv=69    kva=25000   %r=0.0005   !reduce %r, too'
        dssText.Command= '~ wdg=2 bus=800       conn=wye   kv=24.9  kva=25000   %r=0.0005'

        dssText.Command= 'Redirect        IEEELineCodes.dss   ! revised according to Later test feeder doc'
        dssText.Command= 'New Line.L1     Phases=3 Bus1=800.1.2.3  Bus2=802.1.2.3  LineCode=300  Length=2.58   units=kft'
        dssText.Command= 'New Line.L2     Phases=3 Bus1=802.1.2.3  Bus2=806.1.2.3  LineCode=300  Length=1.73   units=kft'
        dssText.Command= 'New Line.L3     Phases=3 Bus1=806.1.2.3  Bus2=808.1.2.3  LineCode=300  Length=32.23   units=kft'
        dssText.Command= 'New Line.L4     Phases=1 Bus1=808.2      Bus2=810.2      LineCode=303  Length=5.804   units=kft'
        dssText.Command= 'New Line.L5     Phases=3 Bus1=808.1.2.3  Bus2=812.1.2.3  LineCode=300  Length=37.5   units=kft'
        dssText.Command= 'New Line.L6     Phases=3 Bus1=812.1.2.3  Bus2=814.1.2.3  LineCode=300  Length=29.73   units=kft'
        dssText.Command= 'New Line.L7     Phases=3 Bus1=814r.1.2.3 Bus2=850.1.2.3  LineCode=301  Length=0.01   units=kft'
        dssText.Command= 'New Line.L8     Phases=1 Bus1=816.1      Bus2=818.1      LineCode=302  Length=1.71   units=kft'
        dssText.Command= 'New Line.L9     Phases=3 Bus1=816.1.2.3  Bus2=824.1.2.3  LineCode=301  Length=10.21   units=kft'
        dssText.Command= 'New Line.L10    Phases=1 Bus1=818.1      Bus2=820.1      LineCode=302  Length=48.15   units=kft'
        dssText.Command= 'New Line.L11    Phases=1 Bus1=820.1      Bus2=822.1      LineCode=302  Length=13.74   units=kft'
        dssText.Command= 'New Line.L12    Phases=1 Bus1=824.2      Bus2=826.2      LineCode=303  Length=3.03   units=kft'
        dssText.Command= 'New Line.L13    Phases=3 Bus1=824.1.2.3  Bus2=828.1.2.3  LineCode=301  Length=0.84   units=kft'
        dssText.Command= 'New Line.L14    Phases=3 Bus1=828.1.2.3  Bus2=830.1.2.3  LineCode=301  Length=20.44   units=kft'
        dssText.Command= 'New Line.L15    Phases=3 Bus1=830.1.2.3  Bus2=854.1.2.3  LineCode=301  Length=0.52   units=kft'
        dssText.Command= 'New Line.L16    Phases=3 Bus1=832.1.2.3  Bus2=858.1.2.3  LineCode=301  Length=4.9   units=kft'
        dssText.Command= 'New Line.L17    Phases=3 Bus1=834.1.2.3  Bus2=860.1.2.3  LineCode=301  Length=2.02   units=kft'
        dssText.Command= 'New Line.L18    Phases=3 Bus1=834.1.2.3  Bus2=842.1.2.3  LineCode=301  Length=0.28   units=kft'
        dssText.Command= 'New Line.L19    Phases=3 Bus1=836.1.2.3  Bus2=840.1.2.3  LineCode=301  Length=0.86   units=kft'
        dssText.Command= 'New Line.L20    Phases=3 Bus1=836.1.2.3  Bus2=862.1.2.3  LineCode=301  Length=0.28   units=kft'
        dssText.Command= 'New Line.L21    Phases=3 Bus1=842.1.2.3  Bus2=844.1.2.3  LineCode=301  Length=1.35   units=kft'
        dssText.Command= 'New Line.L22    Phases=3 Bus1=844.1.2.3  Bus2=846.1.2.3  LineCode=301  Length=3.64   units=kft'
        dssText.Command= 'New Line.L23    Phases=3 Bus1=846.1.2.3  Bus2=848.1.2.3  LineCode=301  Length=0.53   units=kft'
        dssText.Command= 'New Line.L24    Phases=3 Bus1=850.1.2.3  Bus2=816.1.2.3  LineCode=301  Length=0.31   units=kft'
        dssText.Command= 'New Line.L25    Phases=3 Bus1=852r.1.2.3 Bus2=832.1.2.3  LineCode=301  Length=0.01   units=kft'

        dssText.Command= '! 24.9/4.16 kV  Transformer'
        dssText.Command= 'New Transformer.XFM1  Phases=3 Windings=2 Xhl=4.08'
        dssText.Command= '~ wdg=1 bus=832       conn=wye   kv=24.9  kva=500    %r=0.95'
        dssText.Command= '~ wdg=2 bus=888       conn=Wye   kv=4.16  kva=500    %r=0.95'

        dssText.Command= 'New Line.L26    Phases=1 Bus1=854.2      Bus2=856.2      LineCode=303  Length=23.33   units=kft'
        dssText.Command= 'New Line.L27    Phases=3 Bus1=854.1.2.3  Bus2=852.1.2.3  LineCode=301  Length=36.83   units=kft'
        dssText.Command= '! 9-17-10 858-864 changed to phase A per error report'
        dssText.Command= 'New Line.L28    Phases=1 Bus1=858.1      Bus2=864.1      LineCode=303  Length=1.62   units=kft'
        dssText.Command= 'New Line.L29    Phases=3 Bus1=858.1.2.3  Bus2=834.1.2.3  LineCode=301  Length=5.83   units=kft'
        dssText.Command= 'New Line.L30    Phases=3 Bus1=860.1.2.3  Bus2=836.1.2.3  LineCode=301  Length=2.68   units=kft'
        dssText.Command= 'New Line.L31    Phases=1 Bus1=862.2      Bus2=838.2      LineCode=304  Length=4.86   units=kft'
        dssText.Command= 'New Line.L32    Phases=3 Bus1=888.1.2.3  Bus2=890.1.2.3  LineCode=300  Length=10.56   units=kft'
        dssText.Command= '! Capacitors'

        cap844  =  300 *self.Load[self.time*6+i]
        cap848  =  450 *self.Load[self.time*6+i]
        dssText.Command= 'New Capacitor.C844      Bus1=844        Phases=3        kVAR= %s        kV=24.9'%(cap844)
        dssText.Command= 'New Capacitor.C848      Bus1=848        Phases=3        kVAR= %s        kV=24.9'%(cap848)

        dssText.Command= "new transformer.reg1a phases=1 windings=2 buses=(814.1 814r.1) conns='wye wye' "
        dssText.Command= '~ kvs="14.376 14.376" kvas="20000 20000" XHL=1'
        dssText.Command= 'new regcontrol.creg1a transformer=reg1a winding=2 vreg=122 band=2 ptratio=120 ctprim=100 R=2.7 X=1.6 '

        dssText.Command= "new transformer.reg1b phases=1 windings=2 buses=(814.2 814r.2) conns='wye wye'"
        dssText.Command= '~ kvs="14.376 14.376" kvas="20000 20000" XHL=1'
        dssText.Command= 'new regcontrol.creg1b transformer=reg1b winding=2 vreg=122 band=2 ptratio=120 ctprim=100 R=2.7 X=1.6 '

        dssText.Command= "new transformer.reg1c phases=1 windings=2 buses=(814.3 814r.3) conns='wye wye'"
        dssText.Command= '~ kvs="14.376 14.376" kvas="20000 20000" XHL=1'
        dssText.Command= 'new regcontrol.creg1c transformer=reg1c winding=2 vreg=122 band=2 ptratio=120 ctprim=100 R=2.7 X=1.6 '

        dssText.Command= "new transformer.reg2a phases=1 windings=2 buses=(852.1 852r.1) conns='wye wye'"
        dssText.Command= '~ kvs="14.376 14.376" kvas="20000 20000" XHL=1'
        dssText.Command= 'new regcontrol.creg2a transformer=reg2a winding=2 vreg=124 band=2 ptratio=120 ctprim=100 R=2.5 X=1.5 '
        dssText.Command= "new transformer.reg2b phases=1 windings=2 buses=(852.2 852r.2) conns='wye wye'"
        dssText.Command= '~ kvs="14.376 14.376" kvas="20000 20000" XHL=1'
        dssText.Command= 'new regcontrol.creg2b transformer=reg2b winding=2 vreg=124 band=2 ptratio=120 ctprim=100 R=2.5 X=1.5 '
        dssText.Command= "new transformer.reg2c phases=1 windings=2 buses=(852.3 852r.3) conns='wye wye'"
        dssText.Command= '~ kvs="14.376 14.376" kvas="20000 20000" XHL=1'
        dssText.Command= 'new regcontrol.creg2c transformer=reg2c winding=2 vreg=124 band=2 ptratio=120 ctprim=100 R=2.5 X=1.5 '

        LoadS860kw   =  60 *self.Load[self.time*6+i]
        LoadS860kvar =  48 *self.Load[self.time*6+i]
        LoadS840kw   =  27 *self.Load[self.time*6+i]
        LoadS840kvar =  21 *self.Load[self.time*6+i]
        LoadS844kw   =  405 *self.Load[self.time*6+i]
        LoadS844kvar =  315 *self.Load[self.time*6+i]
        LoadS848kw   =  60 *self.Load[self.time*6+i]
        LoadS848var  =  48 *self.Load[self.time*6+i]
        LoadS830akw   = 10 *self.Load[self.time*6+i]
        LoadS830akvar = 5  *self.Load[self.time*6+i]
        LoadS830bkw   = 10 *self.Load[self.time*6+i]
        LoadS830bkvar = 5  *self.Load[self.time*6+i]
        LoadS830ckw   = 25 *self.Load[self.time*6+i]
        LoadS830ckvar = 10 *self.Load[self.time*6+i]
        LoadS890kw   =  450 *self.Load[self.time*6+i]
        LoadS890kvar =  225 *self.Load[self.time*6+i]
        # spot loads
        dssText.Command= 'New Load.S860       Bus1=860   Phases=3 Conn=Wye   Model=1 kV= 24.900   kW= %s kVAR= %s'%(LoadS860kw,LoadS860kvar)
        dssText.Command= 'New Load.S840       Bus1=840   Phases=3 Conn=Wye   Model=5 kV= 24.900   kW= %s kVAR= %s'%(LoadS840kw,LoadS840kvar)
        dssText.Command= 'New Load.S844       Bus1=844   Phases=3 Conn=Wye   Model=2 kV= 24.900   kW= %s kVAR= %s'%(LoadS844kw,LoadS844kvar)
        dssText.Command= 'New Load.S848       Bus1=848   Phases=3 Conn=Delta Model=1 kV= 24.900   kW= %s kVAR= %s'%(LoadS848kw,LoadS848var)
        dssText.Command= 'New Load.S830a      Bus1=830.1.2 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadS830akw,LoadS830akvar)
        dssText.Command= 'New Load.S830b      Bus1=830.2.3 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadS830bkw,LoadS830bkvar)
        dssText.Command= 'New Load.S830c      Bus1=830.3.1 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadS830ckw,LoadS830ckvar)
        dssText.Command= 'New Load.S890       Bus1=890   Phases=3 Conn=Delta Model=5 kV=  4.160   kW= %s kVAR= %s'%(LoadS890kw,LoadS890kvar)


        LoadD802_806bkw   =  15 *self.Load[self.time*6+i]
        LoadD802_806bkvar = 7.5 *self.Load[self.time*6+i]        
        LoadD802_806ckw   = 12.5*self.Load[self.time*6+i]
        LoadD802_806ckvar = 7.0 *self.Load[self.time*6+i]

        LoadD808_810bkw   = 8 *self.Load[self.time*6+i]
        LoadD808_810bkvar = 4 *self.Load[self.time*6+i]

        LoadD818_820akw   = 17  *self.Load[self.time*6+i]
        LoadD818_820akvar = 8.5 *self.Load[self.time*6+i]

        LoadD820_822akw   = 67.5*self.Load[self.time*6+i]
        LoadD820_822akvar = 35  *self.Load[self.time*6+i]   

        LoadD816_824bkw   = 2.5*self.Load[self.time*6+i]
        LoadD816_824bkvar = 1  *self.Load[self.time*6+i]  

        LoadD824_826bkw   = 20 *self.Load[self.time*6+i]
        LoadD824_826bkvar = 10 *self.Load[self.time*6+i]   
        
        LoadD824_828ckw   = 2  *self.Load[self.time*6+i]
        LoadD824_828ckvar = 1  *self.Load[self.time*6+i]   
        
        LoadD828_830akw   = 3.5 *self.Load[self.time*6+i]
        LoadD828_830akvar = 1.5 *self.Load[self.time*6+i]   
        
        LoadD854_856bkw   = 2  *self.Load[self.time*6+i]
        LoadD854_856bkvar = 1  *self.Load[self.time*6+i]   
        
        LoadD832_858akw   = 3.5 *self.Load[self.time*6+i]
        LoadD832_858akvar = 1.5 *self.Load[self.time*6+i]           
        LoadD832_858bkw   = 1.0 *self.Load[self.time*6+i]
        LoadD832_858bkvar = 0.5 *self.Load[self.time*6+i]           
        LoadD832_858ckw   = 3   *self.Load[self.time*6+i]
        LoadD832_858ckvar = 1.5 *self.Load[self.time*6+i]   
        
        # distributed loads
        dssText.Command= 'New Load.D802_806sb Bus1=802.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD802_806bkw,LoadD802_806bkvar)
        dssText.Command= 'New Load.D802_806rb Bus1=806.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD802_806bkw,LoadD802_806bkvar)
        dssText.Command= 'New Load.D802_806sc Bus1=802.3 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD802_806ckw,LoadD802_806ckvar)
        dssText.Command= 'New Load.D802_806rc Bus1=806.3 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD802_806ckw,LoadD802_806ckvar)

        dssText.Command= 'New Load.D808_810sb Bus1=808.2 Phases=1 Conn=Wye   Model=4 kV= 14.376 kW= %s  kVAR= %s '%(LoadD808_810bkw,LoadD808_810bkvar)
        dssText.Command= 'New Load.D808_810rb Bus1=810.2 Phases=1 Conn=Wye   Model=4 kV= 14.376 kW= %s  kVAR= %s '%(LoadD808_810bkw,LoadD808_810bkvar)

        dssText.Command= 'New Load.D818_820sa Bus1=818.1 Phases=1 Conn=Wye   Model=2 kV= 14.376 kW= %s  kVAR= %s '%(LoadD818_820akw,LoadD818_820akvar)
        dssText.Command= 'New Load.D818_820ra Bus1=820.1 Phases=1 Conn=Wye   Model=2 kV= 14.376 kW= %s  kVAR= %s '%(LoadD818_820akw,LoadD818_820akvar)

        dssText.Command= 'New Load.D820_822sa Bus1=820.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD820_822akw,LoadD820_822akvar)
        dssText.Command= 'New Load.D820_822ra Bus1=822.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD820_822akw,LoadD820_822akvar)

        dssText.Command= 'New Load.D816_824sb Bus1=816.2 Phases=1 Conn=Delta Model=5 kV= 24.900 kW= %s  kVAR= %s '%(LoadD816_824bkw,LoadD816_824bkvar)
        dssText.Command= 'New Load.D816_824rb Bus1=824.2 Phases=1 Conn=Delta Model=5 kV= 24.900 kW= %s  kVAR= %s '%(LoadD816_824bkw,LoadD816_824bkvar)

        dssText.Command= 'New Load.D824_826sb Bus1=824.2 Phases=1 Conn=Wye   Model=5 kV= 14.376 kW= %s  kVAR= %s '%(LoadD824_826bkw,LoadD824_826bkvar)
        dssText.Command= 'New Load.D824_826rb Bus1=826.2 Phases=1 Conn=Wye   Model=5 kV= 14.376 kW= %s  kVAR= %s '%(LoadD824_826bkw,LoadD824_826bkvar)

        dssText.Command= 'New Load.D824_828sc Bus1=824.3 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD824_828ckw,LoadD824_828ckvar)
        dssText.Command= 'New Load.D824_828rc Bus1=828.3 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD824_828ckw,LoadD824_828ckvar)

        dssText.Command= 'New Load.D828_830sa Bus1=828.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD828_830akw,LoadD828_830akvar)
        dssText.Command= 'New Load.D828_830ra Bus1=830.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD828_830akw,LoadD828_830akvar)
        dssText.Command= 'New Load.D854_856sb Bus1=854.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD854_856bkw,LoadD854_856bkvar)
        dssText.Command= 'New Load.D854_856rb Bus1=856.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD854_856bkw,LoadD854_856bkvar)

        dssText.Command= 'New Load.D832_858sa Bus1=832.1 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s  kVAR= %s '%(LoadD832_858akw,LoadD832_858akvar)
        dssText.Command= 'New Load.D832_858ra Bus1=858.1 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s  kVAR= %s '%(LoadD832_858akw,LoadD832_858akvar)
        dssText.Command= 'New Load.D832_858sb Bus1=832.2 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s  kVAR= %s '%(LoadD832_858bkw,LoadD832_858bkvar)
        dssText.Command= 'New Load.D832_858rb Bus1=858.2 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s  kVAR= %s '%(LoadD832_858bkw,LoadD832_858bkvar)
        dssText.Command= 'New Load.D832_858sc Bus1=832.3 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s  kVAR= %s '%(LoadD832_858ckw,LoadD832_858ckvar)
        dssText.Command= 'New Load.D832_858rc Bus1=858.3 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s  kVAR= %s '%(LoadD832_858ckw,LoadD832_858ckvar)
        
        LoadD858_864bkw   = 1   *self.Load[self.time*6+i]
        LoadD858_864bkvar = 0.5 *self.Load[self.time*6+i]   
        
        LoadD858_834akw   = 2   *self.Load[self.time*6+i]
        LoadD858_834akvar = 1   *self.Load[self.time*6+i]           
        LoadD858_834bkw   = 7.5 *self.Load[self.time*6+i]
        LoadD858_834bkvar = 4   *self.Load[self.time*6+i]          
        LoadD858_834ckw   = 6.5 *self.Load[self.time*6+i]
        LoadD858_834ckvar = 3.5 *self.Load[self.time*6+i]   
        
        LoadD834_860akw   = 8   *self.Load[self.time*6+i]
        LoadD834_860akvar = 4   *self.Load[self.time*6+i]           
        LoadD834_860bkw   = 10  *self.Load[self.time*6+i]
        LoadD834_860bkvar =  5  *self.Load[self.time*6+i]            
        LoadD834_860ckw   = 55  *self.Load[self.time*6+i]
        LoadD834_860ckvar = 27.5*self.Load[self.time*6+i]   
        
        LoadD860_836akw   = 15  *self.Load[self.time*6+i]
        LoadD860_836akvar = 7.5 *self.Load[self.time*6+i]            
        LoadD860_836bkw   = 5   *self.Load[self.time*6+i]
        LoadD860_836bkvar = 3   *self.Load[self.time*6+i]           
        LoadD860_836ckw   = 21  *self.Load[self.time*6+i]
        LoadD860_836ckvar = 11  *self.Load[self.time*6+i]    
        
        LoadD836_840akw   = 9   *self.Load[self.time*6+i]
        LoadD836_840akvar = 4.5 *self.Load[self.time*6+i]           
        LoadD836_840bkw   = 11  *self.Load[self.time*6+i]
        LoadD836_840bkvar = 5.5 *self.Load[self.time*6+i]    
        
        LoadD862_838bkw   = 14  *self.Load[self.time*6+i]
        LoadD862_838bkvar = 7   *self.Load[self.time*6+i]   
        
        LoadD842_844akw   = 4.5 *self.Load[self.time*6+i]
        LoadD842_844akvar = 2.5 *self.Load[self.time*6+i]    
        
        LoadD844_846bkw   = 12.5*self.Load[self.time*6+i]
        LoadD844_846bkvar = 6   *self.Load[self.time*6+i]          
        LoadD844_846ckw   = 10  *self.Load[self.time*6+i]
        LoadD844_846ckvar = 5.5 *self.Load[self.time*6+i]    
        
        LoadD846_848bkw   = 11.5*self.Load[self.time*6+i]
        LoadD846_848bkvar = 5.5 *self.Load[self.time*6+i]  

        dssText.Command= 'New Load.D858_864sb Bus1=858.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD858_864bkw,LoadD858_864bkvar)
        dssText.Command= 'New Load.D858_864rb Bus1=864.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD858_864bkw,LoadD858_864bkvar)

        dssText.Command= 'New Load.D858_834sa Bus1=858.1.2 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD858_834akw,LoadD858_834akvar)
        dssText.Command= 'New Load.D858_834ra Bus1=834.1.2 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD858_834akw,LoadD858_834akvar)
        dssText.Command= 'New Load.D858_834sb Bus1=858.2.3 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD858_834bkw,LoadD858_834bkvar)
        dssText.Command= 'New Load.D858_834rb Bus1=834.2.3 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD858_834bkw,LoadD858_834bkvar)
        dssText.Command= 'New Load.D858_834sc Bus1=858.3.1 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD858_834ckw,LoadD858_834ckvar)
        dssText.Command= 'New Load.D858_834rc Bus1=834.3.1 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD858_834ckw,LoadD858_834ckvar)

        dssText.Command= 'New Load.D834_860sa Bus1=834.1.2 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadD834_860akw,LoadD834_860akvar)
        dssText.Command= 'New Load.D834_860ra Bus1=860.1.2 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadD834_860akw,LoadD834_860akvar)
        dssText.Command= 'New Load.D834_860sb Bus1=834.2.3 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadD834_860bkw,LoadD834_860bkvar)
        dssText.Command= 'New Load.D834_860rb Bus1=860.2.3 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadD834_860bkw,LoadD834_860bkvar)
        dssText.Command= 'New Load.D834_860sc Bus1=834.3.1 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadD834_860ckw,LoadD834_860ckvar)
        dssText.Command= 'New Load.D834_860rc Bus1=860.3.1 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadD834_860ckw,LoadD834_860ckvar)

        dssText.Command= 'New Load.D860_836sa Bus1=860.1.2 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD860_836akw,LoadD860_836akvar)
        dssText.Command= 'New Load.D860_836ra Bus1=836.1.2 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD860_836akw,LoadD860_836akvar)
        dssText.Command= 'New Load.D860_836sb Bus1=860.2.3 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD860_836bkw,LoadD860_836bkvar)
        dssText.Command= 'New Load.D860_836rb Bus1=836.2.3 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD860_836bkw,LoadD860_836bkvar)
        dssText.Command= 'New Load.D860_836sc Bus1=860.3.1 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD860_836ckw,LoadD860_836ckvar)
        dssText.Command= 'New Load.D860_836rc Bus1=836.3.1 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD860_836ckw,LoadD860_836ckvar)

        dssText.Command= 'New Load.D836_840sa Bus1=836.1.2 Phases=1 Conn=Delta Model=5 kV= 24.900 kW= %s kVAR= %s'%(LoadD836_840akw,LoadD836_840akvar)
        dssText.Command= 'New Load.D836_840ra Bus1=840.1.2 Phases=1 Conn=Delta Model=5 kV= 24.900 kW= %s kVAR= %s'%(LoadD836_840akw,LoadD836_840akvar)
        dssText.Command= 'New Load.D836_840sb Bus1=836.2.3 Phases=1 Conn=Delta Model=5 kV= 24.900 kW= %s kVAR= %s'%(LoadD836_840bkw,LoadD836_840bkvar)
        dssText.Command= 'New Load.D836_840rb Bus1=840.2.3 Phases=1 Conn=Delta Model=5 kV= 24.900 kW= %s kVAR= %s'%(LoadD836_840bkw,LoadD836_840bkvar)

        dssText.Command= 'New Load.D862_838sb Bus1=862.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD862_838bkw,LoadD862_838bkvar)
        dssText.Command= 'New Load.D862_838rb Bus1=838.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD862_838bkw,LoadD862_838bkvar)

        dssText.Command= 'New Load.D842_844sa Bus1=842.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD842_844akw,LoadD842_844akvar)
        dssText.Command= 'New Load.D842_844ra Bus1=844.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD842_844akw,LoadD842_844akvar)

        dssText.Command= 'New Load.D844_846sb Bus1=844.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD844_846bkw,LoadD844_846bkvar)
        dssText.Command= 'New Load.D844_846rb Bus1=846.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD844_846bkw,LoadD844_846bkvar)
        dssText.Command= 'New Load.D844_846sc Bus1=844.3 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD844_846ckw,LoadD844_846ckvar)
        dssText.Command= 'New Load.D844_846rc Bus1=846.3 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD844_846ckw,LoadD844_846ckvar)

        dssText.Command= 'New Load.D846_848sb Bus1=846.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD846_848bkw,LoadD846_848bkvar)
        dssText.Command= 'New Load.D846_848rb Bus1=848.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD846_848bkw,LoadD846_848bkvar)

        dssText.Command= 'Load.s860.vminpu=.85'
        dssText.Command= 'Load.s840.vminpu=.85'
        dssText.Command= 'Load.s844.vminpu=.85'
        dssText.Command= 'Load.s848.vminpu=.85'
        dssText.Command= 'Load.s830a.vminpu=.85'
        dssText.Command= 'Load.s830b.vminpu=.85'
        dssText.Command= 'Load.s830c.vminpu=.85'
        dssText.Command= 'Load.s890.vminpu=.85'
        dssText.Command= 'Load.d802_806sb.vminpu=.85'
        dssText.Command= 'Load.d802_806rb.vminpu=.85'
        dssText.Command= 'Load.d802_806sc.vminpu=.85'
        dssText.Command= 'Load.d802_806rc.vminpu=.85'
        dssText.Command= 'Load.d808_810sb.vminpu=.85'
        dssText.Command= 'Load.d808_810rb.vminpu=.85'
        dssText.Command= 'Load.d818_820sa.vminpu=.85'
        dssText.Command= 'Load.d818_820ra.vminpu=.85'
        dssText.Command= 'Load.d820_822sa.vminpu=.85'
        dssText.Command= 'Load.d820_822ra.vminpu=.85'
        dssText.Command= 'Load.d816_824sb.vminpu=.85'
        dssText.Command= 'Load.d816_824rb.vminpu=.85'
        dssText.Command= 'Load.d824_826sb.vminpu=.85'
        dssText.Command= 'Load.d824_826rb.vminpu=.85'
        dssText.Command= 'Load.d824_828sc.vminpu=.85'
        dssText.Command= 'Load.d824_828rc.vminpu=.85'
        dssText.Command= 'Load.d828_830sa.vminpu=.85'
        dssText.Command= 'Load.d828_830ra.vminpu=.85'
        dssText.Command= 'Load.d854_856sb.vminpu=.85'
        dssText.Command= 'Load.d854_856rb.vminpu=.85'
        dssText.Command= 'Load.d832_858sa.vminpu=.85'
        dssText.Command= 'Load.d832_858ra.vminpu=.85'
        dssText.Command= 'Load.d832_858sb.vminpu=.85'
        dssText.Command= 'Load.d832_858rb.vminpu=.85'
        dssText.Command= 'Load.d832_858sc.vminpu=.85'
        dssText.Command= 'Load.d832_858rc.vminpu=.85'
        dssText.Command= 'Load.d858_864sb.vminpu=.85'
        dssText.Command= 'Load.d858_864rb.vminpu=.85'
        dssText.Command= 'Load.d858_834sa.vminpu=.85'
        dssText.Command= 'Load.d858_834ra.vminpu=.85'
        dssText.Command= 'Load.d858_834sb.vminpu=.85'
        dssText.Command= 'Load.d858_834rb.vminpu=.85'
        dssText.Command= 'Load.d858_834sc.vminpu=.85'
        dssText.Command= 'Load.d858_834rc.vminpu=.85'
        dssText.Command= 'Load.d834_860sa.vminpu=.85'
        dssText.Command= 'Load.d834_860ra.vminpu=.85'
        dssText.Command= 'Load.d834_860sb.vminpu=.85'
        dssText.Command= 'Load.d834_860rb.vminpu=.85'
        dssText.Command= 'Load.d834_860sc.vminpu=.85'
        dssText.Command= 'Load.d834_860rc.vminpu=.85'
        dssText.Command= 'Load.d860_836sa.vminpu=.85'
        dssText.Command= 'Load.d860_836ra.vminpu=.85'
        dssText.Command= 'Load.d860_836sb.vminpu=.85'
        dssText.Command= 'Load.d860_836rb.vminpu=.85'
        dssText.Command= 'Load.d860_836sc.vminpu=.85'
        dssText.Command= 'Load.d860_836rc.vminpu=.85'
        dssText.Command= 'Load.d836_840sa.vminpu=.85'
        dssText.Command= 'Load.d836_840ra.vminpu=.85'
        dssText.Command= 'Load.d836_840sb.vminpu=.85'
        dssText.Command= 'Load.d836_840rb.vminpu=.85'
        dssText.Command= 'Load.d862_838sb.vminpu=.85'
        dssText.Command= 'Load.d862_838rb.vminpu=.85'
        dssText.Command= 'Load.d842_844sa.vminpu=.85'
        dssText.Command= 'Load.d842_844ra.vminpu=.85'
        dssText.Command= 'Load.d844_846sb.vminpu=.85'
        dssText.Command= 'Load.d844_846rb.vminpu=.85'
        dssText.Command= 'Load.d844_846sc.vminpu=.85'
        dssText.Command= 'Load.d844_846rc.vminpu=.85'
        dssText.Command= 'Load.d846_848sb.vminpu=.85'
        dssText.Command= 'Load.d846_848rb.vminpu=.85'


        dssText.Command='new XYCurve.Eff npts=4 xarray=[.1 .2 .4 1.0] yarray=[1 1 1 1]'

        a = pv
        if a>0.2 :
            Qpu = 1
        else :
            Qpu = a/0.2

        
        #830
        dssText.Command="New PVSystem.PVgen830a Phases=1 Bus1=830.1 Pmpp= 100 Irradiance=%s kV=14.376 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=wye EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
        dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
        
        #830
        dssText.Command="New PVSystem.PVgen830b Phases=1 Bus1=830.2 Pmpp= 100 Irradiance=%s kV=14.376 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=wye EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
        dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
        
        #830
        dssText.Command="New PVSystem.PVgen830c Phases=1 Bus1=830.3 Pmpp= 100 Irradiance=%s kV=14.376 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=wye EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
        dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
        
        #840
        dssText.Command="New PVSystem.PVgen840a Phases=1 Bus1=840.1.2 Pmpp= 100 Irradiance=%s kV=24.9 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=delta EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
        dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
        
        #840
        dssText.Command="New PVSystem.PVgen840b Phases=1 Bus1=840.2.3 Pmpp= 100 Irradiance=%s kV=24.9 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=delta EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
        dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
        
        #840
        dssText.Command="New PVSystem.PVgen840c Phases=1 Bus1=840.3.1 Pmpp= 100 Irradiance=%s kV=24.9 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=delta EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
        dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
        
        #890
        dssText.Command="New PVSystem.PVgen890a Phases=1 Bus1=890.1.2 Pmpp= 100 Irradiance=%s kV=4.16 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=delta EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
        dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
        
        dssText.Command="New PVSystem.PVgen890b Phases=1 Bus1=890.2.3 Pmpp= 100 Irradiance=%s kV=4.16 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=delta EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
        dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
        
        dssText.Command="New PVSystem.PVgen890c Phases=1 Bus1=890.3.1 Pmpp= 100 Irradiance=%s kV=4.16 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=delta EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
        dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
        
        dssText.Command= 'Set Voltagebases= "69,24.9,4.16, .48"'
        dssText.Command= 'calcv'
        dssText.Command= 'Set maxiterations = 300'
        dssText.Command= 'Solve'

        Vabc_pu=dssCircuit.AllBusVolts

        V632ar = Vabc_pu[12]
        V632ai = Vabc_pu[13]
        V632br = Vabc_pu[14]
        V632bi = Vabc_pu[15]
        V632cr = Vabc_pu[16]
        V632ci = Vabc_pu[17]

        Vseq = np.zeros((3,1), dtype=complex)
        Vseq = np.array([[V632ar + V632ai*1j],
            [V632br +V632bi*1j],
            [V632cr +V632ci*1j]])
        Vzpn = np.dot(invA,Vseq)
        Vz = Vzpn[0]
        Vp = Vzpn[1]
        Vn = Vzpn[2]
        
        VUFz = abs(Vz)/abs(Vp)*100
        VUFn = abs(Vn)/abs(Vp)*100

        dssCircuit.Transformers.First
        dssCircuit.Transformers.Next
        self.tap7 = dssCircuit.Transformers.Tap

        dssCircuit.RegControls.First
        self.tap1 = dssCircuit.RegControls.TapNumber        

        dssCircuit.RegControls.Next
        self.tap2 = dssCircuit.RegControls.TapNumber       

        dssCircuit.RegControls.Next
        self.tap3 = dssCircuit.RegControls.TapNumber

        dssCircuit.RegControls.Next
        self.tap4 = dssCircuit.RegControls.TapNumber       

        dssCircuit.RegControls.Next
        self.tap5 = dssCircuit.RegControls.TapNumber

        dssCircuit.RegControls.Next
        self.tap6 = dssCircuit.RegControls.TapNumber

        Vabc_pureal=dssCircuit.AllBusVmagPU


        if self.load_model == False :
            with open(path+ 'busvpunoq.csv', 'a', newline='') as mycsvfile:
                wr = csv.writer(mycsvfile)
                wr.writerow(Vabc_pureal)     

        return VUFn , VUFz 

    def _step(self,acts):
        act1 = acts[0]
        act2 = acts[1]
        act3 = acts[2]
        act4 = acts[3]
        act5 = acts[4]
        act6 = acts[5]
        act7 = acts[6]
        act8 = acts[7]
        act9 = acts[8]
        reward = 0

        j=0
        infoV = []
        for j in range(6):
            i = j 
            a = self.PV[self.time*6+i]

            if self.time%15==0 and i == 0 :
                noVUFn, noVUFz = self.noinv(i,a)    # 15분에 1번 탭
            else:
                noVUFn, noVUFz = self.noinvwithouttap(i,a) #이외의 무보상 

            dssObj = win32com.client.Dispatch('OpenDSSEngine.DSS')
            dssText = dssObj.Text 
            dssCircuit = dssObj.ActiveCircuit 
            dssSolution = dssCircuit.Solution 
            dssElem = dssCircuit.ActiveCktElement 
            dssBus = dssCircuit.ActiveBus
            dssText.Command= 'Clear'
            dssText.Command= 'new circuit.IEEE34'
            dssText.Command= '~ basekv=69 pu=1.05 angle=30 mvasc3=200000'

            dssText.Command= 'New Transformer.SubXF Phases=3 Windings=2 Xhl=0.01    ! normally 8'
            dssText.Command= '~ wdg=1 bus=sourcebus conn=Delta kv=69    kva=25000   %r=0.0005   !reduce %r, too'
            dssText.Command= '~ wdg=2 bus=800       conn=wye   kv=24.9  kva=25000   %r=0.0005'

            dssText.Command= 'Redirect        IEEELineCodes.dss   ! revised according to Later test feeder doc'
            dssText.Command= 'New Line.L1     Phases=3 Bus1=800.1.2.3  Bus2=802.1.2.3  LineCode=300  Length=2.58   units=kft'
            dssText.Command= 'New Line.L2     Phases=3 Bus1=802.1.2.3  Bus2=806.1.2.3  LineCode=300  Length=1.73   units=kft'
            dssText.Command= 'New Line.L3     Phases=3 Bus1=806.1.2.3  Bus2=808.1.2.3  LineCode=300  Length=32.23   units=kft'
            dssText.Command= 'New Line.L4     Phases=1 Bus1=808.2      Bus2=810.2      LineCode=303  Length=5.804   units=kft'
            dssText.Command= 'New Line.L5     Phases=3 Bus1=808.1.2.3  Bus2=812.1.2.3  LineCode=300  Length=37.5   units=kft'
            dssText.Command= 'New Line.L6     Phases=3 Bus1=812.1.2.3  Bus2=814.1.2.3  LineCode=300  Length=29.73   units=kft'
            dssText.Command= 'New Line.L7     Phases=3 Bus1=814r.1.2.3 Bus2=850.1.2.3  LineCode=301  Length=0.01   units=kft'
            dssText.Command= 'New Line.L8     Phases=1 Bus1=816.1      Bus2=818.1      LineCode=302  Length=1.71   units=kft'
            dssText.Command= 'New Line.L9     Phases=3 Bus1=816.1.2.3  Bus2=824.1.2.3  LineCode=301  Length=10.21   units=kft'
            dssText.Command= 'New Line.L10    Phases=1 Bus1=818.1      Bus2=820.1      LineCode=302  Length=48.15   units=kft'
            dssText.Command= 'New Line.L11    Phases=1 Bus1=820.1      Bus2=822.1      LineCode=302  Length=13.74   units=kft'
            dssText.Command= 'New Line.L12    Phases=1 Bus1=824.2      Bus2=826.2      LineCode=303  Length=3.03   units=kft'
            dssText.Command= 'New Line.L13    Phases=3 Bus1=824.1.2.3  Bus2=828.1.2.3  LineCode=301  Length=0.84   units=kft'
            dssText.Command= 'New Line.L14    Phases=3 Bus1=828.1.2.3  Bus2=830.1.2.3  LineCode=301  Length=20.44   units=kft'
            dssText.Command= 'New Line.L15    Phases=3 Bus1=830.1.2.3  Bus2=854.1.2.3  LineCode=301  Length=0.52   units=kft'
            dssText.Command= 'New Line.L16    Phases=3 Bus1=832.1.2.3  Bus2=858.1.2.3  LineCode=301  Length=4.9   units=kft'
            dssText.Command= 'New Line.L17    Phases=3 Bus1=834.1.2.3  Bus2=860.1.2.3  LineCode=301  Length=2.02   units=kft'
            dssText.Command= 'New Line.L18    Phases=3 Bus1=834.1.2.3  Bus2=842.1.2.3  LineCode=301  Length=0.28   units=kft'
            dssText.Command= 'New Line.L19    Phases=3 Bus1=836.1.2.3  Bus2=840.1.2.3  LineCode=301  Length=0.86   units=kft'
            dssText.Command= 'New Line.L20    Phases=3 Bus1=836.1.2.3  Bus2=862.1.2.3  LineCode=301  Length=0.28   units=kft'
            dssText.Command= 'New Line.L21    Phases=3 Bus1=842.1.2.3  Bus2=844.1.2.3  LineCode=301  Length=1.35   units=kft'
            dssText.Command= 'New Line.L22    Phases=3 Bus1=844.1.2.3  Bus2=846.1.2.3  LineCode=301  Length=3.64   units=kft'
            dssText.Command= 'New Line.L23    Phases=3 Bus1=846.1.2.3  Bus2=848.1.2.3  LineCode=301  Length=0.53   units=kft'
            dssText.Command= 'New Line.L24    Phases=3 Bus1=850.1.2.3  Bus2=816.1.2.3  LineCode=301  Length=0.31   units=kft'
            dssText.Command= 'New Line.L25    Phases=3 Bus1=852r.1.2.3 Bus2=832.1.2.3  LineCode=301  Length=0.01   units=kft'

            dssText.Command= '! 24.9/4.16 kV  Transformer'
            dssText.Command= 'New Transformer.XFM1  Phases=3 Windings=2 Xhl=4.08'
            dssText.Command= '~ wdg=1 bus=832       conn=wye   kv=24.9  kva=500    %r=0.95'
            dssText.Command= '~ wdg=2 bus=888       conn=Wye   kv=4.16  kva=500    %r=0.95'
            dssText.Command= '~ Tap = %s'%(self.tap7)

            dssText.Command= 'New Line.L26    Phases=1 Bus1=854.2      Bus2=856.2      LineCode=303  Length=23.33   units=kft'
            dssText.Command= 'New Line.L27    Phases=3 Bus1=854.1.2.3  Bus2=852.1.2.3  LineCode=301  Length=36.83   units=kft'
            dssText.Command= '! 9-17-10 858-864 changed to phase A per error report'
            dssText.Command= 'New Line.L28    Phases=1 Bus1=858.1      Bus2=864.1      LineCode=303  Length=1.62   units=kft'
            dssText.Command= 'New Line.L29    Phases=3 Bus1=858.1.2.3  Bus2=834.1.2.3  LineCode=301  Length=5.83   units=kft'
            dssText.Command= 'New Line.L30    Phases=3 Bus1=860.1.2.3  Bus2=836.1.2.3  LineCode=301  Length=2.68   units=kft'
            dssText.Command= 'New Line.L31    Phases=1 Bus1=862.2      Bus2=838.2      LineCode=304  Length=4.86   units=kft'
            dssText.Command= 'New Line.L32    Phases=3 Bus1=888.1.2.3  Bus2=890.1.2.3  LineCode=300  Length=10.56   units=kft'
            dssText.Command= '! Capacitors'

            cap844  =  300 *self.Load[self.time*6+i]
            cap848  =  450 *self.Load[self.time*6+i]
            dssText.Command= 'New Capacitor.C844      Bus1=844        Phases=3        kVAR= %s        kV=24.9'%(cap844)
            dssText.Command= 'New Capacitor.C848      Bus1=848        Phases=3        kVAR= %s        kV=24.9'%(cap848)

            dssText.Command= "new transformer.reg1a phases=1 windings=2 buses=(814.1 814r.1) conns='wye wye' "
            dssText.Command= '~ kvs="14.376 14.376" kvas="20000 20000" XHL=1'
            dssText.Command= 'new regcontrol.creg1a transformer=reg1a winding=2 vreg=122 band=2 ptratio=120 ctprim=100 R=2.7 X=1.6 TapNum = %s'%(self.tap1)

            dssText.Command= "new transformer.reg1b phases=1 windings=2 buses=(814.2 814r.2) conns='wye wye'"
            dssText.Command= '~ kvs="14.376 14.376" kvas="20000 20000" XHL=1'
            dssText.Command= 'new regcontrol.creg1b transformer=reg1b winding=2 vreg=122 band=2 ptratio=120 ctprim=100 R=2.7 X=1.6 TapNum = %s'%(self.tap2)

            dssText.Command= "new transformer.reg1c phases=1 windings=2 buses=(814.3 814r.3) conns='wye wye'"
            dssText.Command= '~ kvs="14.376 14.376" kvas="20000 20000" XHL=1'
            dssText.Command= 'new regcontrol.creg1c transformer=reg1c winding=2 vreg=122 band=2 ptratio=120 ctprim=100 R=2.7 X=1.6 TapNum = %s'%(self.tap3)

            dssText.Command= "new transformer.reg2a phases=1 windings=2 buses=(852.1 852r.1) conns='wye wye'"
            dssText.Command= '~ kvs="14.376 14.376" kvas="20000 20000" XHL=1'
            dssText.Command= 'new regcontrol.creg2a transformer=reg2a winding=2 vreg=124 band=2 ptratio=120 ctprim=100 R=2.5 X=1.5 TapNum = %s'%(self.tap4)
            dssText.Command= "new transformer.reg2b phases=1 windings=2 buses=(852.2 852r.2) conns='wye wye'"
            dssText.Command= '~ kvs="14.376 14.376" kvas="20000 20000" XHL=1'
            dssText.Command= 'new regcontrol.creg2b transformer=reg2b winding=2 vreg=124 band=2 ptratio=120 ctprim=100 R=2.5 X=1.5 TapNum = %s'%(self.tap5)
            dssText.Command= "new transformer.reg2c phases=1 windings=2 buses=(852.3 852r.3) conns='wye wye'"
            dssText.Command= '~ kvs="14.376 14.376" kvas="20000 20000" XHL=1'
            dssText.Command= 'new regcontrol.creg2c transformer=reg2c winding=2 vreg=124 band=2 ptratio=120 ctprim=100 R=2.5 X=1.5 TapNum = %s'%(self.tap6)

            LoadS860kw   =  60 *self.Load[self.time*6+i]
            LoadS860kvar =  48 *self.Load[self.time*6+i]
            LoadS840kw   =  27 *self.Load[self.time*6+i]
            LoadS840kvar =  21 *self.Load[self.time*6+i]
            LoadS844kw   =  405 *self.Load[self.time*6+i]
            LoadS844kvar =  315 *self.Load[self.time*6+i]
            LoadS848kw   =  60 *self.Load[self.time*6+i]
            LoadS848var  =  48 *self.Load[self.time*6+i]
            LoadS830akw   = 10 *self.Load[self.time*6+i]
            LoadS830akvar = 5  *self.Load[self.time*6+i]
            LoadS830bkw   = 10 *self.Load[self.time*6+i]
            LoadS830bkvar = 5  *self.Load[self.time*6+i]
            LoadS830ckw   = 25 *self.Load[self.time*6+i]
            LoadS830ckvar = 10 *self.Load[self.time*6+i]
            LoadS890kw   =  450 *self.Load[self.time*6+i]
            LoadS890kvar =  225 *self.Load[self.time*6+i]
            # spot loads
            dssText.Command= 'New Load.S860       Bus1=860   Phases=3 Conn=Wye   Model=1 kV= 24.900   kW= %s kVAR= %s'%(LoadS860kw,LoadS860kvar)
            dssText.Command= 'New Load.S840       Bus1=840   Phases=3 Conn=Wye   Model=5 kV= 24.900   kW= %s kVAR= %s'%(LoadS840kw,LoadS840kvar)
            dssText.Command= 'New Load.S844       Bus1=844   Phases=3 Conn=Wye   Model=2 kV= 24.900   kW= %s kVAR= %s'%(LoadS844kw,LoadS844kvar)
            dssText.Command= 'New Load.S848       Bus1=848   Phases=3 Conn=Delta Model=1 kV= 24.900   kW= %s kVAR= %s'%(LoadS848kw,LoadS848var)
            dssText.Command= 'New Load.S830a      Bus1=830.1.2 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadS830akw,LoadS830akvar)
            dssText.Command= 'New Load.S830b      Bus1=830.2.3 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadS830bkw,LoadS830bkvar)
            dssText.Command= 'New Load.S830c      Bus1=830.3.1 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadS830ckw,LoadS830ckvar)
            dssText.Command= 'New Load.S890       Bus1=890   Phases=3 Conn=Delta Model=5 kV=  4.160   kW= %s kVAR= %s'%(LoadS890kw,LoadS890kvar)


            LoadD802_806bkw   =  15 *self.Load[self.time*6+i]
            LoadD802_806bkvar = 7.5 *self.Load[self.time*6+i]        
            LoadD802_806ckw   = 12.5*self.Load[self.time*6+i]
            LoadD802_806ckvar = 7.0 *self.Load[self.time*6+i]

            LoadD808_810bkw   = 8 *self.Load[self.time*6+i]
            LoadD808_810bkvar = 4 *self.Load[self.time*6+i]

            LoadD818_820akw   = 17  *self.Load[self.time*6+i]
            LoadD818_820akvar = 8.5 *self.Load[self.time*6+i]

            LoadD820_822akw   = 67.5*self.Load[self.time*6+i]
            LoadD820_822akvar = 35  *self.Load[self.time*6+i]   

            LoadD816_824bkw   = 2.5*self.Load[self.time*6+i]
            LoadD816_824bkvar = 1  *self.Load[self.time*6+i]  

            LoadD824_826bkw   = 20 *self.Load[self.time*6+i]
            LoadD824_826bkvar = 10 *self.Load[self.time*6+i]   
            
            LoadD824_828ckw   = 2  *self.Load[self.time*6+i]
            LoadD824_828ckvar = 1  *self.Load[self.time*6+i]   
            
            LoadD828_830akw   = 3.5 *self.Load[self.time*6+i]
            LoadD828_830akvar = 1.5 *self.Load[self.time*6+i]   
            
            LoadD854_856bkw   = 2  *self.Load[self.time*6+i]
            LoadD854_856bkvar = 1  *self.Load[self.time*6+i]   
            
            LoadD832_858akw   = 3.5 *self.Load[self.time*6+i]
            LoadD832_858akvar = 1.5 *self.Load[self.time*6+i]           
            LoadD832_858bkw   = 1.0 *self.Load[self.time*6+i]
            LoadD832_858bkvar = 0.5 *self.Load[self.time*6+i]           
            LoadD832_858ckw   = 3   *self.Load[self.time*6+i]
            LoadD832_858ckvar = 1.5 *self.Load[self.time*6+i]   
            
            # distributed loads
            dssText.Command= 'New Load.D802_806sb Bus1=802.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD802_806bkw,LoadD802_806bkvar)
            dssText.Command= 'New Load.D802_806rb Bus1=806.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD802_806bkw,LoadD802_806bkvar)
            dssText.Command= 'New Load.D802_806sc Bus1=802.3 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD802_806ckw,LoadD802_806ckvar)
            dssText.Command= 'New Load.D802_806rc Bus1=806.3 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD802_806ckw,LoadD802_806ckvar)

            dssText.Command= 'New Load.D808_810sb Bus1=808.2 Phases=1 Conn=Wye   Model=4 kV= 14.376 kW= %s  kVAR= %s '%(LoadD808_810bkw,LoadD808_810bkvar)
            dssText.Command= 'New Load.D808_810rb Bus1=810.2 Phases=1 Conn=Wye   Model=4 kV= 14.376 kW= %s  kVAR= %s '%(LoadD808_810bkw,LoadD808_810bkvar)

            dssText.Command= 'New Load.D818_820sa Bus1=818.1 Phases=1 Conn=Wye   Model=2 kV= 14.376 kW= %s  kVAR= %s '%(LoadD818_820akw,LoadD818_820akvar)
            dssText.Command= 'New Load.D818_820ra Bus1=820.1 Phases=1 Conn=Wye   Model=2 kV= 14.376 kW= %s  kVAR= %s '%(LoadD818_820akw,LoadD818_820akvar)

            dssText.Command= 'New Load.D820_822sa Bus1=820.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD820_822akw,LoadD820_822akvar)
            dssText.Command= 'New Load.D820_822ra Bus1=822.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD820_822akw,LoadD820_822akvar)

            dssText.Command= 'New Load.D816_824sb Bus1=816.2 Phases=1 Conn=Delta Model=5 kV= 24.900 kW= %s  kVAR= %s '%(LoadD816_824bkw,LoadD816_824bkvar)
            dssText.Command= 'New Load.D816_824rb Bus1=824.2 Phases=1 Conn=Delta Model=5 kV= 24.900 kW= %s  kVAR= %s '%(LoadD816_824bkw,LoadD816_824bkvar)

            dssText.Command= 'New Load.D824_826sb Bus1=824.2 Phases=1 Conn=Wye   Model=5 kV= 14.376 kW= %s  kVAR= %s '%(LoadD824_826bkw,LoadD824_826bkvar)
            dssText.Command= 'New Load.D824_826rb Bus1=826.2 Phases=1 Conn=Wye   Model=5 kV= 14.376 kW= %s  kVAR= %s '%(LoadD824_826bkw,LoadD824_826bkvar)

            dssText.Command= 'New Load.D824_828sc Bus1=824.3 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD824_828ckw,LoadD824_828ckvar)
            dssText.Command= 'New Load.D824_828rc Bus1=828.3 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD824_828ckw,LoadD824_828ckvar)

            dssText.Command= 'New Load.D828_830sa Bus1=828.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD828_830akw,LoadD828_830akvar)
            dssText.Command= 'New Load.D828_830ra Bus1=830.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD828_830akw,LoadD828_830akvar)
            dssText.Command= 'New Load.D854_856sb Bus1=854.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD854_856bkw,LoadD854_856bkvar)
            dssText.Command= 'New Load.D854_856rb Bus1=856.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD854_856bkw,LoadD854_856bkvar)

            dssText.Command= 'New Load.D832_858sa Bus1=832.1 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s  kVAR= %s '%(LoadD832_858akw,LoadD832_858akvar)
            dssText.Command= 'New Load.D832_858ra Bus1=858.1 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s  kVAR= %s '%(LoadD832_858akw,LoadD832_858akvar)
            dssText.Command= 'New Load.D832_858sb Bus1=832.2 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s  kVAR= %s '%(LoadD832_858bkw,LoadD832_858bkvar)
            dssText.Command= 'New Load.D832_858rb Bus1=858.2 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s  kVAR= %s '%(LoadD832_858bkw,LoadD832_858bkvar)
            dssText.Command= 'New Load.D832_858sc Bus1=832.3 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s  kVAR= %s '%(LoadD832_858ckw,LoadD832_858ckvar)
            dssText.Command= 'New Load.D832_858rc Bus1=858.3 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s  kVAR= %s '%(LoadD832_858ckw,LoadD832_858ckvar)
            
            LoadD858_864bkw   = 1   *self.Load[self.time*6+i]
            LoadD858_864bkvar = 0.5 *self.Load[self.time*6+i]   
            
            LoadD858_834akw   = 2   *self.Load[self.time*6+i]
            LoadD858_834akvar = 1   *self.Load[self.time*6+i]           
            LoadD858_834bkw   = 7.5 *self.Load[self.time*6+i]
            LoadD858_834bkvar = 4   *self.Load[self.time*6+i]          
            LoadD858_834ckw   = 6.5 *self.Load[self.time*6+i]
            LoadD858_834ckvar = 3.5 *self.Load[self.time*6+i]   
            
            LoadD834_860akw   = 8   *self.Load[self.time*6+i]
            LoadD834_860akvar = 4   *self.Load[self.time*6+i]           
            LoadD834_860bkw   = 10  *self.Load[self.time*6+i]
            LoadD834_860bkvar =  5  *self.Load[self.time*6+i]            
            LoadD834_860ckw   = 55  *self.Load[self.time*6+i]
            LoadD834_860ckvar = 27.5*self.Load[self.time*6+i]   
            
            LoadD860_836akw   = 15  *self.Load[self.time*6+i]
            LoadD860_836akvar = 7.5 *self.Load[self.time*6+i]            
            LoadD860_836bkw   = 5   *self.Load[self.time*6+i]
            LoadD860_836bkvar = 3   *self.Load[self.time*6+i]           
            LoadD860_836ckw   = 21  *self.Load[self.time*6+i]
            LoadD860_836ckvar = 11  *self.Load[self.time*6+i]    
            
            LoadD836_840akw   = 9   *self.Load[self.time*6+i]
            LoadD836_840akvar = 4.5 *self.Load[self.time*6+i]           
            LoadD836_840bkw   = 11  *self.Load[self.time*6+i]
            LoadD836_840bkvar = 5.5 *self.Load[self.time*6+i]    
            
            LoadD862_838bkw   = 14  *self.Load[self.time*6+i]
            LoadD862_838bkvar = 7   *self.Load[self.time*6+i]   
            
            LoadD842_844akw   = 4.5 *self.Load[self.time*6+i]
            LoadD842_844akvar = 2.5 *self.Load[self.time*6+i]    
            
            LoadD844_846bkw   = 12.5*self.Load[self.time*6+i]
            LoadD844_846bkvar = 6   *self.Load[self.time*6+i]          
            LoadD844_846ckw   = 10  *self.Load[self.time*6+i]
            LoadD844_846ckvar = 5.5 *self.Load[self.time*6+i]    
            
            LoadD846_848bkw   = 11.5*self.Load[self.time*6+i]
            LoadD846_848bkvar = 5.5 *self.Load[self.time*6+i]  

            dssText.Command= 'New Load.D858_864sb Bus1=858.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD858_864bkw,LoadD858_864bkvar)
            dssText.Command= 'New Load.D858_864rb Bus1=864.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s  kVAR= %s '%(LoadD858_864bkw,LoadD858_864bkvar)

            dssText.Command= 'New Load.D858_834sa Bus1=858.1.2 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD858_834akw,LoadD858_834akvar)
            dssText.Command= 'New Load.D858_834ra Bus1=834.1.2 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD858_834akw,LoadD858_834akvar)
            dssText.Command= 'New Load.D858_834sb Bus1=858.2.3 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD858_834bkw,LoadD858_834bkvar)
            dssText.Command= 'New Load.D858_834rb Bus1=834.2.3 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD858_834bkw,LoadD858_834bkvar)
            dssText.Command= 'New Load.D858_834sc Bus1=858.3.1 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD858_834ckw,LoadD858_834ckvar)
            dssText.Command= 'New Load.D858_834rc Bus1=834.3.1 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD858_834ckw,LoadD858_834ckvar)

            dssText.Command= 'New Load.D834_860sa Bus1=834.1.2 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadD834_860akw,LoadD834_860akvar)
            dssText.Command= 'New Load.D834_860ra Bus1=860.1.2 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadD834_860akw,LoadD834_860akvar)
            dssText.Command= 'New Load.D834_860sb Bus1=834.2.3 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadD834_860bkw,LoadD834_860bkvar)
            dssText.Command= 'New Load.D834_860rb Bus1=860.2.3 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadD834_860bkw,LoadD834_860bkvar)
            dssText.Command= 'New Load.D834_860sc Bus1=834.3.1 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadD834_860ckw,LoadD834_860ckvar)
            dssText.Command= 'New Load.D834_860rc Bus1=860.3.1 Phases=1 Conn=Delta Model=2 kV= 24.900 kW= %s kVAR= %s'%(LoadD834_860ckw,LoadD834_860ckvar)

            dssText.Command= 'New Load.D860_836sa Bus1=860.1.2 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD860_836akw,LoadD860_836akvar)
            dssText.Command= 'New Load.D860_836ra Bus1=836.1.2 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD860_836akw,LoadD860_836akvar)
            dssText.Command= 'New Load.D860_836sb Bus1=860.2.3 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD860_836bkw,LoadD860_836bkvar)
            dssText.Command= 'New Load.D860_836rb Bus1=836.2.3 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD860_836bkw,LoadD860_836bkvar)
            dssText.Command= 'New Load.D860_836sc Bus1=860.3.1 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD860_836ckw,LoadD860_836ckvar)
            dssText.Command= 'New Load.D860_836rc Bus1=836.3.1 Phases=1 Conn=Delta Model=1 kV= 24.900 kW= %s kVAR= %s'%(LoadD860_836ckw,LoadD860_836ckvar)

            dssText.Command= 'New Load.D836_840sa Bus1=836.1.2 Phases=1 Conn=Delta Model=5 kV= 24.900 kW= %s kVAR= %s'%(LoadD836_840akw,LoadD836_840akvar)
            dssText.Command= 'New Load.D836_840ra Bus1=840.1.2 Phases=1 Conn=Delta Model=5 kV= 24.900 kW= %s kVAR= %s'%(LoadD836_840akw,LoadD836_840akvar)
            dssText.Command= 'New Load.D836_840sb Bus1=836.2.3 Phases=1 Conn=Delta Model=5 kV= 24.900 kW= %s kVAR= %s'%(LoadD836_840bkw,LoadD836_840bkvar)
            dssText.Command= 'New Load.D836_840rb Bus1=840.2.3 Phases=1 Conn=Delta Model=5 kV= 24.900 kW= %s kVAR= %s'%(LoadD836_840bkw,LoadD836_840bkvar)

            dssText.Command= 'New Load.D862_838sb Bus1=862.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD862_838bkw,LoadD862_838bkvar)
            dssText.Command= 'New Load.D862_838rb Bus1=838.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD862_838bkw,LoadD862_838bkvar)

            dssText.Command= 'New Load.D842_844sa Bus1=842.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD842_844akw,LoadD842_844akvar)
            dssText.Command= 'New Load.D842_844ra Bus1=844.1 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD842_844akw,LoadD842_844akvar)

            dssText.Command= 'New Load.D844_846sb Bus1=844.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD844_846bkw,LoadD844_846bkvar)
            dssText.Command= 'New Load.D844_846rb Bus1=846.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD844_846bkw,LoadD844_846bkvar)
            dssText.Command= 'New Load.D844_846sc Bus1=844.3 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD844_846ckw,LoadD844_846ckvar)
            dssText.Command= 'New Load.D844_846rc Bus1=846.3 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD844_846ckw,LoadD844_846ckvar)

            dssText.Command= 'New Load.D846_848sb Bus1=846.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD846_848bkw,LoadD846_848bkvar)
            dssText.Command= 'New Load.D846_848rb Bus1=848.2 Phases=1 Conn=Wye   Model=1 kV= 14.376 kW= %s kVAR= %s  '%(LoadD846_848bkw,LoadD846_848bkvar)

            dssText.Command= 'Load.s860.vminpu=.85'
            dssText.Command= 'Load.s840.vminpu=.85'
            dssText.Command= 'Load.s844.vminpu=.85'
            dssText.Command= 'Load.s848.vminpu=.85'
            dssText.Command= 'Load.s830a.vminpu=.85'
            dssText.Command= 'Load.s830b.vminpu=.85'
            dssText.Command= 'Load.s830c.vminpu=.85'
            dssText.Command= 'Load.s890.vminpu=.85'
            dssText.Command= 'Load.d802_806sb.vminpu=.85'
            dssText.Command= 'Load.d802_806rb.vminpu=.85'
            dssText.Command= 'Load.d802_806sc.vminpu=.85'
            dssText.Command= 'Load.d802_806rc.vminpu=.85'
            dssText.Command= 'Load.d808_810sb.vminpu=.85'
            dssText.Command= 'Load.d808_810rb.vminpu=.85'
            dssText.Command= 'Load.d818_820sa.vminpu=.85'
            dssText.Command= 'Load.d818_820ra.vminpu=.85'
            dssText.Command= 'Load.d820_822sa.vminpu=.85'
            dssText.Command= 'Load.d820_822ra.vminpu=.85'
            dssText.Command= 'Load.d816_824sb.vminpu=.85'
            dssText.Command= 'Load.d816_824rb.vminpu=.85'
            dssText.Command= 'Load.d824_826sb.vminpu=.85'
            dssText.Command= 'Load.d824_826rb.vminpu=.85'
            dssText.Command= 'Load.d824_828sc.vminpu=.85'
            dssText.Command= 'Load.d824_828rc.vminpu=.85'
            dssText.Command= 'Load.d828_830sa.vminpu=.85'
            dssText.Command= 'Load.d828_830ra.vminpu=.85'
            dssText.Command= 'Load.d854_856sb.vminpu=.85'
            dssText.Command= 'Load.d854_856rb.vminpu=.85'
            dssText.Command= 'Load.d832_858sa.vminpu=.85'
            dssText.Command= 'Load.d832_858ra.vminpu=.85'
            dssText.Command= 'Load.d832_858sb.vminpu=.85'
            dssText.Command= 'Load.d832_858rb.vminpu=.85'
            dssText.Command= 'Load.d832_858sc.vminpu=.85'
            dssText.Command= 'Load.d832_858rc.vminpu=.85'
            dssText.Command= 'Load.d858_864sb.vminpu=.85'
            dssText.Command= 'Load.d858_864rb.vminpu=.85'
            dssText.Command= 'Load.d858_834sa.vminpu=.85'
            dssText.Command= 'Load.d858_834ra.vminpu=.85'
            dssText.Command= 'Load.d858_834sb.vminpu=.85'
            dssText.Command= 'Load.d858_834rb.vminpu=.85'
            dssText.Command= 'Load.d858_834sc.vminpu=.85'
            dssText.Command= 'Load.d858_834rc.vminpu=.85'
            dssText.Command= 'Load.d834_860sa.vminpu=.85'
            dssText.Command= 'Load.d834_860ra.vminpu=.85'
            dssText.Command= 'Load.d834_860sb.vminpu=.85'
            dssText.Command= 'Load.d834_860rb.vminpu=.85'
            dssText.Command= 'Load.d834_860sc.vminpu=.85'
            dssText.Command= 'Load.d834_860rc.vminpu=.85'
            dssText.Command= 'Load.d860_836sa.vminpu=.85'
            dssText.Command= 'Load.d860_836ra.vminpu=.85'
            dssText.Command= 'Load.d860_836sb.vminpu=.85'
            dssText.Command= 'Load.d860_836rb.vminpu=.85'
            dssText.Command= 'Load.d860_836sc.vminpu=.85'
            dssText.Command= 'Load.d860_836rc.vminpu=.85'
            dssText.Command= 'Load.d836_840sa.vminpu=.85'
            dssText.Command= 'Load.d836_840ra.vminpu=.85'
            dssText.Command= 'Load.d836_840sb.vminpu=.85'
            dssText.Command= 'Load.d836_840rb.vminpu=.85'
            dssText.Command= 'Load.d862_838sb.vminpu=.85'
            dssText.Command= 'Load.d862_838rb.vminpu=.85'
            dssText.Command= 'Load.d842_844sa.vminpu=.85'
            dssText.Command= 'Load.d842_844ra.vminpu=.85'
            dssText.Command= 'Load.d844_846sb.vminpu=.85'
            dssText.Command= 'Load.d844_846rb.vminpu=.85'
            dssText.Command= 'Load.d844_846sc.vminpu=.85'
            dssText.Command= 'Load.d844_846rc.vminpu=.85'
            dssText.Command= 'Load.d846_848sb.vminpu=.85'
            dssText.Command= 'Load.d846_848rb.vminpu=.85'


            dssText.Command='new XYCurve.Eff npts=4 xarray=[.1 .2 .4 1.0] yarray=[1 1 1 1]'

            a = self.PV[self.time*6+i]
            if a>0.2 :
                Qpu = 1
            else :
                Qpu = a/0.2

            
            #830
            dssText.Command="New PVSystem.PVgen830a Phases=1 Bus1=830.1 Pmpp= 100 Irradiance=%s kV=14.376 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=wye EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
            dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
            dssText.Command='New XYcurve.vvc830a npts=6 yarray=[%f %f 0 0 -%f -%f] xarray=[ 0.1 %f %f %f %f 1.5]'%(Qpu,Qpu,Qpu,Qpu,self.action[act1][0],self.action[act1][1],self.action[act1][2],self.action[act1][3])
            dssText.Command='New InvControl.VoltVar830a DERList=PVSystem.PVgen830a mode=VOLTVAR voltage_curvex_ref=rated vvc_curve1=vvc830a VoltageChangeTolerance=0.01 VarChangeTolerance=1 RefReactivePower=VAraval'
            #830
            dssText.Command="New PVSystem.PVgen830b Phases=1 Bus1=830.2 Pmpp= 100 Irradiance=%s kV=14.376 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=wye EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
            dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
            dssText.Command='New XYcurve.vvc830b npts=6 yarray=[%f %f 0 0 -%f -%f] xarray=[ 0.1 %f %f %f %f 1.5]'%(Qpu,Qpu,Qpu,Qpu,self.action[act2][0],self.action[act2][1],self.action[act2][2],self.action[act2][3])
            dssText.Command='New InvControl.VoltVar830b DERList=PVSystem.PVgen830b mode=VOLTVAR voltage_curvex_ref=rated vvc_curve1=vvc830b VoltageChangeTolerance=0.01 VarChangeTolerance=1 RefReactivePower=VAraval'
            #830
            dssText.Command="New PVSystem.PVgen830c Phases=1 Bus1=830.3 Pmpp= 100 Irradiance=%s kV=14.376 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=wye EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
            dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
            dssText.Command='New XYcurve.vvc830c npts=6 yarray=[%f %f 0 0 -%f -%f] xarray=[ 0.1 %f %f %f %f 1.5]'%(Qpu,Qpu,Qpu,Qpu,self.action[act3][0],self.action[act3][1],self.action[act3][2],self.action[act3][3])
            dssText.Command='New InvControl.VoltVar830c DERList=PVSystem.PVgen830c mode=VOLTVAR voltage_curvex_ref=rated vvc_curve1=vvc830c VoltageChangeTolerance=0.01 VarChangeTolerance=1 RefReactivePower=VAraval'

            #840
            dssText.Command="New PVSystem.PVgen840a Phases=1 Bus1=840.1.2 Pmpp= 100 Irradiance=%s kV=24.9 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=delta EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
            dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
            dssText.Command='New XYcurve.vvc840a npts=6 yarray=[%f %f 0 0 -%f -%f] xarray=[ 0.1 %f %f %f %f 1.5]'%(Qpu,Qpu,Qpu,Qpu,self.action[act4][0],self.action[act4][1],self.action[act4][2],self.action[act4][3])
            dssText.Command='New InvControl.VoltVar840a DERList=PVSystem.PVgen840a mode=VOLTVAR voltage_curvex_ref=rated vvc_curve1=vvc840a VoltageChangeTolerance=0.01 VarChangeTolerance=1 RefReactivePower=VAraval'
            #840
            dssText.Command="New PVSystem.PVgen840b Phases=1 Bus1=840.2.3 Pmpp= 100 Irradiance=%s kV=24.9 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=delta EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
            dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
            dssText.Command='New XYcurve.vvc840b npts=6 yarray=[%f %f 0 0 -%f -%f] xarray=[ 0.1 %f %f %f %f 1.5]'%(Qpu,Qpu,Qpu,Qpu,self.action[act5][0],self.action[act5][1],self.action[act5][2],self.action[act5][3])
            dssText.Command='New InvControl.VoltVar840b DERList=PVSystem.PVgen840b mode=VOLTVAR voltage_curvex_ref=rated vvc_curve1=vvc840b VoltageChangeTolerance=0.01 VarChangeTolerance=1 RefReactivePower=VAraval'
            #840
            dssText.Command="New PVSystem.PVgen840c Phases=1 Bus1=840.3.1 Pmpp= 100 Irradiance=%s kV=24.9 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=delta EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
            dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
            dssText.Command='New XYcurve.vvc840c npts=6 yarray=[%f %f 0 0 -%f -%f] xarray=[ 0.1 %f %f %f %f 1.5]'%(Qpu,Qpu,Qpu,Qpu,self.action[act6][0],self.action[act6][1],self.action[act6][2],self.action[act6][3])
            dssText.Command='New InvControl.VoltVar840c DERList=PVSystem.PVgen840c mode=VOLTVAR voltage_curvex_ref=rated vvc_curve1=vvc840c VoltageChangeTolerance=0.01 VarChangeTolerance=1 RefReactivePower=VAraval'        
     
            #890
            dssText.Command="New PVSystem.PVgen890a Phases=1 Bus1=890.1.2 Pmpp= 100 Irradiance=%s kV=4.16 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=delta EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
            dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
            dssText.Command='New XYcurve.vvc890a npts=6 yarray=[%f %f 0 0 -%f -%f] xarray=[ 0.1 %f %f %f %f 1.5]'%(Qpu,Qpu,Qpu,Qpu,self.action[act7][0],self.action[act7][1],self.action[act7][2],self.action[act7][3])
            dssText.Command='New InvControl.VoltVar890a DERList=PVSystem.PVgen890a mode=VOLTVAR voltage_curvex_ref=rated vvc_curve1=vvc890a VoltageChangeTolerance=0.01 VarChangeTolerance=1 RefReactivePower=VAraval'

            dssText.Command="New PVSystem.PVgen890b Phases=1 Bus1=890.2.3 Pmpp= 100 Irradiance=%s kV=4.16 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=delta EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
            dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
            dssText.Command='New XYcurve.vvc890b npts=6 yarray=[%f %f 0 0 -%f -%f] xarray=[ 0.1 %f %f %f %f 1.5]'%(Qpu,Qpu,Qpu,Qpu,self.action[act8][0],self.action[act8][1],self.action[act8][2],self.action[act8][3])
            dssText.Command='New InvControl.VoltVar890b DERList=PVSystem.PVgen890b mode=VOLTVAR voltage_curvex_ref=rated vvc_curve1=vvc890b VoltageChangeTolerance=0.01 VarChangeTolerance=1 RefReactivePower=VAraval'

            dssText.Command="New PVSystem.PVgen890c Phases=1 Bus1=890.3.1 Pmpp= 100 Irradiance=%s kV=4.16 kva=110 kvarmax=44 kvarmaxabs=44 pf=1.0 conn=delta EffCurve=Eff PFPriority=NO WattPriority=NO"%(a)
            dssText.Command= '~ %Cutin=0 %Cutout=0 %PminNoVars=0 %PminkvarMax=0'
            dssText.Command='New XYcurve.vvc890c npts=6 yarray=[%f %f 0 0 -%f -%f] xarray=[ 0.1 %f %f %f %f 1.5]'%(Qpu,Qpu,Qpu,Qpu,self.action[act9][0],self.action[act9][1],self.action[act9][2],self.action[act9][3])
            dssText.Command='New InvControl.VoltVar890c DERList=PVSystem.PVgen890c mode=VOLTVAR voltage_curvex_ref=rated vvc_curve1=vvc890c VoltageChangeTolerance=0.01 VarChangeTolerance=1 RefReactivePower=VAraval'

            dssText.Command= 'Set Voltagebases= "69,24.9,4.16, .48"'

            dssText.Command= 'calcv'

            dssText.Command= 'Set maxiterations = 300'
            dssText.Command= 'Solve'

            Vabc_pu=dssCircuit.AllBusVolts

            # FEEDER 1-PHASE VOLTAGE REGULATORS
            # Define low-impedance 2-wdg transformer
            '''
            if self.time > 228 and self.time < 275:
                dssText.Command= 'Show Powers kVA Elem'
            '''

            Vabc_pureal=dssCircuit.AllBusVmagPU
            if self.load_model == False :
                with open(path+ 'busvpu.csv', 'a', newline='') as mycsvfile:
                    wr = csv.writer(mycsvfile)
                    wr.writerow(Vabc_pureal)


            V632ar = Vabc_pu[12]
            V632ai = Vabc_pu[13]
            V632br = Vabc_pu[14]
            V632bi = Vabc_pu[15]
            V632cr = Vabc_pu[16]
            V632ci = Vabc_pu[17]

            Vseq = np.zeros((3,1), dtype=complex)
            Vseq = np.array([[V632ar + V632ai*1j],
                [V632br +V632bi*1j],
                [V632cr +V632ci*1j]])
            Vzpn = np.dot(invA,Vseq)
            Vz = Vzpn[0]
            Vp = Vzpn[1]
            Vn = Vzpn[2]
            
            VUFz = abs(Vz)/abs(Vp)*100
            VUFn = abs(Vn)/abs(Vp)*100
            VUF = abs(VUFn[0]) + abs(VUFz[0])  
            noactVUFn , noactVUFz = self.fixeddroop(i,a)  
            reward += VUF - (abs(noactVUFn[0]) + abs(noactVUFz[0]))

            if self.load_model == False :
                
                with open(path+ 'VUF2.csv', 'a', newline='') as mycsvfile:
                    wr = csv.writer(mycsvfile)
                    wr.writerow([self.time*6+i,a,self.Load[(self.time*6+i)%self.totaltime],abs(VUFn[0]),abs(VUFz[0]),abs(VUFn[0]) + abs(VUFz[0]), abs(noactVUFn[0]) , abs(noactVUFz[0]),abs(noactVUFn[0]) + abs(noactVUFz[0]), abs(noVUFn[0]) , abs(noVUFz[0]),abs(noVUFn[0]) + abs(noVUFz[0])])
        
        state830a = np.hstack([self.PV[(self.time*6+6)%self.totaltime], self.Load[(self.time*6+6)%self.totaltime] ]).flatten()
        state830b = np.hstack([self.PV[(self.time*6+6)%self.totaltime], self.Load[(self.time*6+6)%self.totaltime] ]).flatten()
        state830c = np.hstack([self.PV[(self.time*6+6)%self.totaltime], self.Load[(self.time*6+6)%self.totaltime] ]).flatten()
        state840a = np.hstack([self.PV[(self.time*6+6)%self.totaltime], self.Load[(self.time*6+6)%self.totaltime] ]).flatten()
        state840b = np.hstack([self.PV[(self.time*6+6)%self.totaltime], self.Load[(self.time*6+6)%self.totaltime] ]).flatten()
        state840c = np.hstack([self.PV[(self.time*6+6)%self.totaltime], self.Load[(self.time*6+6)%self.totaltime] ]).flatten()
        state890a = np.hstack([self.PV[(self.time*6+6)%self.totaltime], self.Load[(self.time*6+6)%self.totaltime] ]).flatten()
        state890b = np.hstack([self.PV[(self.time*6+6)%self.totaltime], self.Load[(self.time*6+6)%self.totaltime] ]).flatten()
        state890c = np.hstack([self.PV[(self.time*6+6)%self.totaltime], self.Load[(self.time*6+6)%self.totaltime] ]).flatten()

        next_states = [state830a, state830b, state830c, state840a, state840b, state840c, state890a, state890b, state890c]

        self.rwd = np.zeros(shape=(9,))

        self.rwd[0] = -reward
        self.rwd[1] = -reward
        self.rwd[2] = -reward
        self.rwd[3] = -reward
        self.rwd[4] = -reward
        self.rwd[5] = -reward
        self.rwd[6] = -reward
        self.rwd[7] = -reward
        self.rwd[8] = -reward

        total_rewards = self.rwd
        infoA = [act1,act2,act3,act4,act5,act6,act7,act8,act9]


        return next_states, total_rewards, infoA, infoV