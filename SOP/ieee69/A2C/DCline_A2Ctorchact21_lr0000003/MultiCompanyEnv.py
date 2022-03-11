import pandapower as pp
import pandapower.networks as nw

import pandapower.topology as top
from pandapower.plotting import simple_plot, simple_plotly , pf_res_plotly
import pandapower.plotting as plot
import seaborn
from pandas import read_json
from scipy.stats import beta
import math
import collections
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd

import numpy as np
import math
import csv

def normal_pdf(x, mu=0, sigma=1):
    sqrt_two_pi = math.sqrt(2 * math.pi)
    return (math.exp(-(x-mu)**2 / 2 / sigma**2) / (sqrt_two_pi * sigma))

class env():
    def __init__(self, numAgent, load_model):

        self.numAgent = numAgent
        self.load_model = load_model
        self.time = 0
        self.MaxTime = 24 
        self.state_size = 13
        self.action_size = 21
        self.batch_size = self.MaxTime
        self.interval = 60       
        self.vpu = 90
        #self.PV = [0, 0, 0, 0, 0, 0, 0.04675, 0.28, 0.52625, 0.7765, 0.9, 0.975, 1, 0.75, 0.625, 0.375, 0.1665, 0.095, 0, 0, 0, 0, 0, 0]
        self.LOAD = [0.793681328 , 0.749258257, 0.726207166, 0.727527632, 0.74872029,  0.778324802, 0.815623879, 0.859231848, 0.937367546, 0.982393792, 0.995875583, 0.99258257,  0.92934694,  0.970803039, 0.989501483, 0.991816374, 1,   0.997522089,
         0.994343191, 0.986029148, 0.953262039, 0.92862965,  0.946692315, 0.952903394] 

    def reset(self,ra,rb):
        train = pd.read_csv('./PV.csv')
        pv = np.array(train[ra*24:(ra+1)*24])
        pv = np.transpose(pv)

        self.PV = pv[rb]/900
        #print(self.PV)
        self.LOAD = [0.793681328 , 0.749258257, 0.726207166, 0.727527632, 0.74872029,  0.778324802, 0.815623879, 0.859231848, 0.937367546, 0.982393792, 0.995875583, 0.99258257,  0.92934694,  0.970803039, 0.989501483, 0.991816374, 1,   0.997522089,
         0.994343191, 0.986029148, 0.953262039, 0.92862965,  0.946692315, 0.952903394]
        
        self.time = 0
        net_ieee69 = pp.create_empty_network()
        # Busses
        buses = pp.create_buses(net_ieee69, 69, name=['Bus %i' % i for i in range(1, 70)], vn_kv=self.vpu ,type='b', zone='IEEE69')

        # Lines 
        line1_2   = pp.create_line_from_parameters(net_ieee69, buses[0],  buses[1] , length_km=1, r_ohm_per_km = 0.0005, x_ohm_per_km = 0.0012, c_nf_per_km = 0, max_i_ka = 9, name='Line 1-2')
        line2_3   = pp.create_line_from_parameters(net_ieee69, buses[1],  buses[2] , length_km=1, r_ohm_per_km = 0.0005, x_ohm_per_km = 0.0012, c_nf_per_km = 0, max_i_ka = 9, name='Line 2-3')
        line3_4   = pp.create_line_from_parameters(net_ieee69, buses[2],  buses[3] , length_km=1, r_ohm_per_km = 0.0015, x_ohm_per_km = 0.0036, c_nf_per_km = 0, max_i_ka = 9, name='Line 3-4')
        line4_5   = pp.create_line_from_parameters(net_ieee69, buses[3],  buses[4] , length_km=1, r_ohm_per_km = 0.0251, x_ohm_per_km = 0.0294, c_nf_per_km = 0, max_i_ka = 9, name='Line 4-5')
        line5_6   = pp.create_line_from_parameters(net_ieee69, buses[4],  buses[5] , length_km=1, r_ohm_per_km = 0.3660, x_ohm_per_km = 0.1864, c_nf_per_km = 0, max_i_ka = 9, name='Line 5-6')
        line6_7   = pp.create_line_from_parameters(net_ieee69, buses[5],  buses[6] , length_km=1, r_ohm_per_km = 0.3811, x_ohm_per_km = 0.1941, c_nf_per_km = 0, max_i_ka = 9, name='Line 6-7')
        line7_8   = pp.create_line_from_parameters(net_ieee69, buses[6],  buses[7] , length_km=1, r_ohm_per_km = 0.0922, x_ohm_per_km = 0.0470, c_nf_per_km = 0, max_i_ka = 9, name='Line 7-8')
        line8_9   = pp.create_line_from_parameters(net_ieee69, buses[7],  buses[8] , length_km=1, r_ohm_per_km = 0.0493, x_ohm_per_km = 0.0251, c_nf_per_km = 0, max_i_ka = 9, name='Line 8-9')
        line9_10  = pp.create_line_from_parameters(net_ieee69, buses[8],  buses[9] , length_km=1, r_ohm_per_km = 0.8190, x_ohm_per_km = 0.2707, c_nf_per_km = 0, max_i_ka = 9, name='Line 9-10')
        line10_11 = pp.create_line_from_parameters(net_ieee69, buses[9],  buses[10], length_km=1, r_ohm_per_km = 0.1872, x_ohm_per_km = 0.0619, c_nf_per_km = 0, max_i_ka = 9, name='Line 10-11')
        line11_12 = pp.create_line_from_parameters(net_ieee69, buses[10], buses[11], length_km=1, r_ohm_per_km = 0.7114, x_ohm_per_km = 0.2351, c_nf_per_km = 0, max_i_ka = 9, name='Line 11-12')
        line12_13 = pp.create_line_from_parameters(net_ieee69, buses[11], buses[12], length_km=1, r_ohm_per_km = 1.0300, x_ohm_per_km = 0.3400, c_nf_per_km = 0, max_i_ka = 9, name='Line 12-13')
        line13_14 = pp.create_line_from_parameters(net_ieee69, buses[12], buses[13], length_km=1, r_ohm_per_km = 1.0440, x_ohm_per_km = 0.3450, c_nf_per_km = 0, max_i_ka = 9, name='Line 13-14')
        line14_15 = pp.create_line_from_parameters(net_ieee69, buses[13], buses[14], length_km=1, r_ohm_per_km = 1.0580, x_ohm_per_km = 0.3496, c_nf_per_km = 0, max_i_ka = 9, name='Line 14-15')
        line15_16 = pp.create_line_from_parameters(net_ieee69, buses[14], buses[15], length_km=1, r_ohm_per_km = 0.1966, x_ohm_per_km = 0.0650, c_nf_per_km = 0, max_i_ka = 9, name='Line 15-16')
        line16_17 = pp.create_line_from_parameters(net_ieee69, buses[15], buses[16], length_km=1, r_ohm_per_km = 0.3744, x_ohm_per_km = 0.1238, c_nf_per_km = 0, max_i_ka = 9, name='Line 16-17')
        line17_18 = pp.create_line_from_parameters(net_ieee69, buses[16], buses[17], length_km=1, r_ohm_per_km = 0.0047, x_ohm_per_km = 0.0016, c_nf_per_km = 0, max_i_ka = 9, name='Line 17-18')
        line18_19 = pp.create_line_from_parameters(net_ieee69, buses[17], buses[18], length_km=1, r_ohm_per_km = 0.3276, x_ohm_per_km = 0.1083, c_nf_per_km = 0, max_i_ka = 9, name='Line 18-19')
        line19_20 = pp.create_line_from_parameters(net_ieee69, buses[18], buses[19], length_km=1, r_ohm_per_km = 0.2106, x_ohm_per_km = 0.0696, c_nf_per_km = 0, max_i_ka = 9, name='Line 19-20')
        line20_21 = pp.create_line_from_parameters(net_ieee69, buses[19], buses[20], length_km=1, r_ohm_per_km = 0.3416, x_ohm_per_km = 0.1129, c_nf_per_km = 0, max_i_ka = 9, name='Line 20-21')
        line21_22 = pp.create_line_from_parameters(net_ieee69, buses[20], buses[21], length_km=1, r_ohm_per_km = 0.0140, x_ohm_per_km = 0.0046, c_nf_per_km = 0, max_i_ka = 9, name='Line 21-22')
        line22_23 = pp.create_line_from_parameters(net_ieee69, buses[21], buses[22], length_km=1, r_ohm_per_km = 0.1591, x_ohm_per_km = 0.0526, c_nf_per_km = 0, max_i_ka = 9, name='Line 21-22')
        line23_24 = pp.create_line_from_parameters(net_ieee69, buses[22], buses[23], length_km=1, r_ohm_per_km = 0.3463, x_ohm_per_km = 0.1145, c_nf_per_km = 0, max_i_ka = 9, name='Line 23-24')
        line24_25 = pp.create_line_from_parameters(net_ieee69, buses[23], buses[24], length_km=1, r_ohm_per_km = 0.7488, x_ohm_per_km = 0.2475, c_nf_per_km = 0, max_i_ka = 9, name='Line 24-25')
        line25_26 = pp.create_line_from_parameters(net_ieee69, buses[24], buses[25], length_km=1, r_ohm_per_km = 0.3089, x_ohm_per_km = 0.1021, c_nf_per_km = 0, max_i_ka = 9, name='Line 25-26')
        line26_27 = pp.create_line_from_parameters(net_ieee69, buses[25], buses[26], length_km=1, r_ohm_per_km = 0.1732, x_ohm_per_km = 0.0572, c_nf_per_km = 0, max_i_ka = 9, name='Line 26-27')
        line3_28  = pp.create_line_from_parameters(net_ieee69, buses[2] , buses[27], length_km=1, r_ohm_per_km = 0.0044, x_ohm_per_km = 0.0108, c_nf_per_km = 0, max_i_ka = 9, name='Line 3-28')
        line28_29 = pp.create_line_from_parameters(net_ieee69, buses[27], buses[28], length_km=1, r_ohm_per_km = 0.0640, x_ohm_per_km = 0.1565, c_nf_per_km = 0, max_i_ka = 9, name='Line 28-29')
        line29_30 = pp.create_line_from_parameters(net_ieee69, buses[28], buses[29], length_km=1, r_ohm_per_km = 0.3978, x_ohm_per_km = 0.1315, c_nf_per_km = 0, max_i_ka = 9, name='Line 29-30')
        line30_31 = pp.create_line_from_parameters(net_ieee69, buses[29], buses[30], length_km=1, r_ohm_per_km = 0.0702, x_ohm_per_km = 0.0232, c_nf_per_km = 0, max_i_ka = 9, name='Line 30-31')
        line31_32 = pp.create_line_from_parameters(net_ieee69, buses[30], buses[31], length_km=1, r_ohm_per_km = 0.3510, x_ohm_per_km = 0.1160, c_nf_per_km = 0, max_i_ka = 9, name='Line 31-32')
        line32_33 = pp.create_line_from_parameters(net_ieee69, buses[31], buses[32], length_km=1, r_ohm_per_km = 0.8390, x_ohm_per_km = 0.2816, c_nf_per_km = 0, max_i_ka = 9, name='Line 32-33')
        line33_34 = pp.create_line_from_parameters(net_ieee69, buses[32], buses[33], length_km=1, r_ohm_per_km = 1.7080, x_ohm_per_km = 0.5646, c_nf_per_km = 0, max_i_ka = 9, name='Line 33-34')
        line34_35 = pp.create_line_from_parameters(net_ieee69, buses[33], buses[34], length_km=1, r_ohm_per_km = 1.4740, x_ohm_per_km = 0.4873, c_nf_per_km = 0, max_i_ka = 9, name='Line 34-35')
        line3_36  = pp.create_line_from_parameters(net_ieee69, buses[2] , buses[35], length_km=1, r_ohm_per_km = 0.0044, x_ohm_per_km = 0.0108, c_nf_per_km = 0, max_i_ka = 9, name='Line 3-36')
        line36_37 = pp.create_line_from_parameters(net_ieee69, buses[35], buses[36], length_km=1, r_ohm_per_km = 0.0640, x_ohm_per_km = 0.1565, c_nf_per_km = 0, max_i_ka = 9, name='Line 36-37')
        line37_38 = pp.create_line_from_parameters(net_ieee69, buses[36], buses[37], length_km=1, r_ohm_per_km = 0.1053, x_ohm_per_km = 0.1230, c_nf_per_km = 0, max_i_ka = 9, name='Line 37-38')
        line38_39 = pp.create_line_from_parameters(net_ieee69, buses[37], buses[38], length_km=1, r_ohm_per_km = 0.0304, x_ohm_per_km = 0.0355, c_nf_per_km = 0, max_i_ka = 9, name='Line 38-39')
        line39_40 = pp.create_line_from_parameters(net_ieee69, buses[38], buses[39], length_km=1, r_ohm_per_km = 0.0018, x_ohm_per_km = 0.0021, c_nf_per_km = 0, max_i_ka = 9, name='Line 39-40')
        line40_41 = pp.create_line_from_parameters(net_ieee69, buses[39], buses[40], length_km=1, r_ohm_per_km = 0.7283, x_ohm_per_km = 0.8509, c_nf_per_km = 0, max_i_ka = 9, name='Line 40-41')
        line41_42 = pp.create_line_from_parameters(net_ieee69, buses[40], buses[41], length_km=1, r_ohm_per_km = 0.3100, x_ohm_per_km = 0.3623, c_nf_per_km = 0, max_i_ka = 9, name='Line 41-42')
        line42_43 = pp.create_line_from_parameters(net_ieee69, buses[41], buses[42], length_km=1, r_ohm_per_km = 0.0410, x_ohm_per_km = 0.0478, c_nf_per_km = 0, max_i_ka = 9, name='Line 42-43')
        line43_44 = pp.create_line_from_parameters(net_ieee69, buses[42], buses[43], length_km=1, r_ohm_per_km = 0.0092, x_ohm_per_km = 0.0116, c_nf_per_km = 0, max_i_ka = 9, name='Line 43-44')
        line44_45 = pp.create_line_from_parameters(net_ieee69, buses[43], buses[44], length_km=1, r_ohm_per_km = 0.1089, x_ohm_per_km = 0.1373, c_nf_per_km = 0, max_i_ka = 9, name='Line 44-45')
        line45_46 = pp.create_line_from_parameters(net_ieee69, buses[44], buses[45], length_km=1, r_ohm_per_km = 0.0009, x_ohm_per_km = 0.0012, c_nf_per_km = 0, max_i_ka = 9, name='Line 45-46')
        line4_47  = pp.create_line_from_parameters(net_ieee69, buses[3] , buses[46], length_km=1, r_ohm_per_km = 0.0034, x_ohm_per_km = 0.0084, c_nf_per_km = 0, max_i_ka = 9, name='Line 4-47')
        line47_48 = pp.create_line_from_parameters(net_ieee69, buses[46], buses[47], length_km=1, r_ohm_per_km = 0.0851, x_ohm_per_km = 0.2083, c_nf_per_km = 0, max_i_ka = 9, name='Line 47-48')
        line48_49 = pp.create_line_from_parameters(net_ieee69, buses[47], buses[48], length_km=1, r_ohm_per_km = 0.2898, x_ohm_per_km = 0.7091, c_nf_per_km = 0, max_i_ka = 9, name='Line 48-49')
        line49_50 = pp.create_line_from_parameters(net_ieee69, buses[48], buses[49], length_km=1, r_ohm_per_km = 0.0822, x_ohm_per_km = 0.2011, c_nf_per_km = 0, max_i_ka = 9, name='Line 49-50')
        line8_51  = pp.create_line_from_parameters(net_ieee69, buses[7] , buses[50], length_km=1, r_ohm_per_km = 0.0928, x_ohm_per_km = 0.0473, c_nf_per_km = 0, max_i_ka = 9, name='Line 8-51')
        line51_52 = pp.create_line_from_parameters(net_ieee69, buses[50], buses[51], length_km=1, r_ohm_per_km = 0.3319, x_ohm_per_km = 0.1114, c_nf_per_km = 0, max_i_ka = 9, name='Line 51-52')
        line52_53 = pp.create_line_from_parameters(net_ieee69, buses[51], buses[52], length_km=1, r_ohm_per_km = 0.1740, x_ohm_per_km = 0.1114, c_nf_per_km = 0, max_i_ka = 9, name='Line 52-53')
        line53_54 = pp.create_line_from_parameters(net_ieee69, buses[52], buses[53], length_km=1, r_ohm_per_km = 0.2030, x_ohm_per_km = 0.1034, c_nf_per_km = 0, max_i_ka = 9, name='Line 53-54')
        line54_55 = pp.create_line_from_parameters(net_ieee69, buses[53], buses[54], length_km=1, r_ohm_per_km = 0.2842, x_ohm_per_km = 0.1447, c_nf_per_km = 0, max_i_ka = 9, name='Line 54-55')
        line55_56 = pp.create_line_from_parameters(net_ieee69, buses[54], buses[55], length_km=1, r_ohm_per_km = 0.2813, x_ohm_per_km = 0.1433, c_nf_per_km = 0, max_i_ka = 9, name='Line 55-56')
        line56_57 = pp.create_line_from_parameters(net_ieee69, buses[55], buses[56], length_km=1, r_ohm_per_km = 1.5900, x_ohm_per_km = 0.5337, c_nf_per_km = 0, max_i_ka = 9, name='Line 56-57')
        line57_58 = pp.create_line_from_parameters(net_ieee69, buses[56], buses[57], length_km=1, r_ohm_per_km = 0.7837, x_ohm_per_km = 0.2630, c_nf_per_km = 0, max_i_ka = 9, name='Line 57-58')
        line58_59 = pp.create_line_from_parameters(net_ieee69, buses[57], buses[58], length_km=1, r_ohm_per_km = 0.3042, x_ohm_per_km = 0.1006, c_nf_per_km = 0, max_i_ka = 9, name='Line 58-59')
        line59_60 = pp.create_line_from_parameters(net_ieee69, buses[58], buses[59], length_km=1, r_ohm_per_km = 0.3861, x_ohm_per_km = 0.1172, c_nf_per_km = 0, max_i_ka = 9, name='Line 59-60')
        line60_61 = pp.create_line_from_parameters(net_ieee69, buses[59], buses[60], length_km=1, r_ohm_per_km = 0.5075, x_ohm_per_km = 0.2585, c_nf_per_km = 0, max_i_ka = 9, name='Line 60-61')
        line61_62 = pp.create_line_from_parameters(net_ieee69, buses[60], buses[61], length_km=1, r_ohm_per_km = 0.0974, x_ohm_per_km = 0.0496, c_nf_per_km = 0, max_i_ka = 9, name='Line 61-62')
        line62_63 = pp.create_line_from_parameters(net_ieee69, buses[61], buses[62], length_km=1, r_ohm_per_km = 0.1450, x_ohm_per_km = 0.0738, c_nf_per_km = 0, max_i_ka = 9, name='Line 62-63')
        line63_64 = pp.create_line_from_parameters(net_ieee69, buses[62], buses[63], length_km=1, r_ohm_per_km = 0.7105, x_ohm_per_km = 0.3619, c_nf_per_km = 0, max_i_ka = 9, name='Line 63-64')
        line64_65 = pp.create_line_from_parameters(net_ieee69, buses[63], buses[64], length_km=1, r_ohm_per_km = 1.0410, x_ohm_per_km = 0.5302, c_nf_per_km = 0, max_i_ka = 9, name='Line 64-65')
        line11_66 = pp.create_line_from_parameters(net_ieee69, buses[10], buses[65], length_km=1, r_ohm_per_km = 0.2012, x_ohm_per_km = 0.0611, c_nf_per_km = 0, max_i_ka = 9, name='Line 11-66')
        line66_67 = pp.create_line_from_parameters(net_ieee69, buses[65], buses[66], length_km=1, r_ohm_per_km = 0.0047, x_ohm_per_km = 0.0014, c_nf_per_km = 0, max_i_ka = 9, name='Line 66-67')
        line12_68 = pp.create_line_from_parameters(net_ieee69, buses[11], buses[67], length_km=1, r_ohm_per_km = 0.7394, x_ohm_per_km = 0.2444, c_nf_per_km = 0, max_i_ka = 9, name='Line 12-68')
        line68_69 = pp.create_line_from_parameters(net_ieee69, buses[67], buses[68], length_km=1, r_ohm_per_km = 0.0047, x_ohm_per_km = 0.0016, c_nf_per_km = 0, max_i_ka = 9, name='Line 68-69')


            # Switches
            # S2
       
            # Loads
            # Residential
        pp.create_load(net_ieee69, buses[5] , p_mw = 2.6  , q_mvar = 2.2  , scaling = self.LOAD[self.time]*0.001, name='Load R6')
        pp.create_load(net_ieee69, buses[6] , p_mw = 40.4 , q_mvar = 30   , scaling = self.LOAD[self.time]*0.001, name='Load R7')
        pp.create_load(net_ieee69, buses[7] , p_mw = 75   , q_mvar = 54   , scaling = self.LOAD[self.time]*0.001, name='Load R8')
        pp.create_load(net_ieee69, buses[8] , p_mw = 30   , q_mvar = 22   , scaling = self.LOAD[self.time]*0.001, name='Load R9')
        pp.create_load(net_ieee69, buses[9] , p_mw = 28   , q_mvar = 19   , scaling = self.LOAD[self.time]*0.001, name='Load R10')
        pp.create_load(net_ieee69, buses[10], p_mw = 145  , q_mvar = 104  , scaling = self.LOAD[self.time]*0.001, name='Load R11')
        pp.create_load(net_ieee69, buses[11], p_mw = 145  , q_mvar = 104  , scaling = self.LOAD[self.time]*0.001, name='Load R12')
        pp.create_load(net_ieee69, buses[12], p_mw = 8    , q_mvar = 5    , scaling = self.LOAD[self.time]*0.001, name='Load R13')
        pp.create_load(net_ieee69, buses[13], p_mw = 8    , q_mvar = 5.5  , scaling = self.LOAD[self.time]*0.001, name='Load R14')        
        pp.create_load(net_ieee69, buses[15], p_mw = 45.5 , q_mvar = 30   , scaling = self.LOAD[self.time]*0.001, name='Load R16')
        pp.create_load(net_ieee69, buses[16], p_mw = 60   , q_mvar = 35   , scaling = self.LOAD[self.time]*0.001, name='Load R17')
        pp.create_load(net_ieee69, buses[17], p_mw = 60   , q_mvar = 35   , scaling = self.LOAD[self.time]*0.001, name='Load R18')
        pp.create_load(net_ieee69, buses[19], p_mw = 1    , q_mvar = 0.6  , scaling = self.LOAD[self.time]*0.001, name='Load R20')
        pp.create_load(net_ieee69, buses[20], p_mw = 114  , q_mvar = 81   , scaling = self.LOAD[self.time]*0.001, name='Load R21')
        pp.create_load(net_ieee69, buses[21], p_mw = 5    , q_mvar = 3.5  , scaling = self.LOAD[self.time]*0.001, name='Load R22')
        pp.create_load(net_ieee69, buses[23], p_mw = 28   , q_mvar = 20   , scaling = self.LOAD[self.time]*0.001, name='Load R24')
        pp.create_load(net_ieee69, buses[25], p_mw = 14   , q_mvar = 10   , scaling = self.LOAD[self.time]*0.001, name='Load R26')
        pp.create_load(net_ieee69, buses[26], p_mw = 14   , q_mvar = 10   , scaling = self.LOAD[self.time]*0.001, name='Load R27')
        pp.create_load(net_ieee69, buses[27], p_mw = 26   , q_mvar = 18.6 , scaling = self.LOAD[self.time]*0.001, name='Load R28')
        pp.create_load(net_ieee69, buses[28], p_mw = 26   , q_mvar = 18.6 , scaling = self.LOAD[self.time]*0.001, name='Load R29')        
        pp.create_load(net_ieee69, buses[32], p_mw = 14   , q_mvar = 10   , scaling = self.LOAD[self.time]*0.001, name='Load R33')
        pp.create_load(net_ieee69, buses[33], p_mw = 9.5  , q_mvar = 14   , scaling = self.LOAD[self.time]*0.001, name='Load R34')
        pp.create_load(net_ieee69, buses[34], p_mw = 6    , q_mvar = 4    , scaling = self.LOAD[self.time]*0.001, name='Load R35')
        pp.create_load(net_ieee69, buses[35], p_mw = 26   , q_mvar = 18.55, scaling = self.LOAD[self.time]*0.001, name='Load R36')
        pp.create_load(net_ieee69, buses[36], p_mw = 26   , q_mvar = 18.55, scaling = self.LOAD[self.time]*0.001, name='Load R37')
        pp.create_load(net_ieee69, buses[38], p_mw = 24   , q_mvar = 17   , scaling = self.LOAD[self.time]*0.001, name='Load R39')
        pp.create_load(net_ieee69, buses[39], p_mw = 24   , q_mvar = 17   , scaling = self.LOAD[self.time]*0.001, name='Load R40')
        pp.create_load(net_ieee69, buses[40], p_mw = 1.2  , q_mvar = 1    , scaling = self.LOAD[self.time]*0.001, name='Load R41')
        pp.create_load(net_ieee69, buses[42], p_mw = 6    , q_mvar = 4.3  , scaling = self.LOAD[self.time]*0.001, name='Load R43')
        pp.create_load(net_ieee69, buses[44], p_mw = 39.22, q_mvar = 26.3 , scaling = self.LOAD[self.time]*0.001, name='Load R44')
        pp.create_load(net_ieee69, buses[45], p_mw = 39.22, q_mvar = 26.3 , scaling = self.LOAD[self.time]*0.001, name='Load R45')
        pp.create_load(net_ieee69, buses[47], p_mw = 79   , q_mvar = 56.4 , scaling = self.LOAD[self.time]*0.001, name='Load R46')
        pp.create_load(net_ieee69, buses[48], p_mw = 384.7, q_mvar = 274.5, scaling = self.LOAD[self.time]*0.001, name='Load R48')
        pp.create_load(net_ieee69, buses[49], p_mw = 384.7, q_mvar = 274.5, scaling = self.LOAD[self.time]*0.001, name='Load R50')
        pp.create_load(net_ieee69, buses[50], p_mw = 40.5 , q_mvar = 28.3 , scaling = self.LOAD[self.time]*0.001, name='Load R51')
        pp.create_load(net_ieee69, buses[51], p_mw = 3.6  , q_mvar = 2.7  , scaling = self.LOAD[self.time]*0.001, name='Load R52')
        pp.create_load(net_ieee69, buses[52], p_mw = 4.35 , q_mvar = 3.5  , scaling = self.LOAD[self.time]*0.001, name='Load R53')
        pp.create_load(net_ieee69, buses[53], p_mw = 26.4 , q_mvar = 19   , scaling = self.LOAD[self.time]*0.001, name='Load R54')
        pp.create_load(net_ieee69, buses[54], p_mw = 24   , q_mvar = 17.2 , scaling = self.LOAD[self.time]*0.001, name='Load R55')
        pp.create_load(net_ieee69, buses[58], p_mw = 100  , q_mvar = 72   , scaling = self.LOAD[self.time]*0.001, name='Load R59')
        pp.create_load(net_ieee69, buses[60], p_mw = 1244 , q_mvar = 888  , scaling = self.LOAD[self.time]*0.001, name='Load R61')
        pp.create_load(net_ieee69, buses[61], p_mw = 32   , q_mvar = 23   , scaling = self.LOAD[self.time]*0.001, name='Load R62')
        pp.create_load(net_ieee69, buses[63], p_mw = 227  , q_mvar = 162  , scaling = self.LOAD[self.time]*0.001, name='Load R64')
        pp.create_load(net_ieee69, buses[64], p_mw = 59   , q_mvar = 42   , scaling = self.LOAD[self.time]*0.001, name='Load R65')
        pp.create_load(net_ieee69, buses[65], p_mw = 18   , q_mvar = 13   , scaling = self.LOAD[self.time]*0.001, name='Load R66')
        pp.create_load(net_ieee69, buses[66], p_mw = 18   , q_mvar = 13   , scaling = self.LOAD[self.time]*0.001, name='Load R67')
        pp.create_load(net_ieee69, buses[67], p_mw = 28   , q_mvar = 20   , scaling = self.LOAD[self.time]*0.001, name='Load R68')
        pp.create_load(net_ieee69, buses[68], p_mw = 28   , q_mvar = 20   , scaling = self.LOAD[self.time]*0.001, name='Load R69')

            # Optional distributed energy recources


        '''
        pp.create_sgen(net_ieee33, buses[2] , 0.5*self.PV[0] , q_mvar=0,  name='PV 3', type='PV')
        pp.create_sgen(net_ieee33, buses[7] , 0.5*self.PV[0] , q_mvar=0,  name='PV 8', type='PV')
        pp.create_sgen(net_ieee33, buses[13], 0.5*self.PV[0] , q_mvar=0,  name='PV 14', type='PV')
        pp.create_sgen(net_ieee33, buses[24], 0.5*self.PV[0] , q_mvar=0,  name='PV 25', type='PV')
        pp.create_sgen(net_ieee33, buses[29], 0.5*self.PV[0] , q_mvar=0,  name='PV 30', type='PV')
        pp.create_sgen(net_ieee33, buses[30], 0.5*self.PV[0] , q_mvar=0,  name='PV 31', type='PV')
        

        pp.create_shunt(net_ieee33, buses[17], 0)
        pp.create_storage(net_ieee33, buses[17], p_mw = 0, max_e_mwh = 3, soc_percent = 0.5, min_e_mwh = 0)

        pp.create_shunt(net_ieee33, buses[32], 0)
        pp.create_storage(net_ieee33, buses[32], p_mw = 0, max_e_mwh = 3, soc_percent = 0.5, min_e_mwh = 0)
        '''


        # Ext-Grid

        pp.create_ext_grid(net_ieee69, buses[0],max_p_mw=100, min_p_mw=0)
        #create generators        
        pp.create_gen(net_ieee69, buses[10], p_mw=0, min_p_mw=0, max_p_mw=1, min_q_mvar=-50, max_q_mvar=50, vm_pu=1.0, controllable=True)
        pp.create_gen(net_ieee69, buses[17], p_mw=0, min_p_mw=0, max_p_mw=1, min_q_mvar=-50, max_q_mvar=50, vm_pu=1.0, controllable=True)
        pp.create_gen(net_ieee69, buses[60], p_mw=0, min_p_mw=0, max_p_mw=1, min_q_mvar=-50, max_q_mvar=50, vm_pu=1.0, controllable=True)

        # 발전 비용 곡선
        pp.create_poly_cost(net_ieee69, 0, "gen", cp2_eur_per_mw2 = 15.5,  cp1_eur_per_mw = 65.6)
        pp.create_poly_cost(net_ieee69, 1, "gen", cp2_eur_per_mw2 = 13.32, cp1_eur_per_mw = 50.2)
        pp.create_poly_cost(net_ieee69, 2, "gen", cp2_eur_per_mw2 = 16.88, cp1_eur_per_mw = 40.1)
        pp.create_poly_cost(net_ieee69, 0, "ext_grid", cp1_eur_per_mw = 80)
        #ess
        #dcline
        #pp.create_storage(net_ieee33, buses[17], p_mw = 0, max_e_mwh = 3, soc_percent = 0.5, min_e_mwh = 0)
        #pp.create_storage(net_ieee33, buses[32], p_mw = 0, max_e_mwh = 3, soc_percent = 0.5, min_e_mwh = 0)


        pp.runopp(net_ieee69, delta=1e-16)

#        return PV, WT, LOAD, Pline, Pbus, radial
        self.busv_pu = np.array([net_ieee69.res_bus.vm_pu])
        self.line_percent = np.array([net_ieee69.res_line.p_to_mw]) 
        self.soc = np.array([net_ieee69.storage.soc_percent])

        self.p_mw = np.array([net_ieee69.storage.p_mw])
        # 래디얼이면 rad = 1

        sta1 = np.array([net_ieee69.res_bus.vm_pu[1] , net_ieee69.res_bus.vm_pu[2] , net_ieee69.res_bus.vm_pu[3] , net_ieee69.res_bus.vm_pu[4] , net_ieee69.res_bus.vm_pu[5] , net_ieee69.res_bus.vm_pu[6]])
        sta2 = np.array([net_ieee69.res_bus.vm_pu[66], net_ieee69.res_bus.vm_pu[67], net_ieee69.res_bus.vm_pu[68], net_ieee69.res_bus.vm_pu[13], net_ieee69.res_bus.vm_pu[14], net_ieee69.res_bus.vm_pu[15]])
        sta3 = np.array([net_ieee69.res_bus.vm_pu[63], net_ieee69.res_bus.vm_pu[64], net_ieee69.res_bus.vm_pu[65], net_ieee69.res_bus.vm_pu[66], net_ieee69.res_bus.vm_pu[67], net_ieee69.res_bus.vm_pu[68]])
        sta4 = np.array([net_ieee69.res_bus.vm_pu[12], net_ieee69.res_bus.vm_pu[13], net_ieee69.res_bus.vm_pu[14], net_ieee69.res_bus.vm_pu[15], net_ieee69.res_bus.vm_pu[16], net_ieee69.res_bus.vm_pu[17]])
        sta5 = np.array([net_ieee69.res_bus.vm_pu[24], net_ieee69.res_bus.vm_pu[25], net_ieee69.res_bus.vm_pu[26], net_ieee69.res_bus.vm_pu[51], net_ieee69.res_bus.vm_pu[52], net_ieee69.res_bus.vm_pu[53]])
        sta6 = np.array([net_ieee69.res_bus.vm_pu[21], net_ieee69.res_bus.vm_pu[22], net_ieee69.res_bus.vm_pu[23], net_ieee69.res_bus.vm_pu[24], net_ieee69.res_bus.vm_pu[25], net_ieee69.res_bus.vm_pu[26]])
        sta7 = np.array([net_ieee69.res_bus.vm_pu[48], net_ieee69.res_bus.vm_pu[49], net_ieee69.res_bus.vm_pu[50], net_ieee69.res_bus.vm_pu[51], net_ieee69.res_bus.vm_pu[52], net_ieee69.res_bus.vm_pu[53]])
        state1 = np.hstack([self.time, 0,0,0,0.793681328 , 0.749258257, 0.726207166, sta1 ]).flatten()
        state2 = np.hstack([self.time, 0,0,0,0.793681328 , 0.749258257, 0.726207166, sta2 ]).flatten()
        state3 = np.hstack([self.time, 0,0,0,0.793681328 , 0.749258257, 0.726207166, sta3 ]).flatten()
        state4 = np.hstack([self.time, 0,0,0,0.793681328 , 0.749258257, 0.726207166, sta4 ]).flatten()
        state5 = np.hstack([self.time, 0,0,0,0.793681328 , 0.749258257, 0.726207166, sta5 ]).flatten()
        state6 = np.hstack([self.time, 0,0,0,0.793681328 , 0.749258257, 0.726207166, sta6 ]).flatten()
        state7 = np.hstack([self.time, 0,0,0,0.793681328 , 0.749258257, 0.726207166, sta7 ]).flatten()
        
       
        states = [state1, state2, state3, state4, state5, state6, state7]
        
        return states 

    def step(self, acts1, acts2, acts3, acts4, acts5, acts6, acts7):
        a1 = acts1
        a2 = acts2
        a3 = acts3
        a4 = acts4
        a5 = acts5
        a6 = acts6
        a7 = acts7

        self.rwd = np.zeros(shape=(7,))
        terminal = False
        n_state, rwd, ifo , ifos, ifov= self._step(a1, a2, a3, a4, a5, a6, a7)

        self.rwd[0] = rwd[0]
        self.rwd[1] = rwd[1]
        self.rwd[2] = rwd[2]
        self.rwd[3] = rwd[3]
        self.rwd[4] = rwd[4]
        self.rwd[5] = rwd[5]
        self.rwd[6] = rwd[6]

             
        next_state = n_state
        total_reward = [self.rwd[0],self.rwd[1],self.rwd[2],self.rwd[3],self.rwd[4],self.rwd[5],self.rwd[6]]
        terminals = [False,False,False,False,False,False,False]
        info = ifo
        infos = ifos
        infov =ifov

        self.time = self.time + 1
        if self.time == self.MaxTime:
            self.time = 0
            
            terminals = [True,True,True,True,True,True,True]

        return next_state, total_reward, terminals, info, infos, infov


    def noact(self, act1, act2 , act3 , act4, act5 , act6 , act7):
        total_reward = 0        
        # Busses
        net_ieee69 = pp.create_empty_network()
        # Busses
        buses = pp.create_buses(net_ieee69, 69, name=['Bus %i' % i for i in range(1, 70)], vn_kv=self.vpu ,type='b', zone='IEEE69')

        # Lines 
        line1_2   = pp.create_line_from_parameters(net_ieee69, buses[0],  buses[1] , length_km=1, r_ohm_per_km = 0.0005, x_ohm_per_km = 0.0012, c_nf_per_km = 0, max_i_ka = 9, name='Line 1-2')
        line2_3   = pp.create_line_from_parameters(net_ieee69, buses[1],  buses[2] , length_km=1, r_ohm_per_km = 0.0005, x_ohm_per_km = 0.0012, c_nf_per_km = 0, max_i_ka = 9, name='Line 2-3')
        line3_4   = pp.create_line_from_parameters(net_ieee69, buses[2],  buses[3] , length_km=1, r_ohm_per_km = 0.0015, x_ohm_per_km = 0.0036, c_nf_per_km = 0, max_i_ka = 9, name='Line 3-4')
        line4_5   = pp.create_line_from_parameters(net_ieee69, buses[3],  buses[4] , length_km=1, r_ohm_per_km = 0.0251, x_ohm_per_km = 0.0294, c_nf_per_km = 0, max_i_ka = 9, name='Line 4-5')
        line5_6   = pp.create_line_from_parameters(net_ieee69, buses[4],  buses[5] , length_km=1, r_ohm_per_km = 0.3660, x_ohm_per_km = 0.1864, c_nf_per_km = 0, max_i_ka = 9, name='Line 5-6')
        line6_7   = pp.create_line_from_parameters(net_ieee69, buses[5],  buses[6] , length_km=1, r_ohm_per_km = 0.3811, x_ohm_per_km = 0.1941, c_nf_per_km = 0, max_i_ka = 9, name='Line 6-7')
        line7_8   = pp.create_line_from_parameters(net_ieee69, buses[6],  buses[7] , length_km=1, r_ohm_per_km = 0.0922, x_ohm_per_km = 0.0470, c_nf_per_km = 0, max_i_ka = 9, name='Line 7-8')
        line8_9   = pp.create_line_from_parameters(net_ieee69, buses[7],  buses[8] , length_km=1, r_ohm_per_km = 0.0493, x_ohm_per_km = 0.0251, c_nf_per_km = 0, max_i_ka = 9, name='Line 8-9')
        line9_10  = pp.create_line_from_parameters(net_ieee69, buses[8],  buses[9] , length_km=1, r_ohm_per_km = 0.8190, x_ohm_per_km = 0.2707, c_nf_per_km = 0, max_i_ka = 9, name='Line 9-10')
        line10_11 = pp.create_line_from_parameters(net_ieee69, buses[9],  buses[10], length_km=1, r_ohm_per_km = 0.1872, x_ohm_per_km = 0.0619, c_nf_per_km = 0, max_i_ka = 9, name='Line 10-11')
        line11_12 = pp.create_line_from_parameters(net_ieee69, buses[10], buses[11], length_km=1, r_ohm_per_km = 0.7114, x_ohm_per_km = 0.2351, c_nf_per_km = 0, max_i_ka = 9, name='Line 11-12')
        line12_13 = pp.create_line_from_parameters(net_ieee69, buses[11], buses[12], length_km=1, r_ohm_per_km = 1.0300, x_ohm_per_km = 0.3400, c_nf_per_km = 0, max_i_ka = 9, name='Line 12-13')
        line13_14 = pp.create_line_from_parameters(net_ieee69, buses[12], buses[13], length_km=1, r_ohm_per_km = 1.0440, x_ohm_per_km = 0.3450, c_nf_per_km = 0, max_i_ka = 9, name='Line 13-14')
        line14_15 = pp.create_line_from_parameters(net_ieee69, buses[13], buses[14], length_km=1, r_ohm_per_km = 1.0580, x_ohm_per_km = 0.3496, c_nf_per_km = 0, max_i_ka = 9, name='Line 14-15')
        line15_16 = pp.create_line_from_parameters(net_ieee69, buses[14], buses[15], length_km=1, r_ohm_per_km = 0.1966, x_ohm_per_km = 0.0650, c_nf_per_km = 0, max_i_ka = 9, name='Line 15-16')
        line16_17 = pp.create_line_from_parameters(net_ieee69, buses[15], buses[16], length_km=1, r_ohm_per_km = 0.3744, x_ohm_per_km = 0.1238, c_nf_per_km = 0, max_i_ka = 9, name='Line 16-17')
        line17_18 = pp.create_line_from_parameters(net_ieee69, buses[16], buses[17], length_km=1, r_ohm_per_km = 0.0047, x_ohm_per_km = 0.0016, c_nf_per_km = 0, max_i_ka = 9, name='Line 17-18')
        line18_19 = pp.create_line_from_parameters(net_ieee69, buses[17], buses[18], length_km=1, r_ohm_per_km = 0.3276, x_ohm_per_km = 0.1083, c_nf_per_km = 0, max_i_ka = 9, name='Line 18-19')
        line19_20 = pp.create_line_from_parameters(net_ieee69, buses[18], buses[19], length_km=1, r_ohm_per_km = 0.2106, x_ohm_per_km = 0.0696, c_nf_per_km = 0, max_i_ka = 9, name='Line 19-20')
        line20_21 = pp.create_line_from_parameters(net_ieee69, buses[19], buses[20], length_km=1, r_ohm_per_km = 0.3416, x_ohm_per_km = 0.1129, c_nf_per_km = 0, max_i_ka = 9, name='Line 20-21')
        line21_22 = pp.create_line_from_parameters(net_ieee69, buses[20], buses[21], length_km=1, r_ohm_per_km = 0.0140, x_ohm_per_km = 0.0046, c_nf_per_km = 0, max_i_ka = 9, name='Line 21-22')
        line22_23 = pp.create_line_from_parameters(net_ieee69, buses[21], buses[22], length_km=1, r_ohm_per_km = 0.1591, x_ohm_per_km = 0.0526, c_nf_per_km = 0, max_i_ka = 9, name='Line 21-22')
        line23_24 = pp.create_line_from_parameters(net_ieee69, buses[22], buses[23], length_km=1, r_ohm_per_km = 0.3463, x_ohm_per_km = 0.1145, c_nf_per_km = 0, max_i_ka = 9, name='Line 23-24')
        line24_25 = pp.create_line_from_parameters(net_ieee69, buses[23], buses[24], length_km=1, r_ohm_per_km = 0.7488, x_ohm_per_km = 0.2475, c_nf_per_km = 0, max_i_ka = 9, name='Line 24-25')
        line25_26 = pp.create_line_from_parameters(net_ieee69, buses[24], buses[25], length_km=1, r_ohm_per_km = 0.3089, x_ohm_per_km = 0.1021, c_nf_per_km = 0, max_i_ka = 9, name='Line 25-26')
        line26_27 = pp.create_line_from_parameters(net_ieee69, buses[25], buses[26], length_km=1, r_ohm_per_km = 0.1732, x_ohm_per_km = 0.0572, c_nf_per_km = 0, max_i_ka = 9, name='Line 26-27')
        line3_28  = pp.create_line_from_parameters(net_ieee69, buses[2] , buses[27], length_km=1, r_ohm_per_km = 0.0044, x_ohm_per_km = 0.0108, c_nf_per_km = 0, max_i_ka = 9, name='Line 3-28')
        line28_29 = pp.create_line_from_parameters(net_ieee69, buses[27], buses[28], length_km=1, r_ohm_per_km = 0.0640, x_ohm_per_km = 0.1565, c_nf_per_km = 0, max_i_ka = 9, name='Line 28-29')
        line29_30 = pp.create_line_from_parameters(net_ieee69, buses[28], buses[29], length_km=1, r_ohm_per_km = 0.3978, x_ohm_per_km = 0.1315, c_nf_per_km = 0, max_i_ka = 9, name='Line 29-30')
        line30_31 = pp.create_line_from_parameters(net_ieee69, buses[29], buses[30], length_km=1, r_ohm_per_km = 0.0702, x_ohm_per_km = 0.0232, c_nf_per_km = 0, max_i_ka = 9, name='Line 30-31')
        line31_32 = pp.create_line_from_parameters(net_ieee69, buses[30], buses[31], length_km=1, r_ohm_per_km = 0.3510, x_ohm_per_km = 0.1160, c_nf_per_km = 0, max_i_ka = 9, name='Line 31-32')
        line32_33 = pp.create_line_from_parameters(net_ieee69, buses[31], buses[32], length_km=1, r_ohm_per_km = 0.8390, x_ohm_per_km = 0.2816, c_nf_per_km = 0, max_i_ka = 9, name='Line 32-33')
        line33_34 = pp.create_line_from_parameters(net_ieee69, buses[32], buses[33], length_km=1, r_ohm_per_km = 1.7080, x_ohm_per_km = 0.5646, c_nf_per_km = 0, max_i_ka = 9, name='Line 33-34')
        line34_35 = pp.create_line_from_parameters(net_ieee69, buses[33], buses[34], length_km=1, r_ohm_per_km = 1.4740, x_ohm_per_km = 0.4873, c_nf_per_km = 0, max_i_ka = 9, name='Line 34-35')
        line3_36  = pp.create_line_from_parameters(net_ieee69, buses[2] , buses[35], length_km=1, r_ohm_per_km = 0.0044, x_ohm_per_km = 0.0108, c_nf_per_km = 0, max_i_ka = 9, name='Line 3-36')
        line36_37 = pp.create_line_from_parameters(net_ieee69, buses[35], buses[36], length_km=1, r_ohm_per_km = 0.0640, x_ohm_per_km = 0.1565, c_nf_per_km = 0, max_i_ka = 9, name='Line 36-37')
        line37_38 = pp.create_line_from_parameters(net_ieee69, buses[36], buses[37], length_km=1, r_ohm_per_km = 0.1053, x_ohm_per_km = 0.1230, c_nf_per_km = 0, max_i_ka = 9, name='Line 37-38')
        line38_39 = pp.create_line_from_parameters(net_ieee69, buses[37], buses[38], length_km=1, r_ohm_per_km = 0.0304, x_ohm_per_km = 0.0355, c_nf_per_km = 0, max_i_ka = 9, name='Line 38-39')
        line39_40 = pp.create_line_from_parameters(net_ieee69, buses[38], buses[39], length_km=1, r_ohm_per_km = 0.0018, x_ohm_per_km = 0.0021, c_nf_per_km = 0, max_i_ka = 9, name='Line 39-40')
        line40_41 = pp.create_line_from_parameters(net_ieee69, buses[39], buses[40], length_km=1, r_ohm_per_km = 0.7283, x_ohm_per_km = 0.8509, c_nf_per_km = 0, max_i_ka = 9, name='Line 40-41')
        line41_42 = pp.create_line_from_parameters(net_ieee69, buses[40], buses[41], length_km=1, r_ohm_per_km = 0.3100, x_ohm_per_km = 0.3623, c_nf_per_km = 0, max_i_ka = 9, name='Line 41-42')
        line42_43 = pp.create_line_from_parameters(net_ieee69, buses[41], buses[42], length_km=1, r_ohm_per_km = 0.0410, x_ohm_per_km = 0.0478, c_nf_per_km = 0, max_i_ka = 9, name='Line 42-43')
        line43_44 = pp.create_line_from_parameters(net_ieee69, buses[42], buses[43], length_km=1, r_ohm_per_km = 0.0092, x_ohm_per_km = 0.0116, c_nf_per_km = 0, max_i_ka = 9, name='Line 43-44')
        line44_45 = pp.create_line_from_parameters(net_ieee69, buses[43], buses[44], length_km=1, r_ohm_per_km = 0.1089, x_ohm_per_km = 0.1373, c_nf_per_km = 0, max_i_ka = 9, name='Line 44-45')
        line45_46 = pp.create_line_from_parameters(net_ieee69, buses[44], buses[45], length_km=1, r_ohm_per_km = 0.0009, x_ohm_per_km = 0.0012, c_nf_per_km = 0, max_i_ka = 9, name='Line 45-46')
        line4_47  = pp.create_line_from_parameters(net_ieee69, buses[3] , buses[46], length_km=1, r_ohm_per_km = 0.0034, x_ohm_per_km = 0.0084, c_nf_per_km = 0, max_i_ka = 9, name='Line 4-47')
        line47_48 = pp.create_line_from_parameters(net_ieee69, buses[46], buses[47], length_km=1, r_ohm_per_km = 0.0851, x_ohm_per_km = 0.2083, c_nf_per_km = 0, max_i_ka = 9, name='Line 47-48')
        line48_49 = pp.create_line_from_parameters(net_ieee69, buses[47], buses[48], length_km=1, r_ohm_per_km = 0.2898, x_ohm_per_km = 0.7091, c_nf_per_km = 0, max_i_ka = 9, name='Line 48-49')
        line49_50 = pp.create_line_from_parameters(net_ieee69, buses[48], buses[49], length_km=1, r_ohm_per_km = 0.0822, x_ohm_per_km = 0.2011, c_nf_per_km = 0, max_i_ka = 9, name='Line 49-50')
        line8_51  = pp.create_line_from_parameters(net_ieee69, buses[7] , buses[50], length_km=1, r_ohm_per_km = 0.0928, x_ohm_per_km = 0.0473, c_nf_per_km = 0, max_i_ka = 9, name='Line 8-51')
        line51_52 = pp.create_line_from_parameters(net_ieee69, buses[50], buses[51], length_km=1, r_ohm_per_km = 0.3319, x_ohm_per_km = 0.1114, c_nf_per_km = 0, max_i_ka = 9, name='Line 51-52')
        line52_53 = pp.create_line_from_parameters(net_ieee69, buses[51], buses[52], length_km=1, r_ohm_per_km = 0.1740, x_ohm_per_km = 0.1114, c_nf_per_km = 0, max_i_ka = 9, name='Line 52-53')
        line53_54 = pp.create_line_from_parameters(net_ieee69, buses[52], buses[53], length_km=1, r_ohm_per_km = 0.2030, x_ohm_per_km = 0.1034, c_nf_per_km = 0, max_i_ka = 9, name='Line 53-54')
        line54_55 = pp.create_line_from_parameters(net_ieee69, buses[53], buses[54], length_km=1, r_ohm_per_km = 0.2842, x_ohm_per_km = 0.1447, c_nf_per_km = 0, max_i_ka = 9, name='Line 54-55')
        line55_56 = pp.create_line_from_parameters(net_ieee69, buses[54], buses[55], length_km=1, r_ohm_per_km = 0.2813, x_ohm_per_km = 0.1433, c_nf_per_km = 0, max_i_ka = 9, name='Line 55-56')
        line56_57 = pp.create_line_from_parameters(net_ieee69, buses[55], buses[56], length_km=1, r_ohm_per_km = 1.5900, x_ohm_per_km = 0.5337, c_nf_per_km = 0, max_i_ka = 9, name='Line 56-57')
        line57_58 = pp.create_line_from_parameters(net_ieee69, buses[56], buses[57], length_km=1, r_ohm_per_km = 0.7837, x_ohm_per_km = 0.2630, c_nf_per_km = 0, max_i_ka = 9, name='Line 57-58')
        line58_59 = pp.create_line_from_parameters(net_ieee69, buses[57], buses[58], length_km=1, r_ohm_per_km = 0.3042, x_ohm_per_km = 0.1006, c_nf_per_km = 0, max_i_ka = 9, name='Line 58-59')
        line59_60 = pp.create_line_from_parameters(net_ieee69, buses[58], buses[59], length_km=1, r_ohm_per_km = 0.3861, x_ohm_per_km = 0.1172, c_nf_per_km = 0, max_i_ka = 9, name='Line 59-60')
        line60_61 = pp.create_line_from_parameters(net_ieee69, buses[59], buses[60], length_km=1, r_ohm_per_km = 0.5075, x_ohm_per_km = 0.2585, c_nf_per_km = 0, max_i_ka = 9, name='Line 60-61')
        line61_62 = pp.create_line_from_parameters(net_ieee69, buses[60], buses[61], length_km=1, r_ohm_per_km = 0.0974, x_ohm_per_km = 0.0496, c_nf_per_km = 0, max_i_ka = 9, name='Line 61-62')
        line62_63 = pp.create_line_from_parameters(net_ieee69, buses[61], buses[62], length_km=1, r_ohm_per_km = 0.1450, x_ohm_per_km = 0.0738, c_nf_per_km = 0, max_i_ka = 9, name='Line 62-63')
        line63_64 = pp.create_line_from_parameters(net_ieee69, buses[62], buses[63], length_km=1, r_ohm_per_km = 0.7105, x_ohm_per_km = 0.3619, c_nf_per_km = 0, max_i_ka = 9, name='Line 63-64')
        line64_65 = pp.create_line_from_parameters(net_ieee69, buses[63], buses[64], length_km=1, r_ohm_per_km = 1.0410, x_ohm_per_km = 0.5302, c_nf_per_km = 0, max_i_ka = 9, name='Line 64-65')
        line11_66 = pp.create_line_from_parameters(net_ieee69, buses[10], buses[65], length_km=1, r_ohm_per_km = 0.2012, x_ohm_per_km = 0.0611, c_nf_per_km = 0, max_i_ka = 9, name='Line 11-66')
        line66_67 = pp.create_line_from_parameters(net_ieee69, buses[65], buses[66], length_km=1, r_ohm_per_km = 0.0047, x_ohm_per_km = 0.0014, c_nf_per_km = 0, max_i_ka = 9, name='Line 66-67')
        line12_68 = pp.create_line_from_parameters(net_ieee69, buses[11], buses[67], length_km=1, r_ohm_per_km = 0.7394, x_ohm_per_km = 0.2444, c_nf_per_km = 0, max_i_ka = 9, name='Line 12-68')
        line68_69 = pp.create_line_from_parameters(net_ieee69, buses[67], buses[68], length_km=1, r_ohm_per_km = 0.0047, x_ohm_per_km = 0.0016, c_nf_per_km = 0, max_i_ka = 9, name='Line 68-69')


            # Switches
            # S2
       
            # Loads
            # Residential
        pp.create_load(net_ieee69, buses[5] , p_mw = 2.6  , q_mvar = 2.2  , scaling = self.LOAD[self.time]*0.001, name='Load R6')
        pp.create_load(net_ieee69, buses[6] , p_mw = 40.4 , q_mvar = 30   , scaling = self.LOAD[self.time]*0.001, name='Load R7')
        pp.create_load(net_ieee69, buses[7] , p_mw = 75   , q_mvar = 54   , scaling = self.LOAD[self.time]*0.001, name='Load R8')
        pp.create_load(net_ieee69, buses[8] , p_mw = 30   , q_mvar = 22   , scaling = self.LOAD[self.time]*0.001, name='Load R9')
        pp.create_load(net_ieee69, buses[9] , p_mw = 28   , q_mvar = 19   , scaling = self.LOAD[self.time]*0.001, name='Load R10')
        pp.create_load(net_ieee69, buses[10], p_mw = 145  , q_mvar = 104  , scaling = self.LOAD[self.time]*0.001, name='Load R11')
        pp.create_load(net_ieee69, buses[11], p_mw = 145  , q_mvar = 104  , scaling = self.LOAD[self.time]*0.001, name='Load R12')
        pp.create_load(net_ieee69, buses[12], p_mw = 8    , q_mvar = 5    , scaling = self.LOAD[self.time]*0.001, name='Load R13')
        pp.create_load(net_ieee69, buses[13], p_mw = 8    , q_mvar = 5.5  , scaling = self.LOAD[self.time]*0.001, name='Load R14')        
        pp.create_load(net_ieee69, buses[15], p_mw = 45.5 , q_mvar = 30   , scaling = self.LOAD[self.time]*0.001, name='Load R16')
        pp.create_load(net_ieee69, buses[16], p_mw = 60   , q_mvar = 35   , scaling = self.LOAD[self.time]*0.001, name='Load R17')
        pp.create_load(net_ieee69, buses[17], p_mw = 60   , q_mvar = 35   , scaling = self.LOAD[self.time]*0.001, name='Load R18')
        pp.create_load(net_ieee69, buses[19], p_mw = 1    , q_mvar = 0.6  , scaling = self.LOAD[self.time]*0.001, name='Load R20')
        pp.create_load(net_ieee69, buses[20], p_mw = 114  , q_mvar = 81   , scaling = self.LOAD[self.time]*0.001, name='Load R21')
        pp.create_load(net_ieee69, buses[21], p_mw = 5    , q_mvar = 3.5  , scaling = self.LOAD[self.time]*0.001, name='Load R22')
        pp.create_load(net_ieee69, buses[23], p_mw = 28   , q_mvar = 20   , scaling = self.LOAD[self.time]*0.001, name='Load R24')
        pp.create_load(net_ieee69, buses[25], p_mw = 14   , q_mvar = 10   , scaling = self.LOAD[self.time]*0.001, name='Load R26')
        pp.create_load(net_ieee69, buses[26], p_mw = 14   , q_mvar = 10   , scaling = self.LOAD[self.time]*0.001, name='Load R27')
        pp.create_load(net_ieee69, buses[27], p_mw = 26   , q_mvar = 18.6 , scaling = self.LOAD[self.time]*0.001, name='Load R28')
        pp.create_load(net_ieee69, buses[28], p_mw = 26   , q_mvar = 18.6 , scaling = self.LOAD[self.time]*0.001, name='Load R29')        
        pp.create_load(net_ieee69, buses[32], p_mw = 14   , q_mvar = 10   , scaling = self.LOAD[self.time]*0.001, name='Load R33')
        pp.create_load(net_ieee69, buses[33], p_mw = 9.5  , q_mvar = 14   , scaling = self.LOAD[self.time]*0.001, name='Load R34')
        pp.create_load(net_ieee69, buses[34], p_mw = 6    , q_mvar = 4    , scaling = self.LOAD[self.time]*0.001, name='Load R35')
        pp.create_load(net_ieee69, buses[35], p_mw = 26   , q_mvar = 18.55, scaling = self.LOAD[self.time]*0.001, name='Load R36')
        pp.create_load(net_ieee69, buses[36], p_mw = 26   , q_mvar = 18.55, scaling = self.LOAD[self.time]*0.001, name='Load R37')
        pp.create_load(net_ieee69, buses[38], p_mw = 24   , q_mvar = 17   , scaling = self.LOAD[self.time]*0.001, name='Load R39')
        pp.create_load(net_ieee69, buses[39], p_mw = 24   , q_mvar = 17   , scaling = self.LOAD[self.time]*0.001, name='Load R40')
        pp.create_load(net_ieee69, buses[40], p_mw = 1.2  , q_mvar = 1    , scaling = self.LOAD[self.time]*0.001, name='Load R41')
        pp.create_load(net_ieee69, buses[42], p_mw = 6    , q_mvar = 4.3  , scaling = self.LOAD[self.time]*0.001, name='Load R43')
        pp.create_load(net_ieee69, buses[44], p_mw = 39.22, q_mvar = 26.3 , scaling = self.LOAD[self.time]*0.001, name='Load R44')
        pp.create_load(net_ieee69, buses[45], p_mw = 39.22, q_mvar = 26.3 , scaling = self.LOAD[self.time]*0.001, name='Load R45')
        pp.create_load(net_ieee69, buses[47], p_mw = 79   , q_mvar = 56.4 , scaling = self.LOAD[self.time]*0.001, name='Load R46')
        pp.create_load(net_ieee69, buses[48], p_mw = 384.7, q_mvar = 274.5, scaling = self.LOAD[self.time]*0.001, name='Load R48')
        pp.create_load(net_ieee69, buses[49], p_mw = 384.7, q_mvar = 274.5, scaling = self.LOAD[self.time]*0.001, name='Load R50')
        pp.create_load(net_ieee69, buses[50], p_mw = 40.5 , q_mvar = 28.3 , scaling = self.LOAD[self.time]*0.001, name='Load R51')
        pp.create_load(net_ieee69, buses[51], p_mw = 3.6  , q_mvar = 2.7  , scaling = self.LOAD[self.time]*0.001, name='Load R52')
        pp.create_load(net_ieee69, buses[52], p_mw = 4.35 , q_mvar = 3.5  , scaling = self.LOAD[self.time]*0.001, name='Load R53')
        pp.create_load(net_ieee69, buses[53], p_mw = 26.4 , q_mvar = 19   , scaling = self.LOAD[self.time]*0.001, name='Load R54')
        pp.create_load(net_ieee69, buses[54], p_mw = 24   , q_mvar = 17.2 , scaling = self.LOAD[self.time]*0.001, name='Load R55')
        pp.create_load(net_ieee69, buses[58], p_mw = 100  , q_mvar = 72   , scaling = self.LOAD[self.time]*0.001, name='Load R59')
        pp.create_load(net_ieee69, buses[60], p_mw = 1244 , q_mvar = 888  , scaling = self.LOAD[self.time]*0.001, name='Load R61')
        pp.create_load(net_ieee69, buses[61], p_mw = 32   , q_mvar = 23   , scaling = self.LOAD[self.time]*0.001, name='Load R62')
        pp.create_load(net_ieee69, buses[63], p_mw = 227  , q_mvar = 162  , scaling = self.LOAD[self.time]*0.001, name='Load R64')
        pp.create_load(net_ieee69, buses[64], p_mw = 59   , q_mvar = 42   , scaling = self.LOAD[self.time]*0.001, name='Load R65')
        pp.create_load(net_ieee69, buses[65], p_mw = 18   , q_mvar = 13   , scaling = self.LOAD[self.time]*0.001, name='Load R66')
        pp.create_load(net_ieee69, buses[66], p_mw = 18   , q_mvar = 13   , scaling = self.LOAD[self.time]*0.001, name='Load R67')
        pp.create_load(net_ieee69, buses[67], p_mw = 28   , q_mvar = 20   , scaling = self.LOAD[self.time]*0.001, name='Load R68')
        pp.create_load(net_ieee69, buses[68], p_mw = 28   , q_mvar = 20   , scaling = self.LOAD[self.time]*0.001, name='Load R69')

            # Optional distributed energy recources


        pp.create_ext_grid(net_ieee69, buses[0],max_p_mw=100, min_p_mw=0)
        #create generators        
        pp.create_gen(net_ieee69, buses[10], p_mw=0, min_p_mw=0, max_p_mw=1, min_q_mvar=-50, max_q_mvar=50, vm_pu=1.0, controllable=True)
        pp.create_gen(net_ieee69, buses[17], p_mw=0, min_p_mw=0, max_p_mw=1, min_q_mvar=-50, max_q_mvar=50, vm_pu=1.0, controllable=True)
        pp.create_gen(net_ieee69, buses[60], p_mw=0, min_p_mw=0, max_p_mw=1, min_q_mvar=-50, max_q_mvar=50, vm_pu=1.0, controllable=True)
        # 발전 비용 곡선
        pp.create_poly_cost(net_ieee69, 0, "gen", cp2_eur_per_mw2 = 15.5,  cp1_eur_per_mw = 65.6)
        pp.create_poly_cost(net_ieee69, 1, "gen", cp2_eur_per_mw2 = 13.32, cp1_eur_per_mw = 50.2)
        pp.create_poly_cost(net_ieee69, 2, "gen", cp2_eur_per_mw2 = 16.88, cp1_eur_per_mw = 40.1)
        pp.create_poly_cost(net_ieee69, 0, "ext_grid", cp1_eur_per_mw = 80)
        #ess
        #dcline
        #pp.create_storage(net_ieee33, buses[17], p_mw = 0, max_e_mwh = 3, soc_percent = 0.5, min_e_mwh = 0)
        #pp.create_storage(net_ieee33, buses[32], p_mw = 0, max_e_mwh = 3, soc_percent = 0.5, min_e_mwh = 0)


        pp.create_sgen(net_ieee69, buses[11], p_mw = 0.4*self.PV[self.time-1] , q_mvar=0,  name='PV 12', type='PV')
        pp.create_sgen(net_ieee69, buses[19], p_mw = 0.4*self.PV[self.time-1] , q_mvar=0,  name='PV 20' , type='PV')
        pp.create_sgen(net_ieee69, buses[24], p_mw = 0.4*self.PV[self.time-1] , q_mvar=0,  name='PV 25', type='PV')
        pp.create_sgen(net_ieee69, buses[39], p_mw = 0.4*self.PV[self.time-1] , q_mvar=0,  name='PV 40', type='PV')

        pp.create_ext_grid(net_ieee69, buses[0], vm_pu=act1, va_degree=0.0)

        pp.create_shunt(net_ieee69, buses[68], act3)
        pp.create_storage(net_ieee69, buses[68], p_mw = act2, max_e_mwh = 3, soc_percent = 0.5, min_e_mwh = 0)
        pp.create_shunt(net_ieee69, buses[14], act4)
        pp.create_storage(net_ieee69, buses[14], p_mw = -act2, max_e_mwh = 3, soc_percent = 0.5, min_e_mwh = 0)

        pp.create_shunt(net_ieee69, buses[26], act6)
        pp.create_storage(net_ieee69, buses[26], p_mw = act5, max_e_mwh = 3, soc_percent = 0.5, min_e_mwh = 0)
        pp.create_shunt(net_ieee69, buses[53], act7)
        pp.create_storage(net_ieee69, buses[53], p_mw = -act5, max_e_mwh = 3, soc_percent = 0.5, min_e_mwh = 0)


        pp.runopp(net_ieee69, delta=1e-16)

#        return PV, WT, LOAD, Pline, Pbus, radial
        self.busv_pu = np.array([net_ieee69.res_bus.vm_pu])
        self.line_percent = np.array([net_ieee69.res_line.p_to_mw]) 
        self.soc = np.array([net_ieee69.storage.soc_percent])

        self.p_mw = np.array([net_ieee69.storage.p_mw])
        # 래디얼이면 rad = 1
        cost = net_ieee69.res_cost

        a=10
        Vviol1=0
        for i in range(1,len(net_ieee69.res_bus)):
            x = -a*math.sqrt((1.0-net_ieee69.res_bus.vm_pu[i])**2)
            Vviol1 += x

        Vviol2=0
        for i in range(63,69):
            x = -a*math.sqrt((1.0-net_ieee69.res_bus.vm_pu[i])**2)
            Vviol2 += x

        Vviol3=0
        for i in range(13,19):
            x = -a*math.sqrt((1.0-net_ieee69.res_bus.vm_pu[i])**2)
            Vviol3 += x

        Vviol4=0
        for i in range(21,27):
            x = -a*math.sqrt((1.0-net_ieee69.res_bus.vm_pu[i])**2)
            Vviol4 += x

        Vviol5=0
        for i in range(48,54):
            x = -a*math.sqrt((1.0-net_ieee69.res_bus.vm_pu[i])**2)
            Vviol5 += x

        
        return Vviol1 , Vviol2 , Vviol3 ,Vviol4 , Vviol5 , cost

    def _step(self,acts1,acts2,acts3,acts4, acts5,acts6,acts7):
        total_reward = 0        
        net_ieee69 = pp.create_empty_network()
        # Busses
        buses = pp.create_buses(net_ieee69, 69, name=['Bus %i' % i for i in range(1, 70)], vn_kv=self.vpu ,type='b', zone='IEEE69')

        # Lines 
        line1_2   = pp.create_line_from_parameters(net_ieee69, buses[0],  buses[1] , length_km=1, r_ohm_per_km = 0.0005, x_ohm_per_km = 0.0012, c_nf_per_km = 0, max_i_ka = 9, name='Line 1-2')
        line2_3   = pp.create_line_from_parameters(net_ieee69, buses[1],  buses[2] , length_km=1, r_ohm_per_km = 0.0005, x_ohm_per_km = 0.0012, c_nf_per_km = 0, max_i_ka = 9, name='Line 2-3')
        line3_4   = pp.create_line_from_parameters(net_ieee69, buses[2],  buses[3] , length_km=1, r_ohm_per_km = 0.0015, x_ohm_per_km = 0.0036, c_nf_per_km = 0, max_i_ka = 9, name='Line 3-4')
        line4_5   = pp.create_line_from_parameters(net_ieee69, buses[3],  buses[4] , length_km=1, r_ohm_per_km = 0.0251, x_ohm_per_km = 0.0294, c_nf_per_km = 0, max_i_ka = 9, name='Line 4-5')
        line5_6   = pp.create_line_from_parameters(net_ieee69, buses[4],  buses[5] , length_km=1, r_ohm_per_km = 0.3660, x_ohm_per_km = 0.1864, c_nf_per_km = 0, max_i_ka = 9, name='Line 5-6')
        line6_7   = pp.create_line_from_parameters(net_ieee69, buses[5],  buses[6] , length_km=1, r_ohm_per_km = 0.3811, x_ohm_per_km = 0.1941, c_nf_per_km = 0, max_i_ka = 9, name='Line 6-7')
        line7_8   = pp.create_line_from_parameters(net_ieee69, buses[6],  buses[7] , length_km=1, r_ohm_per_km = 0.0922, x_ohm_per_km = 0.0470, c_nf_per_km = 0, max_i_ka = 9, name='Line 7-8')
        line8_9   = pp.create_line_from_parameters(net_ieee69, buses[7],  buses[8] , length_km=1, r_ohm_per_km = 0.0493, x_ohm_per_km = 0.0251, c_nf_per_km = 0, max_i_ka = 9, name='Line 8-9')
        line9_10  = pp.create_line_from_parameters(net_ieee69, buses[8],  buses[9] , length_km=1, r_ohm_per_km = 0.8190, x_ohm_per_km = 0.2707, c_nf_per_km = 0, max_i_ka = 9, name='Line 9-10')
        line10_11 = pp.create_line_from_parameters(net_ieee69, buses[9],  buses[10], length_km=1, r_ohm_per_km = 0.1872, x_ohm_per_km = 0.0619, c_nf_per_km = 0, max_i_ka = 9, name='Line 10-11')
        line11_12 = pp.create_line_from_parameters(net_ieee69, buses[10], buses[11], length_km=1, r_ohm_per_km = 0.7114, x_ohm_per_km = 0.2351, c_nf_per_km = 0, max_i_ka = 9, name='Line 11-12')
        line12_13 = pp.create_line_from_parameters(net_ieee69, buses[11], buses[12], length_km=1, r_ohm_per_km = 1.0300, x_ohm_per_km = 0.3400, c_nf_per_km = 0, max_i_ka = 9, name='Line 12-13')
        line13_14 = pp.create_line_from_parameters(net_ieee69, buses[12], buses[13], length_km=1, r_ohm_per_km = 1.0440, x_ohm_per_km = 0.3450, c_nf_per_km = 0, max_i_ka = 9, name='Line 13-14')
        line14_15 = pp.create_line_from_parameters(net_ieee69, buses[13], buses[14], length_km=1, r_ohm_per_km = 1.0580, x_ohm_per_km = 0.3496, c_nf_per_km = 0, max_i_ka = 9, name='Line 14-15')
        line15_16 = pp.create_line_from_parameters(net_ieee69, buses[14], buses[15], length_km=1, r_ohm_per_km = 0.1966, x_ohm_per_km = 0.0650, c_nf_per_km = 0, max_i_ka = 9, name='Line 15-16')
        line16_17 = pp.create_line_from_parameters(net_ieee69, buses[15], buses[16], length_km=1, r_ohm_per_km = 0.3744, x_ohm_per_km = 0.1238, c_nf_per_km = 0, max_i_ka = 9, name='Line 16-17')
        line17_18 = pp.create_line_from_parameters(net_ieee69, buses[16], buses[17], length_km=1, r_ohm_per_km = 0.0047, x_ohm_per_km = 0.0016, c_nf_per_km = 0, max_i_ka = 9, name='Line 17-18')
        line18_19 = pp.create_line_from_parameters(net_ieee69, buses[17], buses[18], length_km=1, r_ohm_per_km = 0.3276, x_ohm_per_km = 0.1083, c_nf_per_km = 0, max_i_ka = 9, name='Line 18-19')
        line19_20 = pp.create_line_from_parameters(net_ieee69, buses[18], buses[19], length_km=1, r_ohm_per_km = 0.2106, x_ohm_per_km = 0.0696, c_nf_per_km = 0, max_i_ka = 9, name='Line 19-20')
        line20_21 = pp.create_line_from_parameters(net_ieee69, buses[19], buses[20], length_km=1, r_ohm_per_km = 0.3416, x_ohm_per_km = 0.1129, c_nf_per_km = 0, max_i_ka = 9, name='Line 20-21')
        line21_22 = pp.create_line_from_parameters(net_ieee69, buses[20], buses[21], length_km=1, r_ohm_per_km = 0.0140, x_ohm_per_km = 0.0046, c_nf_per_km = 0, max_i_ka = 9, name='Line 21-22')
        line22_23 = pp.create_line_from_parameters(net_ieee69, buses[21], buses[22], length_km=1, r_ohm_per_km = 0.1591, x_ohm_per_km = 0.0526, c_nf_per_km = 0, max_i_ka = 9, name='Line 21-22')
        line23_24 = pp.create_line_from_parameters(net_ieee69, buses[22], buses[23], length_km=1, r_ohm_per_km = 0.3463, x_ohm_per_km = 0.1145, c_nf_per_km = 0, max_i_ka = 9, name='Line 23-24')
        line24_25 = pp.create_line_from_parameters(net_ieee69, buses[23], buses[24], length_km=1, r_ohm_per_km = 0.7488, x_ohm_per_km = 0.2475, c_nf_per_km = 0, max_i_ka = 9, name='Line 24-25')
        line25_26 = pp.create_line_from_parameters(net_ieee69, buses[24], buses[25], length_km=1, r_ohm_per_km = 0.3089, x_ohm_per_km = 0.1021, c_nf_per_km = 0, max_i_ka = 9, name='Line 25-26')
        line26_27 = pp.create_line_from_parameters(net_ieee69, buses[25], buses[26], length_km=1, r_ohm_per_km = 0.1732, x_ohm_per_km = 0.0572, c_nf_per_km = 0, max_i_ka = 9, name='Line 26-27')
        line3_28  = pp.create_line_from_parameters(net_ieee69, buses[2] , buses[27], length_km=1, r_ohm_per_km = 0.0044, x_ohm_per_km = 0.0108, c_nf_per_km = 0, max_i_ka = 9, name='Line 3-28')
        line28_29 = pp.create_line_from_parameters(net_ieee69, buses[27], buses[28], length_km=1, r_ohm_per_km = 0.0640, x_ohm_per_km = 0.1565, c_nf_per_km = 0, max_i_ka = 9, name='Line 28-29')
        line29_30 = pp.create_line_from_parameters(net_ieee69, buses[28], buses[29], length_km=1, r_ohm_per_km = 0.3978, x_ohm_per_km = 0.1315, c_nf_per_km = 0, max_i_ka = 9, name='Line 29-30')
        line30_31 = pp.create_line_from_parameters(net_ieee69, buses[29], buses[30], length_km=1, r_ohm_per_km = 0.0702, x_ohm_per_km = 0.0232, c_nf_per_km = 0, max_i_ka = 9, name='Line 30-31')
        line31_32 = pp.create_line_from_parameters(net_ieee69, buses[30], buses[31], length_km=1, r_ohm_per_km = 0.3510, x_ohm_per_km = 0.1160, c_nf_per_km = 0, max_i_ka = 9, name='Line 31-32')
        line32_33 = pp.create_line_from_parameters(net_ieee69, buses[31], buses[32], length_km=1, r_ohm_per_km = 0.8390, x_ohm_per_km = 0.2816, c_nf_per_km = 0, max_i_ka = 9, name='Line 32-33')
        line33_34 = pp.create_line_from_parameters(net_ieee69, buses[32], buses[33], length_km=1, r_ohm_per_km = 1.7080, x_ohm_per_km = 0.5646, c_nf_per_km = 0, max_i_ka = 9, name='Line 33-34')
        line34_35 = pp.create_line_from_parameters(net_ieee69, buses[33], buses[34], length_km=1, r_ohm_per_km = 1.4740, x_ohm_per_km = 0.4873, c_nf_per_km = 0, max_i_ka = 9, name='Line 34-35')
        line3_36  = pp.create_line_from_parameters(net_ieee69, buses[2] , buses[35], length_km=1, r_ohm_per_km = 0.0044, x_ohm_per_km = 0.0108, c_nf_per_km = 0, max_i_ka = 9, name='Line 3-36')
        line36_37 = pp.create_line_from_parameters(net_ieee69, buses[35], buses[36], length_km=1, r_ohm_per_km = 0.0640, x_ohm_per_km = 0.1565, c_nf_per_km = 0, max_i_ka = 9, name='Line 36-37')
        line37_38 = pp.create_line_from_parameters(net_ieee69, buses[36], buses[37], length_km=1, r_ohm_per_km = 0.1053, x_ohm_per_km = 0.1230, c_nf_per_km = 0, max_i_ka = 9, name='Line 37-38')
        line38_39 = pp.create_line_from_parameters(net_ieee69, buses[37], buses[38], length_km=1, r_ohm_per_km = 0.0304, x_ohm_per_km = 0.0355, c_nf_per_km = 0, max_i_ka = 9, name='Line 38-39')
        line39_40 = pp.create_line_from_parameters(net_ieee69, buses[38], buses[39], length_km=1, r_ohm_per_km = 0.0018, x_ohm_per_km = 0.0021, c_nf_per_km = 0, max_i_ka = 9, name='Line 39-40')
        line40_41 = pp.create_line_from_parameters(net_ieee69, buses[39], buses[40], length_km=1, r_ohm_per_km = 0.7283, x_ohm_per_km = 0.8509, c_nf_per_km = 0, max_i_ka = 9, name='Line 40-41')
        line41_42 = pp.create_line_from_parameters(net_ieee69, buses[40], buses[41], length_km=1, r_ohm_per_km = 0.3100, x_ohm_per_km = 0.3623, c_nf_per_km = 0, max_i_ka = 9, name='Line 41-42')
        line42_43 = pp.create_line_from_parameters(net_ieee69, buses[41], buses[42], length_km=1, r_ohm_per_km = 0.0410, x_ohm_per_km = 0.0478, c_nf_per_km = 0, max_i_ka = 9, name='Line 42-43')
        line43_44 = pp.create_line_from_parameters(net_ieee69, buses[42], buses[43], length_km=1, r_ohm_per_km = 0.0092, x_ohm_per_km = 0.0116, c_nf_per_km = 0, max_i_ka = 9, name='Line 43-44')
        line44_45 = pp.create_line_from_parameters(net_ieee69, buses[43], buses[44], length_km=1, r_ohm_per_km = 0.1089, x_ohm_per_km = 0.1373, c_nf_per_km = 0, max_i_ka = 9, name='Line 44-45')
        line45_46 = pp.create_line_from_parameters(net_ieee69, buses[44], buses[45], length_km=1, r_ohm_per_km = 0.0009, x_ohm_per_km = 0.0012, c_nf_per_km = 0, max_i_ka = 9, name='Line 45-46')
        line4_47  = pp.create_line_from_parameters(net_ieee69, buses[3] , buses[46], length_km=1, r_ohm_per_km = 0.0034, x_ohm_per_km = 0.0084, c_nf_per_km = 0, max_i_ka = 9, name='Line 4-47')
        line47_48 = pp.create_line_from_parameters(net_ieee69, buses[46], buses[47], length_km=1, r_ohm_per_km = 0.0851, x_ohm_per_km = 0.2083, c_nf_per_km = 0, max_i_ka = 9, name='Line 47-48')
        line48_49 = pp.create_line_from_parameters(net_ieee69, buses[47], buses[48], length_km=1, r_ohm_per_km = 0.2898, x_ohm_per_km = 0.7091, c_nf_per_km = 0, max_i_ka = 9, name='Line 48-49')
        line49_50 = pp.create_line_from_parameters(net_ieee69, buses[48], buses[49], length_km=1, r_ohm_per_km = 0.0822, x_ohm_per_km = 0.2011, c_nf_per_km = 0, max_i_ka = 9, name='Line 49-50')
        line8_51  = pp.create_line_from_parameters(net_ieee69, buses[7] , buses[50], length_km=1, r_ohm_per_km = 0.0928, x_ohm_per_km = 0.0473, c_nf_per_km = 0, max_i_ka = 9, name='Line 8-51')
        line51_52 = pp.create_line_from_parameters(net_ieee69, buses[50], buses[51], length_km=1, r_ohm_per_km = 0.3319, x_ohm_per_km = 0.1114, c_nf_per_km = 0, max_i_ka = 9, name='Line 51-52')
        line52_53 = pp.create_line_from_parameters(net_ieee69, buses[51], buses[52], length_km=1, r_ohm_per_km = 0.1740, x_ohm_per_km = 0.1114, c_nf_per_km = 0, max_i_ka = 9, name='Line 52-53')
        line53_54 = pp.create_line_from_parameters(net_ieee69, buses[52], buses[53], length_km=1, r_ohm_per_km = 0.2030, x_ohm_per_km = 0.1034, c_nf_per_km = 0, max_i_ka = 9, name='Line 53-54')
        line54_55 = pp.create_line_from_parameters(net_ieee69, buses[53], buses[54], length_km=1, r_ohm_per_km = 0.2842, x_ohm_per_km = 0.1447, c_nf_per_km = 0, max_i_ka = 9, name='Line 54-55')
        line55_56 = pp.create_line_from_parameters(net_ieee69, buses[54], buses[55], length_km=1, r_ohm_per_km = 0.2813, x_ohm_per_km = 0.1433, c_nf_per_km = 0, max_i_ka = 9, name='Line 55-56')
        line56_57 = pp.create_line_from_parameters(net_ieee69, buses[55], buses[56], length_km=1, r_ohm_per_km = 1.5900, x_ohm_per_km = 0.5337, c_nf_per_km = 0, max_i_ka = 9, name='Line 56-57')
        line57_58 = pp.create_line_from_parameters(net_ieee69, buses[56], buses[57], length_km=1, r_ohm_per_km = 0.7837, x_ohm_per_km = 0.2630, c_nf_per_km = 0, max_i_ka = 9, name='Line 57-58')
        line58_59 = pp.create_line_from_parameters(net_ieee69, buses[57], buses[58], length_km=1, r_ohm_per_km = 0.3042, x_ohm_per_km = 0.1006, c_nf_per_km = 0, max_i_ka = 9, name='Line 58-59')
        line59_60 = pp.create_line_from_parameters(net_ieee69, buses[58], buses[59], length_km=1, r_ohm_per_km = 0.3861, x_ohm_per_km = 0.1172, c_nf_per_km = 0, max_i_ka = 9, name='Line 59-60')
        line60_61 = pp.create_line_from_parameters(net_ieee69, buses[59], buses[60], length_km=1, r_ohm_per_km = 0.5075, x_ohm_per_km = 0.2585, c_nf_per_km = 0, max_i_ka = 9, name='Line 60-61')
        line61_62 = pp.create_line_from_parameters(net_ieee69, buses[60], buses[61], length_km=1, r_ohm_per_km = 0.0974, x_ohm_per_km = 0.0496, c_nf_per_km = 0, max_i_ka = 9, name='Line 61-62')
        line62_63 = pp.create_line_from_parameters(net_ieee69, buses[61], buses[62], length_km=1, r_ohm_per_km = 0.1450, x_ohm_per_km = 0.0738, c_nf_per_km = 0, max_i_ka = 9, name='Line 62-63')
        line63_64 = pp.create_line_from_parameters(net_ieee69, buses[62], buses[63], length_km=1, r_ohm_per_km = 0.7105, x_ohm_per_km = 0.3619, c_nf_per_km = 0, max_i_ka = 9, name='Line 63-64')
        line64_65 = pp.create_line_from_parameters(net_ieee69, buses[63], buses[64], length_km=1, r_ohm_per_km = 1.0410, x_ohm_per_km = 0.5302, c_nf_per_km = 0, max_i_ka = 9, name='Line 64-65')
        line11_66 = pp.create_line_from_parameters(net_ieee69, buses[10], buses[65], length_km=1, r_ohm_per_km = 0.2012, x_ohm_per_km = 0.0611, c_nf_per_km = 0, max_i_ka = 9, name='Line 11-66')
        line66_67 = pp.create_line_from_parameters(net_ieee69, buses[65], buses[66], length_km=1, r_ohm_per_km = 0.0047, x_ohm_per_km = 0.0014, c_nf_per_km = 0, max_i_ka = 9, name='Line 66-67')
        line12_68 = pp.create_line_from_parameters(net_ieee69, buses[11], buses[67], length_km=1, r_ohm_per_km = 0.7394, x_ohm_per_km = 0.2444, c_nf_per_km = 0, max_i_ka = 9, name='Line 12-68')
        line68_69 = pp.create_line_from_parameters(net_ieee69, buses[67], buses[68], length_km=1, r_ohm_per_km = 0.0047, x_ohm_per_km = 0.0016, c_nf_per_km = 0, max_i_ka = 9, name='Line 68-69')


            # Switches
            # S2
       
            # Loads
            # Residential
        pp.create_load(net_ieee69, buses[5] , p_mw = 2.6  , q_mvar = 2.2  , scaling = self.LOAD[self.time]*0.001, name='Load R6')
        pp.create_load(net_ieee69, buses[6] , p_mw = 40.4 , q_mvar = 30   , scaling = self.LOAD[self.time]*0.001, name='Load R7')
        pp.create_load(net_ieee69, buses[7] , p_mw = 75   , q_mvar = 54   , scaling = self.LOAD[self.time]*0.001, name='Load R8')
        pp.create_load(net_ieee69, buses[8] , p_mw = 30   , q_mvar = 22   , scaling = self.LOAD[self.time]*0.001, name='Load R9')
        pp.create_load(net_ieee69, buses[9] , p_mw = 28   , q_mvar = 19   , scaling = self.LOAD[self.time]*0.001, name='Load R10')
        pp.create_load(net_ieee69, buses[10], p_mw = 145  , q_mvar = 104  , scaling = self.LOAD[self.time]*0.001, name='Load R11')
        pp.create_load(net_ieee69, buses[11], p_mw = 145  , q_mvar = 104  , scaling = self.LOAD[self.time]*0.001, name='Load R12')
        pp.create_load(net_ieee69, buses[12], p_mw = 8    , q_mvar = 5    , scaling = self.LOAD[self.time]*0.001, name='Load R13')
        pp.create_load(net_ieee69, buses[13], p_mw = 8    , q_mvar = 5.5  , scaling = self.LOAD[self.time]*0.001, name='Load R14')        
        pp.create_load(net_ieee69, buses[15], p_mw = 45.5 , q_mvar = 30   , scaling = self.LOAD[self.time]*0.001, name='Load R16')
        pp.create_load(net_ieee69, buses[16], p_mw = 60   , q_mvar = 35   , scaling = self.LOAD[self.time]*0.001, name='Load R17')
        pp.create_load(net_ieee69, buses[17], p_mw = 60   , q_mvar = 35   , scaling = self.LOAD[self.time]*0.001, name='Load R18')
        pp.create_load(net_ieee69, buses[19], p_mw = 1    , q_mvar = 0.6  , scaling = self.LOAD[self.time]*0.001, name='Load R20')
        pp.create_load(net_ieee69, buses[20], p_mw = 114  , q_mvar = 81   , scaling = self.LOAD[self.time]*0.001, name='Load R21')
        pp.create_load(net_ieee69, buses[21], p_mw = 5    , q_mvar = 3.5  , scaling = self.LOAD[self.time]*0.001, name='Load R22')
        pp.create_load(net_ieee69, buses[23], p_mw = 28   , q_mvar = 20   , scaling = self.LOAD[self.time]*0.001, name='Load R24')
        pp.create_load(net_ieee69, buses[25], p_mw = 14   , q_mvar = 10   , scaling = self.LOAD[self.time]*0.001, name='Load R26')
        pp.create_load(net_ieee69, buses[26], p_mw = 14   , q_mvar = 10   , scaling = self.LOAD[self.time]*0.001, name='Load R27')
        pp.create_load(net_ieee69, buses[27], p_mw = 26   , q_mvar = 18.6 , scaling = self.LOAD[self.time]*0.001, name='Load R28')
        pp.create_load(net_ieee69, buses[28], p_mw = 26   , q_mvar = 18.6 , scaling = self.LOAD[self.time]*0.001, name='Load R29')        
        pp.create_load(net_ieee69, buses[32], p_mw = 14   , q_mvar = 10   , scaling = self.LOAD[self.time]*0.001, name='Load R33')
        pp.create_load(net_ieee69, buses[33], p_mw = 9.5  , q_mvar = 14   , scaling = self.LOAD[self.time]*0.001, name='Load R34')
        pp.create_load(net_ieee69, buses[34], p_mw = 6    , q_mvar = 4    , scaling = self.LOAD[self.time]*0.001, name='Load R35')
        pp.create_load(net_ieee69, buses[35], p_mw = 26   , q_mvar = 18.55, scaling = self.LOAD[self.time]*0.001, name='Load R36')
        pp.create_load(net_ieee69, buses[36], p_mw = 26   , q_mvar = 18.55, scaling = self.LOAD[self.time]*0.001, name='Load R37')
        pp.create_load(net_ieee69, buses[38], p_mw = 24   , q_mvar = 17   , scaling = self.LOAD[self.time]*0.001, name='Load R39')
        pp.create_load(net_ieee69, buses[39], p_mw = 24   , q_mvar = 17   , scaling = self.LOAD[self.time]*0.001, name='Load R40')
        pp.create_load(net_ieee69, buses[40], p_mw = 1.2  , q_mvar = 1    , scaling = self.LOAD[self.time]*0.001, name='Load R41')
        pp.create_load(net_ieee69, buses[42], p_mw = 6    , q_mvar = 4.3  , scaling = self.LOAD[self.time]*0.001, name='Load R43')
        pp.create_load(net_ieee69, buses[44], p_mw = 39.22, q_mvar = 26.3 , scaling = self.LOAD[self.time]*0.001, name='Load R44')
        pp.create_load(net_ieee69, buses[45], p_mw = 39.22, q_mvar = 26.3 , scaling = self.LOAD[self.time]*0.001, name='Load R45')
        pp.create_load(net_ieee69, buses[47], p_mw = 79   , q_mvar = 56.4 , scaling = self.LOAD[self.time]*0.001, name='Load R46')
        pp.create_load(net_ieee69, buses[48], p_mw = 384.7, q_mvar = 274.5, scaling = self.LOAD[self.time]*0.001, name='Load R48')
        pp.create_load(net_ieee69, buses[49], p_mw = 384.7, q_mvar = 274.5, scaling = self.LOAD[self.time]*0.001, name='Load R50')
        pp.create_load(net_ieee69, buses[50], p_mw = 40.5 , q_mvar = 28.3 , scaling = self.LOAD[self.time]*0.001, name='Load R51')
        pp.create_load(net_ieee69, buses[51], p_mw = 3.6  , q_mvar = 2.7  , scaling = self.LOAD[self.time]*0.001, name='Load R52')
        pp.create_load(net_ieee69, buses[52], p_mw = 4.35 , q_mvar = 3.5  , scaling = self.LOAD[self.time]*0.001, name='Load R53')
        pp.create_load(net_ieee69, buses[53], p_mw = 26.4 , q_mvar = 19   , scaling = self.LOAD[self.time]*0.001, name='Load R54')
        pp.create_load(net_ieee69, buses[54], p_mw = 24   , q_mvar = 17.2 , scaling = self.LOAD[self.time]*0.001, name='Load R55')
        pp.create_load(net_ieee69, buses[58], p_mw = 100  , q_mvar = 72   , scaling = self.LOAD[self.time]*0.001, name='Load R59')
        pp.create_load(net_ieee69, buses[60], p_mw = 1244 , q_mvar = 888  , scaling = self.LOAD[self.time]*0.001, name='Load R61')
        pp.create_load(net_ieee69, buses[61], p_mw = 32   , q_mvar = 23   , scaling = self.LOAD[self.time]*0.001, name='Load R62')
        pp.create_load(net_ieee69, buses[63], p_mw = 227  , q_mvar = 162  , scaling = self.LOAD[self.time]*0.001, name='Load R64')
        pp.create_load(net_ieee69, buses[64], p_mw = 59   , q_mvar = 42   , scaling = self.LOAD[self.time]*0.001, name='Load R65')
        pp.create_load(net_ieee69, buses[65], p_mw = 18   , q_mvar = 13   , scaling = self.LOAD[self.time]*0.001, name='Load R66')
        pp.create_load(net_ieee69, buses[66], p_mw = 18   , q_mvar = 13   , scaling = self.LOAD[self.time]*0.001, name='Load R67')
        pp.create_load(net_ieee69, buses[67], p_mw = 28   , q_mvar = 20   , scaling = self.LOAD[self.time]*0.001, name='Load R68')
        pp.create_load(net_ieee69, buses[68], p_mw = 28   , q_mvar = 20   , scaling = self.LOAD[self.time]*0.001, name='Load R69')

            # Optional distributed energy recources


        pp.create_ext_grid(net_ieee69, buses[0],max_p_mw=100, min_p_mw=0)
        #create generators        
        pp.create_gen(net_ieee69, buses[10], p_mw=0, min_p_mw=0, max_p_mw=1, min_q_mvar=-50, max_q_mvar=50, vm_pu=1.0, controllable=True)
        pp.create_gen(net_ieee69, buses[17], p_mw=0, min_p_mw=0, max_p_mw=1, min_q_mvar=-50, max_q_mvar=50, vm_pu=1.0, controllable=True)
        pp.create_gen(net_ieee69, buses[60], p_mw=0, min_p_mw=0, max_p_mw=1, min_q_mvar=-50, max_q_mvar=50, vm_pu=1.0, controllable=True)
        # 발전 비용 곡선
        pp.create_poly_cost(net_ieee69, 0, "gen", cp2_eur_per_mw2 = 15.5,  cp1_eur_per_mw = 65.6)
        pp.create_poly_cost(net_ieee69, 1, "gen", cp2_eur_per_mw2 = 13.32, cp1_eur_per_mw = 50.2)
        pp.create_poly_cost(net_ieee69, 2, "gen", cp2_eur_per_mw2 = 16.88, cp1_eur_per_mw = 40.1)
        pp.create_poly_cost(net_ieee69, 0, "ext_grid", cp1_eur_per_mw = 80)
        #ess
        #dcline
        #pp.create_storage(net_ieee33, buses[17], p_mw = 0, max_e_mwh = 3, soc_percent = 0.5, min_e_mwh = 0)
        #pp.create_storage(net_ieee33, buses[32], p_mw = 0, max_e_mwh = 3, soc_percent = 0.5, min_e_mwh = 0)


        pp.create_sgen(net_ieee69, buses[11], p_mw = 0.4*self.PV[self.time-1] , q_mvar=0,  name='PV 12', type='PV')
        pp.create_sgen(net_ieee69, buses[19], p_mw = 0.4*self.PV[self.time-1] , q_mvar=0,  name='PV 20' , type='PV')
        pp.create_sgen(net_ieee69, buses[24], p_mw = 0.4*self.PV[self.time-1] , q_mvar=0,  name='PV 25', type='PV')
        pp.create_sgen(net_ieee69, buses[39], p_mw = 0.4*self.PV[self.time-1] , q_mvar=0,  name='PV 40', type='PV')

        pp.create_ext_grid(net_ieee69, buses[0], vm_pu=acts1, va_degree=0.0)

        pp.create_shunt(net_ieee69, buses[68], acts3)
        pp.create_storage(net_ieee69, buses[68], p_mw = acts2, max_e_mwh = 3, soc_percent = 0.5, min_e_mwh = 0)
        pp.create_shunt(net_ieee69, buses[14], acts4)
        pp.create_storage(net_ieee69, buses[14], p_mw = -acts2, max_e_mwh = 3, soc_percent = 0.5, min_e_mwh = 0)

        pp.create_shunt(net_ieee69, buses[26], acts6)
        pp.create_storage(net_ieee69, buses[26], p_mw = acts5, max_e_mwh = 3, soc_percent = 0.5, min_e_mwh = 0)
        pp.create_shunt(net_ieee69, buses[53], acts7)
        pp.create_storage(net_ieee69, buses[53], p_mw = -acts5, max_e_mwh = 3, soc_percent = 0.5, min_e_mwh = 0)


        pp.runopp(net_ieee69, delta=1e-16)

#        return PV, WT, LOAD, Pline, Pbus, radial
        self.busv_pu = np.array([net_ieee69.res_bus.vm_pu])
        self.line_percent = np.array([net_ieee69.res_line.p_to_mw]) 
        self.soc = np.array([net_ieee69.storage.soc_percent])
        self.p_mw = np.array([net_ieee69.storage.p_mw])
        
        cost = net_ieee69.res_cost

        a=10
        Vviol1=0
        for i in range(1,len(net_ieee69.res_bus)):
            x = -a*math.sqrt((1.0-net_ieee69.res_bus.vm_pu[i])**2)
            Vviol1 += x

        Vviol2=0
        for i in range(63,69):
            x = -a*math.sqrt((1.0-net_ieee69.res_bus.vm_pu[i])**2)
            Vviol2 += x

        Vviol3=0
        for i in range(13,19):
            x = -a*math.sqrt((1.0-net_ieee69.res_bus.vm_pu[i])**2)
            Vviol3 += x

        Vviol4=0
        for i in range(21,27):
            x = -a*math.sqrt((1.0-net_ieee69.res_bus.vm_pu[i])**2)
            Vviol4 += x

        Vviol5=0
        for i in range(48,54):
            x = -a*math.sqrt((1.0-net_ieee69.res_bus.vm_pu[i])**2)
            Vviol5 += x

        b=15
        Pbus1 = 0
        for i in range(1,len(net_ieee69.res_bus)):
            Vbus = net_ieee69.res_bus.vm_pu[i]
            if Vbus < 0.95:
                Pbus1 += -100*b*(0.95-Vbus)/len(net_ieee69.res_bus)
            elif 0.95 <= Vbus < 1.05:
                Pbus1 += 0
            else :
                Pbus1 += -100*b*(Vbus-1.05)/len(net_ieee69.res_bus)

        Pbus2 = 0
        for i in range(63,69):
            Vbus = net_ieee69.res_bus.vm_pu[i]
            if Vbus < 0.95:
                Pbus2 += -100*b*(0.95-Vbus)/len(net_ieee69.res_bus)
            elif 0.95 <= Vbus < 1.05:
                Pbus2 += 0
            else :
                Pbus2 += -100*b*(Vbus-1.05)/len(net_ieee69.res_bus)

        Pbus3 = 0
        for i in range(13,19):
            Vbus = net_ieee69.res_bus.vm_pu[i]
            if Vbus < 0.95:
                Pbus3 += -100*b*(0.95-Vbus)/len(net_ieee69.res_bus)
            elif 0.95 <= Vbus < 1.05:
                Pbus3 += 0
            else :
                Pbus3 += -100*b*(Vbus-1.05)/len(net_ieee69.res_bus)

        Pbus4 = 0
        for i in range(21,27):
            Vbus = net_ieee69.res_bus.vm_pu[i]
            if Vbus < 0.95:
                Pbus4 += -100*b*(0.95-Vbus)/len(net_ieee69.res_bus)
            elif 0.95 <= Vbus < 1.05:
                Pbus4 += 0
            else :
                Pbus4 += -100*b*(Vbus-1.05)/len(net_ieee69.res_bus)

        Pbus5 = 0
        for i in range(48,54):
            Vbus = net_ieee69.res_bus.vm_pu[i]
            if Vbus < 0.95:
                Pbus5 += -100*b*(0.95-Vbus)/len(net_ieee69.res_bus)
            elif 0.95 <= Vbus < 1.05:
                Pbus5 += 0
            else :
                Pbus5 += -100*b*(Vbus-1.05)/len(net_ieee69.res_bus)
        sta1 = np.array([net_ieee69.res_bus.vm_pu[1] , net_ieee69.res_bus.vm_pu[2] , net_ieee69.res_bus.vm_pu[3] , net_ieee69.res_bus.vm_pu[4] , net_ieee69.res_bus.vm_pu[5] , net_ieee69.res_bus.vm_pu[6]])
        sta2 = np.array([net_ieee69.res_bus.vm_pu[66], net_ieee69.res_bus.vm_pu[67], net_ieee69.res_bus.vm_pu[68], net_ieee69.res_bus.vm_pu[13], net_ieee69.res_bus.vm_pu[14], net_ieee69.res_bus.vm_pu[15]])
        sta3 = np.array([net_ieee69.res_bus.vm_pu[63], net_ieee69.res_bus.vm_pu[64], net_ieee69.res_bus.vm_pu[65], net_ieee69.res_bus.vm_pu[66], net_ieee69.res_bus.vm_pu[67], net_ieee69.res_bus.vm_pu[68]])
        sta4 = np.array([net_ieee69.res_bus.vm_pu[12], net_ieee69.res_bus.vm_pu[13], net_ieee69.res_bus.vm_pu[14], net_ieee69.res_bus.vm_pu[15], net_ieee69.res_bus.vm_pu[16], net_ieee69.res_bus.vm_pu[17]])
        sta5 = np.array([net_ieee69.res_bus.vm_pu[24], net_ieee69.res_bus.vm_pu[25], net_ieee69.res_bus.vm_pu[26], net_ieee69.res_bus.vm_pu[51], net_ieee69.res_bus.vm_pu[52], net_ieee69.res_bus.vm_pu[53]])
        sta6 = np.array([net_ieee69.res_bus.vm_pu[21], net_ieee69.res_bus.vm_pu[22], net_ieee69.res_bus.vm_pu[23], net_ieee69.res_bus.vm_pu[24], net_ieee69.res_bus.vm_pu[25], net_ieee69.res_bus.vm_pu[26]])
        sta7 = np.array([net_ieee69.res_bus.vm_pu[48], net_ieee69.res_bus.vm_pu[49], net_ieee69.res_bus.vm_pu[50], net_ieee69.res_bus.vm_pu[51], net_ieee69.res_bus.vm_pu[52], net_ieee69.res_bus.vm_pu[53]])
        
       
        
        state1 = np.hstack([self.time, self.PV[(self.time-1)%self.MaxTime], self.PV[(self.time)%self.MaxTime], self.PV[(self.time+1)%self.MaxTime], self.LOAD[(self.time-1)%self.MaxTime], self.LOAD[(self.time)%self.MaxTime], self.LOAD[(self.time+1)%self.MaxTime], sta1 ]).flatten()
        state2 = np.hstack([self.time, self.PV[(self.time-1)%self.MaxTime], self.PV[(self.time)%self.MaxTime], self.PV[(self.time+1)%self.MaxTime], self.LOAD[(self.time-1)%self.MaxTime], self.LOAD[(self.time)%self.MaxTime], self.LOAD[(self.time+1)%self.MaxTime], sta2 ]).flatten()
        state3 = np.hstack([self.time, self.PV[(self.time-1)%self.MaxTime], self.PV[(self.time)%self.MaxTime], self.PV[(self.time+1)%self.MaxTime], self.LOAD[(self.time-1)%self.MaxTime], self.LOAD[(self.time)%self.MaxTime], self.LOAD[(self.time+1)%self.MaxTime], sta3 ]).flatten()
        state4 = np.hstack([self.time, self.PV[(self.time-1)%self.MaxTime], self.PV[(self.time)%self.MaxTime], self.PV[(self.time+1)%self.MaxTime], self.LOAD[(self.time-1)%self.MaxTime], self.LOAD[(self.time)%self.MaxTime], self.LOAD[(self.time+1)%self.MaxTime], sta4 ]).flatten()
        state5 = np.hstack([self.time, self.PV[(self.time-1)%self.MaxTime], self.PV[(self.time)%self.MaxTime], self.PV[(self.time+1)%self.MaxTime], self.LOAD[(self.time-1)%self.MaxTime], self.LOAD[(self.time)%self.MaxTime], self.LOAD[(self.time+1)%self.MaxTime], sta5 ]).flatten()
        state6 = np.hstack([self.time, self.PV[(self.time-1)%self.MaxTime], self.PV[(self.time)%self.MaxTime], self.PV[(self.time+1)%self.MaxTime], self.LOAD[(self.time-1)%self.MaxTime], self.LOAD[(self.time)%self.MaxTime], self.LOAD[(self.time+1)%self.MaxTime], sta6 ]).flatten()
        state7 = np.hstack([self.time, self.PV[(self.time-1)%self.MaxTime], self.PV[(self.time)%self.MaxTime], self.PV[(self.time+1)%self.MaxTime], self.LOAD[(self.time-1)%self.MaxTime], self.LOAD[(self.time)%self.MaxTime], self.LOAD[(self.time+1)%self.MaxTime], sta7 ]).flatten()
        #print('time=',self.time)
        #print('PV=',self.PV[self.time-1][0])
        #print('PVNext=',self.PV[self.time][0])
        
        next_states = [state1, state2, state3, state4, state5, state6, state7]


        noV1, _, _, _, _, _  = self.noact(act1 = 1.013, act2 = acts2, act3 = acts3 , act4 = acts4, act5 = acts5, act6 = acts6 , act7 = acts7)
        _, noV2, noV3, noV4, noV5, _ = self.noact(act1 = acts1, act2 = acts2, act3 = 0 , act4 = 0, act5 = acts5, act6 = 0 , act7 = 0)
        _, _,  _, _, _, nocostsop1 = self.noact(act1 = acts1, act2 = 0, act3 = acts3 , act4 = acts4, act5 = acts5, act6 = acts6 , act7 = acts7)
        _, _,  _, _, _, nocostsop2 = self.noact(act1 = acts1, act2 = acts2, act3 = acts3 , act4 = acts4, act5 = 0, act6 = acts6 , act7 = acts7)

        info = [acts1 ,acts2,acts3 ,acts4 ,acts5,acts6 ,acts7,cost,nocostsop1,nocostsop2]
        infos =  net_ieee69.res_sgen.p_mw
        infov =  net_ieee69.res_bus.vm_pu

        self.rwd = np.zeros(shape=(7,))        
        self.rwd[0] = Vviol1  - noV1                    # tap
        self.rwd[1] = - (cost - nocostsop1)  *10000000  # dcline p (ess 69, 15)
        self.rwd[2] = (Vviol2 + Pbus2 - noV2)*10000     # dcline q (svc 69)
        self.rwd[3] = (Vviol3 + Pbus3 - noV3)*10000     # dcline q (svc 15)
        self.rwd[4] = - (cost - nocostsop2)  *10000000  # dcline p (ess 27, 54)
        self.rwd[5] = (Vviol4 + Pbus4 - noV4)*10000     # dcline q (svc 27)
        self.rwd[6] = (Vviol5 + Pbus5 - noV5)*10000     # dcline q (svc 54)
        '''
        print('Vviol1 =',Vviol1)
        print('Pbus1 = ', Pbus1 )
        print('cost =',cost)
        print('nocost =',nocost)
        print((cost-nocost)*100)'''
        total_rewards=self.rwd
        return next_states , total_rewards , info, infos, infov
