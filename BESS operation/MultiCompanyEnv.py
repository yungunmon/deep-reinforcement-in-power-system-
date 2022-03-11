import pandapower.networks as nw
import pandapower as pp  
import pandapower.topology as top
from pandapower.plotting import simple_plot, simple_plotly , pf_res_plotly
import pandapower.plotting as plot
import seaborn
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
        self.state_size = 7
        self.action_size = 20
        self.batch_size = self.MaxTime
        self.interval = 60       
        self.action_domain = [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        self.PV = [0, 0, 0, 0, 0, 0, 0.04675, 0.28, 0.52625, 0.7765, 0.9, 0.975, 1, 0.75, 0.625, 0.375, 0.1665, 0.095, 0, 0, 0, 0, 0, 0]
        self.LOAD = [0.793681328 , 0.749258257, 0.726207166, 0.727527632, 0.74872029,  0.778324802, 0.815623879, 0.859231848, 0.937367546, 0.982393792, 0.995875583, 0.99258257,  0.92934694,  0.970803039, 0.989501483, 0.991816374, 1,   0.997522089,
         0.994343191, 0.986029148, 0.953262039, 0.92862965,  0.946692315, 0.952903394]
        #self.SMP = [111.36, 191.44, 110.46, 110.91, 112.04, 107.74, 111.02, 110.57, 89.1, 106.84, 107.29, 109.33, 108.99, 107.41, 106.73, 107.74, 109.55, 207.75, 194.62, 90.14, 110.91, 106.16, 65.78, 112.04 ]
        self.SMP = [11.36, 19.44, 11.46, 11.91, 11.04, 10.74, 11.02, 11.57, 89.1, 106.84, 107.29, 109.33, 108.99, 107.41, 106.73, 107.74, 109.55, 207.75, 194.62, 90.14, 110.91, 106.16, 65.78, 11.04 ]
        self.soc=0.5

    def reset(self):
        np.random.seed(seed=0)
        self.PV = [0, 0, 0, 0, 0, 0, 0.04675, 0.28, 0.52625, 0.7765, 0.9, 0.975, 1, 0.75, 0.625, 0.375, 0.1665, 0.095, 0, 0, 0, 0, 0, 0]
        self.LOAD = [0.793681328 , 0.749258257, 0.726207166, 0.727527632, 0.74872029,  0.778324802, 0.815623879, 0.859231848, 0.937367546, 0.982393792, 0.995875583, 0.99258257,  0.92934694,  0.970803039, 0.989501483, 0.991816374, 1,   0.997522089,
         0.994343191, 0.986029148, 0.953262039, 0.92862965,  0.946692315, 0.952903394]
        #self.SMP = [111.36, 191.44, 110.46, 110.91, 112.04, 107.74, 111.02, 110.57, 89.1, 106.84, 107.29, 109.33, 108.99, 107.41, 106.73, 107.74, 109.55, 207.75, 194.62, 90.14, 110.91, 106.16, 65.78, 112.04 ]
        self.SMP = [11.36, 19.44, 11.46, 11.91, 11.04, 10.74, 11.02, 11.57, 89.1, 106.84, 107.29, 109.33, 108.99, 107.41, 106.73, 107.74, 109.55, 207.75, 194.62, 90.14, 110.91, 106.16, 65.78, 11.04 ]

        self.rwd = np.zeros(shape=(1,))
        self.time = 0
        net_ieee33 = pp.create_empty_network()
        

        # Busses
        buses = pp.create_buses(net_ieee33, 33, name=['Bus %i' % i for i in range(1, 34)], vn_kv=12.66 ,type='b', zone='IEEE33')

        # Lines
        line1_2 = pp.create_line_from_parameters(net_ieee33, buses[0], buses[1], length_km=1, r_ohm_per_km = 0.0922, x_ohm_per_km = 0.0470, c_nf_per_km = 0, max_i_ka = 9, name='Line 1-2')
        line2_3 = pp.create_line_from_parameters(net_ieee33, buses[1], buses[2], length_km=1, r_ohm_per_km = 0.4930, x_ohm_per_km = 0.2511, c_nf_per_km = 0, max_i_ka = 9, name='Line 2-3')
        line3_4 = pp.create_line_from_parameters(net_ieee33, buses[2], buses[3], length_km=1, r_ohm_per_km = 0.3660, x_ohm_per_km = 0.1864, c_nf_per_km = 0, max_i_ka = 9, name='Line 3-4')
        line4_5 = pp.create_line_from_parameters(net_ieee33, buses[3], buses[4], length_km=1, r_ohm_per_km = 0.3811, x_ohm_per_km = 0.1941, c_nf_per_km = 0, max_i_ka = 9, name='Line 4-5')
        line5_6 = pp.create_line_from_parameters(net_ieee33, buses[4], buses[5], length_km=1, r_ohm_per_km = 0.8190, x_ohm_per_km = 0.7070, c_nf_per_km = 0, max_i_ka = 9, name='Line 5-6')
        line6_7 = pp.create_line_from_parameters(net_ieee33, buses[5], buses[6], length_km=1, r_ohm_per_km = 0.1872, x_ohm_per_km = 0.6188, c_nf_per_km = 0, max_i_ka = 9, name='Line 6-7')
        line7_8 = pp.create_line_from_parameters(net_ieee33, buses[6], buses[7], length_km=1, r_ohm_per_km = 0.7114, x_ohm_per_km = 0.2351, c_nf_per_km = 0, max_i_ka = 9, name='Line 7-8')
        line8_9 = pp.create_line_from_parameters(net_ieee33, buses[7], buses[8], length_km=1, r_ohm_per_km = 1.0300, x_ohm_per_km = 0.7400, c_nf_per_km = 0, max_i_ka = 9, name='Line 8-9')
        line9_10 = pp.create_line_from_parameters(net_ieee33, buses[8], buses[9], length_km=1, r_ohm_per_km = 1.0440, x_ohm_per_km = 0.7400, c_nf_per_km = 0, max_i_ka = 9, name='Line 9-10')
        line10_11 = pp.create_line_from_parameters(net_ieee33, buses[9], buses[10], length_km=1, r_ohm_per_km = 0.1966, x_ohm_per_km = 0.0650, c_nf_per_km = 0, max_i_ka = 9, name='Line 10-11')
        line11_12 = pp.create_line_from_parameters(net_ieee33, buses[10], buses[11], length_km=1, r_ohm_per_km = 0.3744, x_ohm_per_km = 0.1238, c_nf_per_km = 0, max_i_ka = 9, name='Line 11-12')
        line12_13 = pp.create_line_from_parameters(net_ieee33, buses[11], buses[12], length_km=1, r_ohm_per_km = 1.4680, x_ohm_per_km = 1.1550, c_nf_per_km = 0, max_i_ka = 9, name='Line 12-13')
        line13_14 = pp.create_line_from_parameters(net_ieee33, buses[12], buses[13], length_km=1, r_ohm_per_km = 0.5416, x_ohm_per_km = 0.7129, c_nf_per_km = 0, max_i_ka = 9, name='Line 13-14')
        line14_15 = pp.create_line_from_parameters(net_ieee33, buses[13], buses[14], length_km=1, r_ohm_per_km = 0.5910, x_ohm_per_km = 0.5260, c_nf_per_km = 0, max_i_ka = 9, name='Line 14-15')
        line15_16 = pp.create_line_from_parameters(net_ieee33, buses[14], buses[15], length_km=1, r_ohm_per_km = 0.7463, x_ohm_per_km = 0.5450, c_nf_per_km = 0, max_i_ka = 9, name='Line 15-16')
        line16_17 = pp.create_line_from_parameters(net_ieee33, buses[15], buses[16], length_km=1, r_ohm_per_km = 1.2890, x_ohm_per_km = 1.7210, c_nf_per_km = 0, max_i_ka = 9, name='Line 16-17')
        line17_18 = pp.create_line_from_parameters(net_ieee33, buses[16], buses[17], length_km=1, r_ohm_per_km = 0.7320, x_ohm_per_km = 0.5740, c_nf_per_km = 0, max_i_ka = 9, name='Line 17-18')
        line1_19 = pp.create_line_from_parameters(net_ieee33, buses[1], buses[18], length_km=1, r_ohm_per_km = 0.1640, x_ohm_per_km = 0.1565, c_nf_per_km = 0, max_i_ka = 9, name='Line 2-19')
        line19_20 = pp.create_line_from_parameters(net_ieee33, buses[18], buses[19], length_km=1, r_ohm_per_km = 1.5042, x_ohm_per_km = 1.3554, c_nf_per_km = 0, max_i_ka = 9, name='Line 19-20')
        line20_21 = pp.create_line_from_parameters(net_ieee33, buses[19], buses[20], length_km=1, r_ohm_per_km = 0.4095, x_ohm_per_km = 0.4784, c_nf_per_km = 0, max_i_ka = 9, name='Line 20-21')
        line21_22 = pp.create_line_from_parameters(net_ieee33, buses[20], buses[21], length_km=1, r_ohm_per_km = 0.7089, x_ohm_per_km = 0.9373, c_nf_per_km = 0, max_i_ka = 9, name='Line 21-22')
        line2_23 = pp.create_line_from_parameters(net_ieee33, buses[2], buses[22], length_km=1, r_ohm_per_km = 0.4512, x_ohm_per_km = 0.3083, c_nf_per_km = 0, max_i_ka = 9, name='Line 3-23')
        line23_24 = pp.create_line_from_parameters(net_ieee33, buses[22], buses[23], length_km=1, r_ohm_per_km = 0.8980, x_ohm_per_km = 0.7091, c_nf_per_km = 0, max_i_ka = 9, name='Line 23-24')
        line24_25 = pp.create_line_from_parameters(net_ieee33, buses[23], buses[24], length_km=1, r_ohm_per_km = 0.8960, x_ohm_per_km = 0.7011, c_nf_per_km = 0, max_i_ka = 9, name='Line 24-25')
        line6_26 = pp.create_line_from_parameters(net_ieee33, buses[5], buses[25], length_km=1, r_ohm_per_km = 0.2030, x_ohm_per_km = 0.1034, c_nf_per_km = 0, max_i_ka = 9, name='Line 6-26')
        line26_27 = pp.create_line_from_parameters(net_ieee33, buses[25], buses[26], length_km=1, r_ohm_per_km = 0.2842, x_ohm_per_km = 0.1447, c_nf_per_km = 0, max_i_ka = 9, name='Line 26-27')
        line27_28 = pp.create_line_from_parameters(net_ieee33, buses[26], buses[27], length_km=1, r_ohm_per_km = 1.0590, x_ohm_per_km = 0.9337, c_nf_per_km = 0, max_i_ka = 9, name='Line 27-28')
        line28_29 = pp.create_line_from_parameters(net_ieee33, buses[27], buses[28], length_km=1, r_ohm_per_km = 0.8042, x_ohm_per_km = 0.7006, c_nf_per_km = 0, max_i_ka = 9, name='Line 28-29')
        line29_30 = pp.create_line_from_parameters(net_ieee33, buses[28], buses[29], length_km=1, r_ohm_per_km = 0.5075, x_ohm_per_km = 0.2585, c_nf_per_km = 0, max_i_ka = 9, name='Line 29-30')
        line30_31 = pp.create_line_from_parameters(net_ieee33, buses[29], buses[30], length_km=1, r_ohm_per_km = 0.9744, x_ohm_per_km = 0.9630, c_nf_per_km = 0, max_i_ka = 9, name='Line 30-31')
        line31_32 = pp.create_line_from_parameters(net_ieee33, buses[30], buses[31], length_km=1, r_ohm_per_km = 0.3105, x_ohm_per_km = 0.3619, c_nf_per_km = 0, max_i_ka = 9, name='Line 31-32')
        line32_33 = pp.create_line_from_parameters(net_ieee33, buses[31], buses[32], length_km=1, r_ohm_per_km = 0.3410, x_ohm_per_km = 0.5302, c_nf_per_km = 0, max_i_ka = 9, name='Line 32-33')
        
            # Ext-Grid
        pp.create_ext_grid(net_ieee33, buses[0], vm_pu=1.0, va_degree=0.0)


            # Switches
            # S2
       
            # Loads
            # Residential
        pp.create_load(net_ieee33, buses[1], p_mw = 0.1 , q_mvar = 0.06 , scaling = self.LOAD[self.time], name='Load R2')
        pp.create_load(net_ieee33, buses[2], p_mw = 0.09 , q_mvar = 0.04 , scaling = self.LOAD[self.time], name='Load R3')
        pp.create_load(net_ieee33, buses[3], p_mw = 0.12 , q_mvar = 0.08 , scaling = self.LOAD[self.time], name='Load R4')
        pp.create_load(net_ieee33, buses[4], p_mw = 0.06 , q_mvar = 0.03 , scaling = self.LOAD[self.time], name='Load R5')
        pp.create_load(net_ieee33, buses[5], p_mw = 0.06 , q_mvar = 0.02 , scaling = self.LOAD[self.time], name='Load R6')
        pp.create_load(net_ieee33, buses[6], p_mw = 0.2 , q_mvar = 0.1 , scaling = self.LOAD[self.time], name='Load R7')
        pp.create_load(net_ieee33, buses[7], p_mw = 0.2 , q_mvar = 0.1 , scaling = self.LOAD[self.time], name='Load R8')
        pp.create_load(net_ieee33, buses[8], p_mw = 0.06 , q_mvar = 0.02 , scaling = self.LOAD[self.time], name='Load R9')
        pp.create_load(net_ieee33, buses[9], p_mw = 0.06 , q_mvar = 0.02 , scaling = self.LOAD[self.time], name='Load R10')
        pp.create_load(net_ieee33, buses[10], p_mw = 0.045 , q_mvar = 0.03 , scaling = self.LOAD[self.time], name='Load R11')
        pp.create_load(net_ieee33, buses[11], p_mw = 0.06 , q_mvar = 0.035 , scaling = self.LOAD[self.time], name='Load R12')
        pp.create_load(net_ieee33, buses[12], p_mw = 0.06 , q_mvar = 0.035 , scaling = self.LOAD[self.time], name='Load R13')
        pp.create_load(net_ieee33, buses[13], p_mw = 0.12 , q_mvar = 0.08 , scaling = self.LOAD[self.time], name='Load R14')
        pp.create_load(net_ieee33, buses[14], p_mw = 0.06 , q_mvar = 0.01 , scaling = self.LOAD[self.time], name='Load R15')
        pp.create_load(net_ieee33, buses[15], p_mw = 0.06 , q_mvar = 0.02 , scaling = self.LOAD[self.time], name='Load R16')
        pp.create_load(net_ieee33, buses[16], p_mw = 0.06 , q_mvar = 0.02 , scaling = self.LOAD[self.time], name='Load R17')
        pp.create_load(net_ieee33, buses[17], p_mw = 0.09 , q_mvar = 0.04 , scaling = self.LOAD[self.time], name='Load R18')
        pp.create_load(net_ieee33, buses[18], p_mw = 0.09 , q_mvar = 0.04 , scaling = self.LOAD[self.time], name='Load R19')
        pp.create_load(net_ieee33, buses[19], p_mw = 0.09 , q_mvar = 0.04 , scaling = self.LOAD[self.time], name='Load R20')
        pp.create_load(net_ieee33, buses[20], p_mw = 0.09 , q_mvar = 0.04 , scaling = self.LOAD[self.time], name='Load R21')
        pp.create_load(net_ieee33, buses[21], p_mw = 0.09 , q_mvar = 0.04 , scaling = self.LOAD[self.time], name='Load R22')
        pp.create_load(net_ieee33, buses[22], p_mw = 0.09 , q_mvar = 0.05 , scaling = self.LOAD[self.time], name='Load R23')
        pp.create_load(net_ieee33, buses[23], p_mw = 0.42 , q_mvar = 0.2 , scaling = self.LOAD[self.time], name='Load R24')
        pp.create_load(net_ieee33, buses[24], p_mw = 0.42 , q_mvar = 0.2 , scaling = self.LOAD[self.time], name='Load R25')
        pp.create_load(net_ieee33, buses[25], p_mw = 0.06 , q_mvar = 0.025 , scaling = self.LOAD[self.time], name='Load R26')
        pp.create_load(net_ieee33, buses[26], p_mw = 0.06 , q_mvar = 0.025 , scaling = self.LOAD[self.time], name='Load R27')
        pp.create_load(net_ieee33, buses[27], p_mw = 0.06 , q_mvar = 0.02 , scaling = self.LOAD[self.time], name='Load R28')
        pp.create_load(net_ieee33, buses[28], p_mw = 0.12 , q_mvar = 0.07 , scaling = self.LOAD[self.time], name='Load R29')
        pp.create_load(net_ieee33, buses[29], p_mw = 0.2 , q_mvar = 0.6 , scaling = self.LOAD[self.time], name='Load R30')
        pp.create_load(net_ieee33, buses[30], p_mw = 0.15 , q_mvar = 0.07 , scaling = self.LOAD[self.time], name='Load R31')
        pp.create_load(net_ieee33, buses[31], p_mw = 0.21 , q_mvar = 0.1 , scaling = self.LOAD[self.time], name='Load R32')
        pp.create_load(net_ieee33, buses[32], p_mw = 0.06 , q_mvar = 0.04 , scaling = self.LOAD[self.time], name='Load R33')

            # Optional distributed energy recources
        '''
        pp.create_sgen(net_ieee33, buses[2], self.PV[self.time]*0.2, q_mvar=0,  name='PV 3', type='PV')
        pp.create_sgen(net_ieee33, buses[7], self.PV[self.time]*0.2, q_mvar=0,  name='PV 8', type='PV')
        pp.create_sgen(net_ieee33, buses[13], self.PV[self.time]*0.2, q_mvar=0,  name='PV 14', type='PV')
        pp.create_sgen(net_ieee33, buses[24], self.PV[self.time]*0.3, q_mvar=0,  name='PV 25', type='PV')
        pp.create_sgen(net_ieee33, buses[29], self.PV[self.time]*0.3, q_mvar=0,  name='PV 29', type='PV')
        pp.create_sgen(net_ieee33, buses[30], self.PV[self.time]*0.5, q_mvar=0,  name='PV 30', type='PV')'''

        pp.create_sgen(net_ieee33, buses[2],  self.PV[self.time]*1.0*(1+((np.random.rand()-0.5)*2)*0.1) , q_mvar=0,  name='PV 3', type='PV')
        pp.create_sgen(net_ieee33, buses[7],  self.PV[self.time]*1.0*(1+((np.random.rand()-0.5)*2)*0.1) , q_mvar=0,  name='PV 8', type='PV')
        pp.create_sgen(net_ieee33, buses[13], self.PV[self.time]*1.0*(1+((np.random.rand()-0.5)*2)*0.1) , q_mvar=0,  name='PV 14', type='PV')
        pp.create_sgen(net_ieee33, buses[24], self.PV[self.time]*0.8*(1+((np.random.rand()-0.5)*2)*0.1) , q_mvar=0,  name='PV 25', type='PV')
        pp.create_sgen(net_ieee33, buses[29], self.PV[self.time]*0.9*(1+((np.random.rand()-0.5)*2)*0.1) , q_mvar=0,  name='PV 29', type='PV')
        pp.create_sgen(net_ieee33, buses[30], self.PV[self.time]*1.0*(1+((np.random.rand()-0.5)*2)*0.1) , q_mvar=0,  name='PV 30', type='PV')

        pp.create_storage(net_ieee33, buses[1], p_mw = 0, max_e_mwh = 2.5, soc_percent = 0.5, min_e_mwh = 0)
       
        pp.runpp(net_ieee33)


#        return PV, WT, LOAD, Pline, Pbus, radial
        self.busv_pu = np.array([net_ieee33.res_bus.vm_pu])
        self.line_percent = np.array([net_ieee33.res_line.p_to_mw]) 
        self.soc = np.array([net_ieee33.storage.soc_percent])
        self.p_mw = np.array([net_ieee33.storage.p_mw])



        pv_ = np.array([self.PV[(self.time)% self.MaxTime]-self.PV[(self.time -1)% self.MaxTime]])
        load_ = np.array([self.LOAD[(self.time)% self.MaxTime]-self.LOAD[(self.time -1)% self.MaxTime]])
        smp_ = np.array([self.SMP[(self.time)% self.MaxTime]-self.SMP[(self.time -1)% self.MaxTime]])
        '''
        pv_ = np.array([self.PV[(self.time -2)% self.MaxTime],self.PV[(self.time -1)% self.MaxTime], self.PV[(self.time)% self.MaxTime], self.PV[(self.time + 1)% self.MaxTime],self.PV[(self.time + 2)% self.MaxTime]])
        load_ = np.array([self.LOAD[(self.time -2)% self.MaxTime],self.LOAD[(self.time -1)% self.MaxTime], self.LOAD[(self.time)% self.MaxTime], self.LOAD[(self.time + 1)% self.MaxTime],self.LOAD[(self.time + 2)% self.MaxTime]])
        smp_ = np.array([self.SMP[(self.time-2)% self.MaxTime],self.SMP[(self.time-1)% self.MaxTime], self.SMP[(self.time)% self.MaxTime], self.SMP[(self.time + 1)% self.MaxTime],self.SMP[(self.time + 2)% self.MaxTime]])
        '''
        state1 = np.hstack([pv_, load_, smp_, self.soc[0]+0.4*self.p_mw[0], self.p_mw[0], self.line_percent[0][0], self.SMP[self.time]]).flatten()
        
        states = [state1]
        return states 

    def step(self, actions, soc):

        self.rwd = np.zeros(shape=(1,))
        terminal = False
        domain =[-0.5, -0.45, -0.4, -0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]       
        n_state, rwd, ifo ,ifos = self._step(actions, soc)

        self.rwd[0] = rwd[0]
             
        next_state = n_state
        total_reward = [self.rwd[0]]
        terminals = [False]
        info = ifo
        infos = ifos

        self.time = self.time + 1
        if self.time == self.MaxTime:
            self.time = 0
            
            terminals = [True]

        return next_state, total_reward, terminals, info ,infos


    def _step(self,action,soc):

        total_reward = 0
        o = action
        
        net_ieee33 = pp.create_empty_network()
        

        # Busses
        buses = pp.create_buses(net_ieee33, 33, name=['Bus %i' % i for i in range(1, 34)], vn_kv=12.66 ,type='b', zone='IEEE33')

        # Lines
        line1_2 = pp.create_line_from_parameters(net_ieee33, buses[0], buses[1], length_km=1, r_ohm_per_km = 0.0922, x_ohm_per_km = 0.0470, c_nf_per_km = 0, max_i_ka = 9, name='Line 1-2')
        line2_3 = pp.create_line_from_parameters(net_ieee33, buses[1], buses[2], length_km=1, r_ohm_per_km = 0.4930, x_ohm_per_km = 0.2511, c_nf_per_km = 0, max_i_ka = 9, name='Line 2-3')
        line3_4 = pp.create_line_from_parameters(net_ieee33, buses[2], buses[3], length_km=1, r_ohm_per_km = 0.3660, x_ohm_per_km = 0.1864, c_nf_per_km = 0, max_i_ka = 9, name='Line 3-4')
        line4_5 = pp.create_line_from_parameters(net_ieee33, buses[3], buses[4], length_km=1, r_ohm_per_km = 0.3811, x_ohm_per_km = 0.1941, c_nf_per_km = 0, max_i_ka = 9, name='Line 4-5')
        line5_6 = pp.create_line_from_parameters(net_ieee33, buses[4], buses[5], length_km=1, r_ohm_per_km = 0.8190, x_ohm_per_km = 0.7070, c_nf_per_km = 0, max_i_ka = 9, name='Line 5-6')
        line6_7 = pp.create_line_from_parameters(net_ieee33, buses[5], buses[6], length_km=1, r_ohm_per_km = 0.1872, x_ohm_per_km = 0.6188, c_nf_per_km = 0, max_i_ka = 9, name='Line 6-7')
        line7_8 = pp.create_line_from_parameters(net_ieee33, buses[6], buses[7], length_km=1, r_ohm_per_km = 0.7114, x_ohm_per_km = 0.2351, c_nf_per_km = 0, max_i_ka = 9, name='Line 7-8')
        line8_9 = pp.create_line_from_parameters(net_ieee33, buses[7], buses[8], length_km=1, r_ohm_per_km = 1.0300, x_ohm_per_km = 0.7400, c_nf_per_km = 0, max_i_ka = 9, name='Line 8-9')
        line9_10 = pp.create_line_from_parameters(net_ieee33, buses[8], buses[9], length_km=1, r_ohm_per_km = 1.0440, x_ohm_per_km = 0.7400, c_nf_per_km = 0, max_i_ka = 9, name='Line 9-10')
        line10_11 = pp.create_line_from_parameters(net_ieee33, buses[9], buses[10], length_km=1, r_ohm_per_km = 0.1966, x_ohm_per_km = 0.0650, c_nf_per_km = 0, max_i_ka = 9, name='Line 10-11')
        line11_12 = pp.create_line_from_parameters(net_ieee33, buses[10], buses[11], length_km=1, r_ohm_per_km = 0.3744, x_ohm_per_km = 0.1238, c_nf_per_km = 0, max_i_ka = 9, name='Line 11-12')
        line12_13 = pp.create_line_from_parameters(net_ieee33, buses[11], buses[12], length_km=1, r_ohm_per_km = 1.4680, x_ohm_per_km = 1.1550, c_nf_per_km = 0, max_i_ka = 9, name='Line 12-13')
        line13_14 = pp.create_line_from_parameters(net_ieee33, buses[12], buses[13], length_km=1, r_ohm_per_km = 0.5416, x_ohm_per_km = 0.7129, c_nf_per_km = 0, max_i_ka = 9, name='Line 13-14')
        line14_15 = pp.create_line_from_parameters(net_ieee33, buses[13], buses[14], length_km=1, r_ohm_per_km = 0.5910, x_ohm_per_km = 0.5260, c_nf_per_km = 0, max_i_ka = 9, name='Line 14-15')
        line15_16 = pp.create_line_from_parameters(net_ieee33, buses[14], buses[15], length_km=1, r_ohm_per_km = 0.7463, x_ohm_per_km = 0.5450, c_nf_per_km = 0, max_i_ka = 9, name='Line 15-16')
        line16_17 = pp.create_line_from_parameters(net_ieee33, buses[15], buses[16], length_km=1, r_ohm_per_km = 1.2890, x_ohm_per_km = 1.7210, c_nf_per_km = 0, max_i_ka = 9, name='Line 16-17')
        line17_18 = pp.create_line_from_parameters(net_ieee33, buses[16], buses[17], length_km=1, r_ohm_per_km = 0.7320, x_ohm_per_km = 0.5740, c_nf_per_km = 0, max_i_ka = 9, name='Line 17-18')
        line1_19 = pp.create_line_from_parameters(net_ieee33, buses[1], buses[18], length_km=1, r_ohm_per_km = 0.1640, x_ohm_per_km = 0.1565, c_nf_per_km = 0, max_i_ka = 9, name='Line 2-19')
        line19_20 = pp.create_line_from_parameters(net_ieee33, buses[18], buses[19], length_km=1, r_ohm_per_km = 1.5042, x_ohm_per_km = 1.3554, c_nf_per_km = 0, max_i_ka = 9, name='Line 19-20')
        line20_21 = pp.create_line_from_parameters(net_ieee33, buses[19], buses[20], length_km=1, r_ohm_per_km = 0.4095, x_ohm_per_km = 0.4784, c_nf_per_km = 0, max_i_ka = 9, name='Line 20-21')
        line21_22 = pp.create_line_from_parameters(net_ieee33, buses[20], buses[21], length_km=1, r_ohm_per_km = 0.7089, x_ohm_per_km = 0.9373, c_nf_per_km = 0, max_i_ka = 9, name='Line 21-22')
        line2_23 = pp.create_line_from_parameters(net_ieee33, buses[2], buses[22], length_km=1, r_ohm_per_km = 0.4512, x_ohm_per_km = 0.3083, c_nf_per_km = 0, max_i_ka = 9, name='Line 3-23')
        line23_24 = pp.create_line_from_parameters(net_ieee33, buses[22], buses[23], length_km=1, r_ohm_per_km = 0.8980, x_ohm_per_km = 0.7091, c_nf_per_km = 0, max_i_ka = 9, name='Line 23-24')
        line24_25 = pp.create_line_from_parameters(net_ieee33, buses[23], buses[24], length_km=1, r_ohm_per_km = 0.8960, x_ohm_per_km = 0.7011, c_nf_per_km = 0, max_i_ka = 9, name='Line 24-25')
        line6_26 = pp.create_line_from_parameters(net_ieee33, buses[5], buses[25], length_km=1, r_ohm_per_km = 0.2030, x_ohm_per_km = 0.1034, c_nf_per_km = 0, max_i_ka = 9, name='Line 6-26')
        line26_27 = pp.create_line_from_parameters(net_ieee33, buses[25], buses[26], length_km=1, r_ohm_per_km = 0.2842, x_ohm_per_km = 0.1447, c_nf_per_km = 0, max_i_ka = 9, name='Line 26-27')
        line27_28 = pp.create_line_from_parameters(net_ieee33, buses[26], buses[27], length_km=1, r_ohm_per_km = 1.0590, x_ohm_per_km = 0.9337, c_nf_per_km = 0, max_i_ka = 9, name='Line 27-28')
        line28_29 = pp.create_line_from_parameters(net_ieee33, buses[27], buses[28], length_km=1, r_ohm_per_km = 0.8042, x_ohm_per_km = 0.7006, c_nf_per_km = 0, max_i_ka = 9, name='Line 28-29')
        line29_30 = pp.create_line_from_parameters(net_ieee33, buses[28], buses[29], length_km=1, r_ohm_per_km = 0.5075, x_ohm_per_km = 0.2585, c_nf_per_km = 0, max_i_ka = 9, name='Line 29-30')
        line30_31 = pp.create_line_from_parameters(net_ieee33, buses[29], buses[30], length_km=1, r_ohm_per_km = 0.9744, x_ohm_per_km = 0.9630, c_nf_per_km = 0, max_i_ka = 9, name='Line 30-31')
        line31_32 = pp.create_line_from_parameters(net_ieee33, buses[30], buses[31], length_km=1, r_ohm_per_km = 0.3105, x_ohm_per_km = 0.3619, c_nf_per_km = 0, max_i_ka = 9, name='Line 31-32')
        line32_33 = pp.create_line_from_parameters(net_ieee33, buses[31], buses[32], length_km=1, r_ohm_per_km = 0.3410, x_ohm_per_km = 0.5302, c_nf_per_km = 0, max_i_ka = 9, name='Line 32-33')
        
            # Ext-Grid
        pp.create_ext_grid(net_ieee33, buses[0], vm_pu=1.0, va_degree=0.0)


            # Switches
            # S2
       
            # Loads
            # Residential
        pp.create_load(net_ieee33, buses[1], p_mw = 0.1 , q_mvar = 0.06 , scaling = self.LOAD[self.time], name='Load R2')
        pp.create_load(net_ieee33, buses[2], p_mw = 0.09 , q_mvar = 0.04 , scaling = self.LOAD[self.time], name='Load R3')
        pp.create_load(net_ieee33, buses[3], p_mw = 0.12 , q_mvar = 0.08 , scaling = self.LOAD[self.time], name='Load R4')
        pp.create_load(net_ieee33, buses[4], p_mw = 0.06 , q_mvar = 0.03 , scaling = self.LOAD[self.time], name='Load R5')
        pp.create_load(net_ieee33, buses[5], p_mw = 0.06 , q_mvar = 0.02 , scaling = self.LOAD[self.time], name='Load R6')
        pp.create_load(net_ieee33, buses[6], p_mw = 0.2 , q_mvar = 0.1 , scaling = self.LOAD[self.time], name='Load R7')
        pp.create_load(net_ieee33, buses[7], p_mw = 0.2 , q_mvar = 0.1 , scaling = self.LOAD[self.time], name='Load R8')
        pp.create_load(net_ieee33, buses[8], p_mw = 0.06 , q_mvar = 0.02 , scaling = self.LOAD[self.time], name='Load R9')
        pp.create_load(net_ieee33, buses[9], p_mw = 0.06 , q_mvar = 0.02 , scaling = self.LOAD[self.time], name='Load R10')
        pp.create_load(net_ieee33, buses[10], p_mw = 0.045 , q_mvar = 0.03 , scaling = self.LOAD[self.time], name='Load R11')
        pp.create_load(net_ieee33, buses[11], p_mw = 0.06 , q_mvar = 0.035 , scaling = self.LOAD[self.time], name='Load R12')
        pp.create_load(net_ieee33, buses[12], p_mw = 0.06 , q_mvar = 0.035 , scaling = self.LOAD[self.time], name='Load R13')
        pp.create_load(net_ieee33, buses[13], p_mw = 0.12 , q_mvar = 0.08 , scaling = self.LOAD[self.time], name='Load R14')
        pp.create_load(net_ieee33, buses[14], p_mw = 0.06 , q_mvar = 0.01 , scaling = self.LOAD[self.time], name='Load R15')
        pp.create_load(net_ieee33, buses[15], p_mw = 0.06 , q_mvar = 0.02 , scaling = self.LOAD[self.time], name='Load R16')
        pp.create_load(net_ieee33, buses[16], p_mw = 0.06 , q_mvar = 0.02 , scaling = self.LOAD[self.time], name='Load R17')
        pp.create_load(net_ieee33, buses[17], p_mw = 0.09 , q_mvar = 0.04 , scaling = self.LOAD[self.time], name='Load R18')
        pp.create_load(net_ieee33, buses[18], p_mw = 0.09 , q_mvar = 0.04 , scaling = self.LOAD[self.time], name='Load R19')
        pp.create_load(net_ieee33, buses[19], p_mw = 0.09 , q_mvar = 0.04 , scaling = self.LOAD[self.time], name='Load R20')
        pp.create_load(net_ieee33, buses[20], p_mw = 0.09 , q_mvar = 0.04 , scaling = self.LOAD[self.time], name='Load R21')
        pp.create_load(net_ieee33, buses[21], p_mw = 0.09 , q_mvar = 0.04 , scaling = self.LOAD[self.time], name='Load R22')
        pp.create_load(net_ieee33, buses[22], p_mw = 0.09 , q_mvar = 0.05 , scaling = self.LOAD[self.time], name='Load R23')
        pp.create_load(net_ieee33, buses[23], p_mw = 0.42 , q_mvar = 0.2 , scaling = self.LOAD[self.time], name='Load R24')
        pp.create_load(net_ieee33, buses[24], p_mw = 0.42 , q_mvar = 0.2 , scaling = self.LOAD[self.time], name='Load R25')
        pp.create_load(net_ieee33, buses[25], p_mw = 0.06 , q_mvar = 0.025 , scaling = self.LOAD[self.time], name='Load R26')
        pp.create_load(net_ieee33, buses[26], p_mw = 0.06 , q_mvar = 0.025 , scaling = self.LOAD[self.time], name='Load R27')
        pp.create_load(net_ieee33, buses[27], p_mw = 0.06 , q_mvar = 0.02 , scaling = self.LOAD[self.time], name='Load R28')
        pp.create_load(net_ieee33, buses[28], p_mw = 0.12 , q_mvar = 0.07 , scaling = self.LOAD[self.time], name='Load R29')
        pp.create_load(net_ieee33, buses[29], p_mw = 0.2 , q_mvar = 0.6 , scaling = self.LOAD[self.time], name='Load R30')
        pp.create_load(net_ieee33, buses[30], p_mw = 0.15 , q_mvar = 0.07 , scaling = self.LOAD[self.time], name='Load R31')
        pp.create_load(net_ieee33, buses[31], p_mw = 0.21 , q_mvar = 0.1 , scaling = self.LOAD[self.time], name='Load R32')
        pp.create_load(net_ieee33, buses[32], p_mw = 0.06 , q_mvar = 0.04 , scaling = self.LOAD[self.time], name='Load R33')

            # Optional distributed energy recources
        '''
        pp.create_sgen(net_ieee33, buses[2], self.PV[self.time]*0.2, q_mvar=0,  name='PV 3', type='PV')
        pp.create_sgen(net_ieee33, buses[7], self.PV[self.time]*0.2, q_mvar=0,  name='PV 8', type='PV')
        pp.create_sgen(net_ieee33, buses[13], self.PV[self.time]*0.2, q_mvar=0,  name='PV 14', type='PV')
        pp.create_sgen(net_ieee33, buses[24], self.PV[self.time]*0.3, q_mvar=0,  name='PV 25', type='PV')
        pp.create_sgen(net_ieee33, buses[29], self.PV[self.time]*0.3, q_mvar=0,  name='PV 29', type='PV')
        pp.create_sgen(net_ieee33, buses[30], self.PV[self.time]*0.5, q_mvar=0,  name='PV 30', type='PV')'''

        pp.create_sgen(net_ieee33, buses[2],  self.PV[self.time]*1.0*(1+((np.random.rand()-0.5)*2)*0.1) , q_mvar=0,  name='PV 3', type='PV')
        pp.create_sgen(net_ieee33, buses[7],  self.PV[self.time]*1.0*(1+((np.random.rand()-0.5)*2)*0.1) , q_mvar=0,  name='PV 8', type='PV')
        pp.create_sgen(net_ieee33, buses[13], self.PV[self.time]*1.0*(1+((np.random.rand()-0.5)*2)*0.1) , q_mvar=0,  name='PV 14', type='PV')
        pp.create_sgen(net_ieee33, buses[24], self.PV[self.time]*0.8*(1+((np.random.rand()-0.5)*2)*0.1) , q_mvar=0,  name='PV 25', type='PV')
        pp.create_sgen(net_ieee33, buses[29], self.PV[self.time]*0.9*(1+((np.random.rand()-0.5)*2)*0.1) , q_mvar=0,  name='PV 29', type='PV')
        pp.create_sgen(net_ieee33, buses[30], self.PV[self.time]*1.0*(1+((np.random.rand()-0.5)*2)*0.1) , q_mvar=0,  name='PV 30', type='PV')


        pp.create_storage(net_ieee33, buses[1], p_mw = o, max_e_mwh = 2.5, soc_percent = soc, min_e_mwh = 0)
        pp.runpp(net_ieee33)
        
        self.busv_pu = np.array([net_ieee33.res_bus.vm_pu])
        self.line_percent = np.array([net_ieee33.res_line.p_to_mw]) 
        self.soc = np.array([net_ieee33.storage.soc_percent])
        self.p_mw = np.array([net_ieee33.storage.p_mw])

        '''    
        pv_ = np.array([self.PV[(self.time -2)% self.MaxTime],self.PV[(self.time -1)% self.MaxTime], self.PV[(self.time)% self.MaxTime], self.PV[(self.time + 1)% self.MaxTime],self.PV[(self.time + 2)% self.MaxTime]])
        load_ = np.array([self.LOAD[(self.time -2)% self.MaxTime],self.LOAD[(self.time -1)% self.MaxTime], self.LOAD[(self.time)% self.MaxTime], self.LOAD[(self.time + 1)% self.MaxTime],self.LOAD[(self.time + 2)% self.MaxTime]])
        smp_ = np.array([self.SMP[(self.time-2)% self.MaxTime],self.SMP[(self.time-1)% self.MaxTime], self.SMP[(self.time)% self.MaxTime], self.SMP[(self.time + 1)% self.MaxTime],self.SMP[(self.time + 2)% self.MaxTime]])
        '''
        pv_ = np.array([self.PV[(self.time)% self.MaxTime]-self.PV[(self.time -1)% self.MaxTime]])
        load_ = np.array([self.LOAD[(self.time)% self.MaxTime]-self.LOAD[(self.time -1)% self.MaxTime]])
        smp_ = np.array([self.SMP[(self.time)% self.MaxTime]-self.SMP[(self.time -1)% self.MaxTime]])
        
        state1 = np.hstack([pv_, load_, smp_, self.soc[0]+0.4*self.p_mw[0], self.p_mw[0], self.line_percent[0][0], self.SMP[self.time]]).flatten()
        next_states = [state1]

        info = [o , next_states[0][3], self.line_percent[0][0],self.SMP[self.time]]
        infos = net_ieee33.res_sgen.p_mw

        
        total_reward +=  self.line_percent[0][0]*self.SMP[self.time]

        
        if self.time == 23:
            if self.soc[0]+0.4*self.p_mw[0] < 0.4:
                total_reward1 = -10000
            else:
                total_reward1 = 10000
            
        else :
            total_reward1 = 0


        
        self.rwd = np.zeros(shape=(1,))        
        self.rwd[0] = total_reward + total_reward1 

        total_rewards=self.rwd
        return next_states , total_rewards , info, infos