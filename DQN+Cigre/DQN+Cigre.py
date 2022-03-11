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


colors = seaborn.color_palette()
net_cigre_mv = pp.create_empty_network()

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

    
class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 15)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else : 
            return out.argmax().item()    
    
    
    
    
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)
        q_out = q(s)
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
    
def main():
    env=env()
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    print_interval = 20
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    for n_epi in range(10000):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        s = env.reset()
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)      
            s_prime, r, done, info = env.net(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r/100.0,s_prime, done_mask))
            s = s_prime
            score += r
            if done:
                break
            
        if memory.size()>2000:
            train(q, q_target, memory, optimizer)

        if n_epi%print_interval==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0
    env.close()
if __name__ == '__main__':
    main()    
    
    
def bet(a,b,P):
    aa = beta.rvs(a, b, size=1)*P
    return aa

def win(a,mn,med,mx,Pmx):
    Vwind=np.random.weibull(a, 1) * 15

    if Vwind<mn:
        return 0
    elif mn<Vwind<med:
        return Pmx*(Vwind**2-mn**2)/(med**2-mn**2)
    elif med<Vwind<mx:
        return Pmx
    else:
        return 0
    

class env():
    def reset(self):
        self.PV = bet(5,2,1)[0]       
        self.WT = int(win(1.5, 3,15,25,150))/150
        self.LOAD = (bet(0.5,0.5,1)[0]+1)/1.5
        
        return np.array(PV,WT,LOAD)
    
    def net(out):
        PV = bet(5,2,1)[0]       
        WT = int(win(1.5, 3,15,25,150))/150
        LOAD = (bet(0.5,0.5,1)[0]+1)/1.5
                
        out = out
    
        # Linedata
        line_data = {'c_nf_per_km': 151.1749, 'r_ohm_per_km': 0.501,'x_ohm_per_km': 0.716, 'max_i_ka': 0.145,'type': 'cs'}
        pp.create_std_type(net_cigre_mv, line_data, name='CABLE_CIGRE_MV', element='line')
        line_data = {'c_nf_per_km': 10.09679, 'r_ohm_per_km': 0.510,'x_ohm_per_km': 0.366, 'max_i_ka': 0.195,'type': 'ol'}
        pp.create_std_type(net_cigre_mv, line_data, name='OHL_CIGRE_MV', element='line')

        # Busses
        bus0 = pp.create_bus(net_cigre_mv, name='Bus 0', vn_kv=110, type='b', zone='CIGRE_MV')
        buses = pp.create_buses(net_cigre_mv, 14, name=['Bus %i' % i for i in range(1, 15)], vn_kv=20,type='b', zone='CIGRE_MV')

        # Lines
        line1_2 = pp.create_line(net_cigre_mv, buses[0], buses[1], length_km=2.82,std_type='CABLE_CIGRE_MV', name='Line 1-2')
        line2_3 = pp.create_line(net_cigre_mv, buses[1], buses[2], length_km=4.42,std_type='CABLE_CIGRE_MV', name='Line 2-3')
        line3_4 = pp.create_line(net_cigre_mv, buses[2], buses[3], length_km=0.61,std_type='CABLE_CIGRE_MV', name='Line 3-4')
        line4_5 = pp.create_line(net_cigre_mv, buses[3], buses[4], length_km=0.56,std_type='CABLE_CIGRE_MV', name='Line 4-5')
        line5_6 = pp.create_line(net_cigre_mv, buses[4], buses[5], length_km=1.54,std_type='CABLE_CIGRE_MV', name='Line 5-6')
        line7_8 = pp.create_line(net_cigre_mv, buses[6], buses[7], length_km=1.67,std_type='CABLE_CIGRE_MV', name='Line 7-8')
        line8_9 = pp.create_line(net_cigre_mv, buses[7], buses[8], length_km=0.32,std_type='CABLE_CIGRE_MV', name='Line 8-9')
        line9_10 = pp.create_line(net_cigre_mv, buses[8], buses[9], length_km=0.77,std_type='CABLE_CIGRE_MV', name='Line 9-10')
        line10_11 = pp.create_line(net_cigre_mv, buses[9], buses[10], length_km=0.33,std_type='CABLE_CIGRE_MV', name='Line 10-11')
        line3_8 = pp.create_line(net_cigre_mv, buses[2], buses[7], length_km=1.3,std_type='CABLE_CIGRE_MV', name='Line 3-8')
        line12_13 = pp.create_line(net_cigre_mv, buses[11], buses[12], length_km=4.89,std_type='OHL_CIGRE_MV', name='Line 12-13')
        line13_14 = pp.create_line(net_cigre_mv, buses[12], buses[13], length_km=2.99,std_type='OHL_CIGRE_MV', name='Line 13-14')

        line6_7 = pp.create_line(net_cigre_mv, buses[5], buses[6], length_km=0.24,std_type='CABLE_CIGRE_MV', name='Line 6-7')
        line4_11 = pp.create_line(net_cigre_mv, buses[10], buses[3], length_km=0.49,std_type='CABLE_CIGRE_MV', name='Line 11-4')
        line8_14 = pp.create_line(net_cigre_mv, buses[13], buses[7], length_km=2.,std_type='OHL_CIGRE_MV', name='Line 14-8')

            # Ext-Grid
        pp.create_ext_grid(net_cigre_mv, bus0, vm_pu=1.03, va_degree=0.,s_sc_max_mva=5000, s_sc_min_mva=5000, rx_max=0.1, rx_min=0.1)

            # Trafos
        trafo0 = pp.create_transformer_from_parameters(net_cigre_mv, bus0, buses[0], sn_mva=25,vn_hv_kv=110, vn_lv_kv=20, vkr_percent=0.16,
                                                           vk_percent=12.00107, pfe_kw=0, i0_percent=0,shift_degree=30.0, name='Trafo 0-1')
        trafo1 = pp.create_transformer_from_parameters(net_cigre_mv, bus0, buses[11], sn_mva=25,vn_hv_kv=110, vn_lv_kv=20, vkr_percent=0.16,
                                                           vk_percent=12.00107, pfe_kw=0, i0_percent=0,shift_degree=30.0, name='Trafo 0-12')

            # Switches
            # S2


        pp.create_switch(net_cigre_mv, buses[5], line6_7, et='l', closed=o[0], type='LBS')
        #pp.create_switch(net_cigre_mv, buses[6], line6_7, et='l', closed=False, type='LBS', name='S2')
            # S3
        #pp.create_switch(net_cigre_mv, buses[3], line4_11, et='l', closed=False, type='LBS', name='S3')
        pp.create_switch(net_cigre_mv, buses[10], line4_11, et='l', closed=o[1], type='LBS')
            # S1
        #pp.create_switch(net_cigre_mv, buses[7], line8_14, et='l', closed=False, type='LBS', name='S1')
        pp.create_switch(net_cigre_mv, buses[13], line8_14, et='l', closed=o[2], type='LBS')
            # trafos
        pp.create_switch(net_cigre_mv, bus0, trafo0, et='t', closed=1, type='CB')
        pp.create_switch(net_cigre_mv, bus0, trafo1, et='t', closed=1, type='CB')

        # Switches
        pp.create_switch(net_cigre_mv, buses[0], line1_2, et='l', closed=o[3], type='LBS')
        pp.create_switch(net_cigre_mv, buses[1], line2_3, et='l', closed=o[4], type='LBS')
        pp.create_switch(net_cigre_mv, buses[2], line3_4, et='l', closed=o[5], type='LBS')
        pp.create_switch(net_cigre_mv, buses[3], line4_5, et='l', closed=o[6], type='LBS')
        pp.create_switch(net_cigre_mv, buses[4], line5_6, et='l', closed=o[7], type='LBS')
        pp.create_switch(net_cigre_mv, buses[6], line7_8, et='l', closed=o[8], type='LBS')
        pp.create_switch(net_cigre_mv, buses[7], line8_9, et='l', closed=o[9], type='LBS')
        pp.create_switch(net_cigre_mv, buses[8], line9_10, et='l', closed=o[10], type='LBS')
        pp.create_switch(net_cigre_mv, buses[9], line10_11, et='l', closed=o[11], type='LBS')
        pp.create_switch(net_cigre_mv, buses[2], line3_8, et='l', closed=o[12], type='LBS')
        pp.create_switch(net_cigre_mv, buses[11], line12_13, et='l', closed=o[13], type='LBS')
        pp.create_switch(net_cigre_mv, buses[12], line13_14, et='l', closed=o[14], type='LBS')



            # Loads
            # Residential
        pp.create_load_from_cosphi(net_cigre_mv, buses[0], LOAD*15.3, 0.98, "ind", name='Load R1')
        pp.create_load_from_cosphi(net_cigre_mv, buses[2], LOAD*0.285, 0.97, "ind", name='Load R3')
        pp.create_load_from_cosphi(net_cigre_mv, buses[3], LOAD*0.445, 0.97, "ind", name='Load R4')
        pp.create_load_from_cosphi(net_cigre_mv, buses[4], LOAD*0.750, 0.97, "ind", name='Load R5')
        pp.create_load_from_cosphi(net_cigre_mv, buses[5], LOAD*0.565, 0.97, "ind", name='Load R6')
        pp.create_load_from_cosphi(net_cigre_mv, buses[7], LOAD*0.605, 0.97, "ind", name='Load R8')
        pp.create_load_from_cosphi(net_cigre_mv, buses[9], LOAD*0.490, 0.97, "ind", name='Load R10')
        pp.create_load_from_cosphi(net_cigre_mv, buses[10], LOAD*0.340, 0.97, "ind", name='Load R11')
        pp.create_load_from_cosphi(net_cigre_mv, buses[11], LOAD*15.3, 0.98, "ind", name='Load R12')
        pp.create_load_from_cosphi(net_cigre_mv, buses[13], LOAD*0.215, 0.97, "ind", name='Load R14')

            # Commercial / Industrial
        pp.create_load_from_cosphi(net_cigre_mv, buses[0], LOAD*5.1, 0.95, "ind", name='Load CI1')
        pp.create_load_from_cosphi(net_cigre_mv, buses[2], LOAD*0.265, 0.85, "ind", name='Load CI3')
        pp.create_load_from_cosphi(net_cigre_mv, buses[6], LOAD*0.090, 0.85, "ind", name='Load CI7')
        pp.create_load_from_cosphi(net_cigre_mv, buses[8], LOAD*0.675, 0.85, "ind", name='Load CI9')
        pp.create_load_from_cosphi(net_cigre_mv, buses[9], LOAD*0.080, 0.85, "ind", name='Load CI10')
        pp.create_load_from_cosphi(net_cigre_mv, buses[11], LOAD*5.28, 0.95, "ind", name='Load CI12')
        pp.create_load_from_cosphi(net_cigre_mv, buses[12], LOAD*0.04, 0.85, "ind", name='Load CI13')
        pp.create_load_from_cosphi(net_cigre_mv, buses[13], LOAD*0.390, 0.85, "ind", name='Load CI14')


            # Optional distributed energy recources

        pp.create_sgen(net_cigre_mv, buses[4], PV*0.02, q_mvar=0, sn_mva=0.02, name='PV 3', type='PV')
        pp.create_sgen(net_cigre_mv, buses[5], PV*0.02, q_mvar=0, sn_mva=0.02, name='PV 4', type='PV')
        pp.create_sgen(net_cigre_mv, buses[6], PV*0.02, q_mvar=0, sn_mva=0.03, name='PV 5', type='PV')
        pp.create_sgen(net_cigre_mv, buses[8], PV*0.03, q_mvar=0, sn_mva=0.03, name='PV 6', type='PV')
        pp.create_sgen(net_cigre_mv, buses[9], PV*0.03, q_mvar=0, sn_mva=0.03, name='PV 8', type='PV')
        pp.create_sgen(net_cigre_mv, buses[10], WT*1.5, q_mvar=0, sn_mva=0.03, name='PV 9', type='WP')
        pp.create_sgen(net_cigre_mv, buses[11], WT*1.5, q_mvar=0, sn_mva=0.04, name='PV 10', type='WP')
        pp.create_sgen(net_cigre_mv, buses[12], WT*1.5, q_mvar=0, sn_mva=0.01, name='PV 11', type='WP')
        pp.create_sgen(net_cigre_mv, buses[13], WT*1.5, q_mvar=0, sn_mva=1.5, name='WKA 7', type='WP')

        #pp.create_storage(net_cigre_mv, bus=buses[4], p_mw=0.6, max_e_mwh=nan, sn_mva=0.2,name='Battery 1', type='Battery', max_p_mw=0.6, min_p_mw=-0.6)
        pp.create_sgen(net_cigre_mv, bus=buses[4], p_mw=0.033, sn_mva=0.033,name='Residential fuel cell 1', type='Residential fuel cell')
        pp.create_sgen(net_cigre_mv, bus=buses[8], p_mw=0.310, sn_mva=0.31, name='CHP diesel 1',type='CHP diesel')
        pp.create_sgen(net_cigre_mv, bus=buses[8], p_mw=0.212, sn_mva=0.212, name='Fuel cell 1',type='Fuel cell')
        #pp.create_storage(net_cigre_mv, bus=buses[9], p_mw=0.200, max_e_mwh=nan, sn_mva=0.2,name='Battery 2', type='Battery', max_p_mw=0.2, min_p_mw=-0.2)
        pp.create_sgen(net_cigre_mv, bus=buses[9], p_mw=0.014, sn_mva=.014,name='Residential fuel cell 2', type='Residential fuel cell')

            # Bus geo data
        net_cigre_mv.bus_geodata = read_json("""{"x":{"0":7.0,"1":4.0,"2":4.0,"3":4.0,"4":2.5,"5":1.0,"6":1.0,"7":8.0,"8":8.0,"9":6.0,"10":4.0,"11":4.0,"12":10.0,"13":10.0,"14":10.0},"y":{"0":16,"1":15,"2":13,"3":11,"4":9,"5":7,"6":3,"7":3,"8":5,"9":5,"10":5,"11":7,"12":15,"13":11,"14":5}}""")
            # Match bus.index
        net_cigre_mv.bus_geodata = net_cigre_mv.bus_geodata.loc[net_cigre_mv.bus.index]
        #ax = plot.simple_plot(net_cigre_mv, show_plot=False)




        bc = plot.create_bus_collection(net_cigre_mv, buses=net_cigre_mv.bus.index, color=colors[0], size=0.1, zorder=3)
        lc = plot.create_line_collection(net_cigre_mv, lines=net_cigre_mv.line.index, color=colors[1], zorder=2)
        load = plot.create_load_collection(net_cigre_mv, loads=net_cigre_mv.load.index, size=0.4)
        ext_grid = plot.create_ext_grid_collection(net_cigre_mv, size=0.2)

        #plot.draw_collections([load,ext_grid], ax=ax)


        pp.runopp(net_cigre_mv)
        print(net_cigre_mv.res_bus.vm_pu)
        print(net_cigre_mv.res_line.loading_percent)
        mg = top.create_nxgraph(net_cigre_mv, nogobuses =[0] )
        radial = nx.has_path(mg,1,12)

        a=1.5
        b=1.5
        Pbus = 0
        Pline = 0 
        for i in range(len(net_cigre_mv.res_line)):
            line = net_cigre_mv.res_line.loading_percent[i]
            if line < 100:
                Pline += 0

            else :
                Pline += -a*(line-100)/len(net_cigre_mv.res_line)/len(net_cigre_mv.res_line)


        for i in range(len(net_cigre_mv.res_bus)):
            Vbus = net_cigre_mv.res_bus.vm_pu[i]
            if Vbus < 0.95:
                Pbus += -100*b*(0.95-Vbus)/len(net_cigre_mv.res_bus)


            elif 0.95 <= Vbus < 1.05:
                Pbus += 0

            else :
                Pbus += -100*b*(Vbus-1.05)/len(net_cigre_mv.res_bus)
        
        print("Pline = %f" %Pline ,"Pbus = %f"%Pbus, "radial %d"%radial)
        return PV, WT, LOAD, Pline, Pbus, radial

o = [1,1,0,1,1,1,1,1,1,1,1,1,1,1,1]
env.net(out=o)
