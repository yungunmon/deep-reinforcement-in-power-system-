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


def opf(bat):

    PV = [0.000000001, 0.000000001, 0.000000001, 0.000000001, 0.000000001, 0.000000001, 0.04675, 0.28, 0.52625, 0.7765, 0.9, 0.975, 1, 0.75, 0.625, 0.375, 0.1665, 0.095, 0.000000001, 0.000000001, 0.000000001, 0.000000001, 0.000000001, 0.000000001]
    LOAD = [0.793681328 , 0.749258257, 0.726207166, 0.727527632, 0.74872029,  0.778324802, 0.815623879, 0.859231848, 0.937367546, 0.982393792, 0.995875583, 0.99258257,  0.92934694,  0.970803039, 0.989501483, 0.991816374, 1,   0.997522089,
     0.994343191, 0.986029148, 0.953262039, 0.92862965,  0.946692315, 0.952903394]
    time = bat 
    net_ieee39 = pp.create_empty_network()        

    # Busses
    bus1  = pp.create_bus(net_ieee39, name='Bus 1' , vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus2  = pp.create_bus(net_ieee39, name='Bus 2' , vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus3  = pp.create_bus(net_ieee39, name='Bus 3' , vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus4  = pp.create_bus(net_ieee39, name='Bus 4' , vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus5  = pp.create_bus(net_ieee39, name='Bus 5' , vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus6  = pp.create_bus(net_ieee39, name='Bus 6' , vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus7  = pp.create_bus(net_ieee39, name='Bus 7' , vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus8  = pp.create_bus(net_ieee39, name='Bus 8' , vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus9  = pp.create_bus(net_ieee39, name='Bus 9' , vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus10 = pp.create_bus(net_ieee39, name='Bus 10', vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus11 = pp.create_bus(net_ieee39, name='Bus 11', vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus12 = pp.create_bus(net_ieee39, name='Bus 12', vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus13 = pp.create_bus(net_ieee39, name='Bus 13', vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus14 = pp.create_bus(net_ieee39, name='Bus 14', vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus15 = pp.create_bus(net_ieee39, name='Bus 15', vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus16 = pp.create_bus(net_ieee39, name='Bus 16', vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus17 = pp.create_bus(net_ieee39, name='Bus 17', vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus18 = pp.create_bus(net_ieee39, name='Bus 18', vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus19 = pp.create_bus(net_ieee39, name='Bus 19', vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus20 = pp.create_bus(net_ieee39, name='Bus 20', vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus21 = pp.create_bus(net_ieee39, name='Bus 21', vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus22 = pp.create_bus(net_ieee39, name='Bus 22', vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus23 = pp.create_bus(net_ieee39, name='Bus 23', vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus24 = pp.create_bus(net_ieee39, name='Bus 24', vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus25 = pp.create_bus(net_ieee39, name='Bus 25', vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus26 = pp.create_bus(net_ieee39, name='Bus 26', vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus27 = pp.create_bus(net_ieee39, name='Bus 27', vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus28 = pp.create_bus(net_ieee39, name='Bus 28', vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus29 = pp.create_bus(net_ieee39, name='Bus 29', vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus30 = pp.create_bus(net_ieee39, name='Bus 30', vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus31 = pp.create_bus(net_ieee39, name='Bus 31', vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus32 = pp.create_bus(net_ieee39, name='Bus 32', vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus33 = pp.create_bus(net_ieee39, name='Bus 33', vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus34 = pp.create_bus(net_ieee39, name='Bus 34', vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus35 = pp.create_bus(net_ieee39, name='Bus 35', vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus36 = pp.create_bus(net_ieee39, name='Bus 36', vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus37 = pp.create_bus(net_ieee39, name='Bus 37', vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus38 = pp.create_bus(net_ieee39, name='Bus 38', vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')
    bus39 = pp.create_bus(net_ieee39, name='Bus 39', vn_kv=345 , max_vm_pu = 1.02, min_vm_pu = 0.95, type='b', zone='IEEE39')

    # Lines
    line1_2   =  pp.create_line_from_parameters(net_ieee39, bus1  , bus2  , length_km=1, r_ohm_per_km = 4.165875, x_ohm_per_km = 48.9192745, c_nf_per_km = 1557.117674694494099, max_i_ka = 1.00408742467761, name='Line 1-2')
    line1_39  =  pp.create_line_from_parameters(net_ieee39, bus1  , bus39 , length_km=1, r_ohm_per_km = 1.19025 , x_ohm_per_km = 29.75625  , c_nf_per_km = 1671.444476915515224, max_i_ka = 1.67347904112935, name='Line 2-3')
    line6_7   =  pp.create_line_from_parameters(net_ieee39, bus6  , bus7  , length_km=1, r_ohm_per_km = 0.71415 , x_ohm_per_km = 10.9503   , c_nf_per_km = 251.830967855270927 , max_i_ka = 1.50613113701642, name='Line 3-4')
    line6_11  =  pp.create_line_from_parameters(net_ieee39, bus6  , bus11 , length_km=1, r_ohm_per_km = 0.833175, x_ohm_per_km = 9.76005   , c_nf_per_km = 309.551517124753389 , max_i_ka = 0.80326993974209, name='Line 4-5')
    line7_8   =  pp.create_line_from_parameters(net_ieee39, bus7  , bus8  , length_km=1, r_ohm_per_km = 0.4761  , x_ohm_per_km = 5.47515   , c_nf_per_km = 173.830225599213577 , max_i_ka = 1.50613113701642, name='Line 5-6')
    line8_9   =  pp.create_line_from_parameters(net_ieee39, bus8  , bus9  , length_km=1, r_ohm_per_km = 2.737575, x_ohm_per_km = 43.206075 , c_nf_per_km = 847.75663869154937  , max_i_ka = 1.50613113701642, name='Line 6-7')
    line9_39  =  pp.create_line_from_parameters(net_ieee39, bus9  , bus39 , length_km=1, r_ohm_per_km = 1.19025 , x_ohm_per_km = 29.75625  , c_nf_per_km = 2674.311163064824086, max_i_ka = 1.50613113701642, name='Line 7-8')
    line10_11 =  pp.create_line_from_parameters(net_ieee39, bus10 , bus11 , length_km=1, r_ohm_per_km = 0.4761  , x_ohm_per_km = 5.118075  , c_nf_per_km = 162.464403156188098 , max_i_ka = 1.00408742467761, name='Line 8-9')
    line10_13 =  pp.create_line_from_parameters(net_ieee39, bus10 , bus13 , length_km=1, r_ohm_per_km = 0.4761  , x_ohm_per_km = 5.118075  , c_nf_per_km = 162.464403156188098 , max_i_ka = 1.00408742467761, name='Line 9-10')
    line13_14 =  pp.create_line_from_parameters(net_ieee39, bus13 , bus14 , length_km=1, r_ohm_per_km = 1.071225, x_ohm_per_km = 12.021525 , c_nf_per_km = 383.986511163391072 , max_i_ka = 1.00408742467761, name='Line 10-11')
    line14_15 =  pp.create_line_from_parameters(net_ieee39, bus14 , bus15 , length_km=1, r_ohm_per_km = 2.14245 , x_ohm_per_km = 25.828425 , c_nf_per_km = 815.664904734771426 , max_i_ka = 1.00408742467761, name='Line 11-12')
    line15_16 =  pp.create_line_from_parameters(net_ieee39, bus15 , bus16 , length_km=1, r_ohm_per_km = 1.071225, x_ohm_per_km = 11.18835  , c_nf_per_km = 381.089340736737483 , max_i_ka = 1.00408742467761, name='Line 12-13')
    line2_3   =  pp.create_line_from_parameters(net_ieee39, bus2  , bus3  , length_km=1, r_ohm_per_km = 1.547325, x_ohm_per_km = 17.972775 , c_nf_per_km = 573.194025950227342 , max_i_ka = 0.83673952056468, name='Line 13-14')
    line16_17 =  pp.create_line_from_parameters(net_ieee39, bus16 , bus17 , length_km=1, r_ohm_per_km = 0.833175, x_ohm_per_km = 10.593225 , c_nf_per_km = 573.194025950227342 , max_i_ka = 1.00408742467761, name='Line 14-15')
    line16_19 =  pp.create_line_from_parameters(net_ieee39, bus16 , bus19 , length_km=1, r_ohm_per_km = 1.9044  , x_ohm_per_km = 23.209875 , c_nf_per_km = 677.492161309755375 , max_i_ka = 1.00408742467761, name='Line 15-16')
    line16_21 =  pp.create_line_from_parameters(net_ieee39, bus16 , bus21 , length_km=1, r_ohm_per_km = 0.9522  , x_ohm_per_km = 16.068375 , c_nf_per_km = 567.845403624097685 , max_i_ka = 1.00408742467761, name='Line 16-17')
    line16_24 =  pp.create_line_from_parameters(net_ieee39, bus16 , bus24 , length_km=1, r_ohm_per_km = 0.357075, x_ohm_per_km = 7.022475  , c_nf_per_km = 151.544299240340052 , max_i_ka = 1.00408742467761, name='Line 17-18')
    line17_18 =  pp.create_line_from_parameters(net_ieee39, bus17 , bus18 , length_km=1, r_ohm_per_km = 0.833175, x_ohm_per_km = 9.76005   , c_nf_per_km = 293.951368673541879 , max_i_ka = 1.00408742467761, name='Line 2-19')
    line17_27 =  pp.create_line_from_parameters(net_ieee39, bus17 , bus27 , length_km=1, r_ohm_per_km = 1.547325, x_ohm_per_km = 20.591325 , c_nf_per_km = 716.715391701372937 , max_i_ka = 1.00408742467761, name='Line 19-20')
    line21_22 =  pp.create_line_from_parameters(net_ieee39, bus21 , bus22 , length_km=1, r_ohm_per_km = 0.9522  , x_ohm_per_km = 16.6635   , c_nf_per_km = 571.634011105106197 , max_i_ka = 1.50613113701642, name='Line 20-21')
    line22_23 =  pp.create_line_from_parameters(net_ieee39, bus22 , bus23 , length_km=1, r_ohm_per_km = 0.71415 , x_ohm_per_km = 11.4264   , c_nf_per_km = 411.398200584805466 , max_i_ka = 1.00408742467761, name='Line 21-22')
    line23_24 =  pp.create_line_from_parameters(net_ieee39, bus23 , bus24 , length_km=1, r_ohm_per_km = 2.61855 , x_ohm_per_km = 41.65875  , c_nf_per_km = 804.521941555334706 , max_i_ka = 1.00408742467761, name='Line 3-23')
    #line23_36 =  pp.create_line_from_parameters(net_ieee39, bus23 , bus36 , length_km=1, r_ohm_per_km = 0.595125, x_ohm_per_km = 32.3748   , c_nf_per_km = 0.0                 , max_i_ka = 1.50613113701642, name='Line 23-24')
    line2_25  =  pp.create_line_from_parameters(net_ieee39, bus2  , bus25 , length_km=1, r_ohm_per_km = 8.33175 , x_ohm_per_km = 10.23615  , c_nf_per_km = 325.374524839553544 , max_i_ka = 0.83673952056468, name='Line 24-25')
    line25_26 =  pp.create_line_from_parameters(net_ieee39, bus25 , bus26 , length_km=1, r_ohm_per_km = 3.80880 , x_ohm_per_km = 38.445075 , c_nf_per_km = 1183.382689656184766, max_i_ka = 1.00408742467761, name='Line 6-26')
    line26_27 =  pp.create_line_from_parameters(net_ieee39, bus26 , bus27 , length_km=1, r_ohm_per_km = 1.66635 , x_ohm_per_km = 17.496675 , c_nf_per_km = 533.970795558609893 , max_i_ka = 1.00408742467761, name='Line 26-27')
    line26_28 =  pp.create_line_from_parameters(net_ieee39, bus26 , bus28 , length_km=1, r_ohm_per_km = 5.118075, x_ohm_per_km = 56.41785  , c_nf_per_km = 1738.747974519313402, max_i_ka = 1.00408742467761, name='Line 27-28')
    line26_29 =  pp.create_line_from_parameters(net_ieee39, bus26 , bus29 , length_km=1, r_ohm_per_km = 6.784425, x_ohm_per_km = 74.390625 , c_nf_per_km = 2293.221822328086546, max_i_ka = 1.00408742467761, name='Line 28-29')
    line28_29 =  pp.create_line_from_parameters(net_ieee39, bus28 , bus29 , length_km=1, r_ohm_per_km = 1.66635 , x_ohm_per_km = 17.972775 , c_nf_per_km = 554.919566335951004 , max_i_ka = 1.00408742467761, name='Line 28-29')
    line3_4   =  pp.create_line_from_parameters(net_ieee39, bus3  , bus4  , length_km=1, r_ohm_per_km = 1.547325, x_ohm_per_km = 25.352325 , c_nf_per_km = 493.410409585460172 , max_i_ka = 0.83673952056468, name='Line 29-30')
    line3_18  =  pp.create_line_from_parameters(net_ieee39, bus3  , bus18 , length_km=1, r_ohm_per_km = 1.309275, x_ohm_per_km = 15.830325 , c_nf_per_km = 476.473105552716163 , max_i_ka = 0.83673952056468, name='Line 30-31')
    line4_5   =  pp.create_line_from_parameters(net_ieee39, bus4  , bus5  , length_km=1, r_ohm_per_km = 0.9522  , x_ohm_per_km = 15.2352   , c_nf_per_km = 299.07713173608289  , max_i_ka = 1.00408742467761, name='Line 31-32')
    line4_14  =  pp.create_line_from_parameters(net_ieee39, bus4  , bus14 , length_km=1, r_ohm_per_km = 0.9522  , x_ohm_per_km = 15.354225 , c_nf_per_km = 307.991502279632243 , max_i_ka = 0.83673952056468, name='Line 32-33')
    line5_6   =  pp.create_line_from_parameters(net_ieee39, bus5  , bus6  , length_km=1, r_ohm_per_km = 0.23805 , x_ohm_per_km = 3.09465   , c_nf_per_km = 96.720920397511136  , max_i_ka = 2.00817484935522, name='Line 31-32')
    line5_8   =  pp.create_line_from_parameters(net_ieee39, bus5  , bus8  , length_km=1, r_ohm_per_km = 0.9522  , x_ohm_per_km = 13.3308   , c_nf_per_km = 328.94027305697341  , max_i_ka = 1.50613113701642, name='Line 32-33')
    
    trasfo2_30 = pp.create_transformer_from_parameters(net_ieee39, bus2, bus30, sn_mva=900,vn_hv_kv=345, vn_lv_kv=345, vkr_percent=0.0, max_loading_percent = 100,
                                          vk_percent=16.29, pfe_kw=0, i0_percent=0,shift_degree=0.0, name='Trafo 2-30')
    
    trasfo6_31 = pp.create_transformer_from_parameters(net_ieee39, bus6, bus31, sn_mva=1800,vn_hv_kv=345, vn_lv_kv=345, vkr_percent=0.0, max_loading_percent = 100,
                                          vk_percent=45, pfe_kw=0, i0_percent=0,shift_degree=0.0, name='Trafo 6-31')
    
    trasfo29_38 = pp.create_transformer_from_parameters(net_ieee39, bus29, bus38, sn_mva=1200, vn_hv_kv=345, vn_lv_kv=345, vkr_percent=0.96, max_loading_percent = 100,
                                          vk_percent=18.744599222175971, pfe_kw=0, i0_percent=0, shift_degree=0.0, name='Trafo 29-38')
    
    trasfo10_32 = pp.create_transformer_from_parameters(net_ieee39, bus10, bus32, sn_mva=900, vn_hv_kv=345, vn_lv_kv=345, vkr_percent=0.0, max_loading_percent = 100,
                                          vk_percent=18., pfe_kw=0, i0_percent=0, shift_degree=0.0, name='Trafo 10-32')
    
    trasfo12_11 = pp.create_transformer_from_parameters(net_ieee39, bus12, bus11, sn_mva=500, vn_hv_kv=345, vn_lv_kv=345, vkr_percent=0.8, max_loading_percent = 100,
                                          vk_percent=21.76470767090613, pfe_kw=0, i0_percent=0, shift_degree=0.0, name='Trafo 12-11')
    
    trasfo12_13 = pp.create_transformer_from_parameters(net_ieee39, bus12, bus13, sn_mva=500,vn_hv_kv=345, vn_lv_kv=345, vkr_percent=0.8, max_loading_percent = 100,
                                          vk_percent=21.76470767090613, pfe_kw=0, i0_percent=0, shift_degree=0.0, name='Trafo 12-13')
    
    trasfo19_20 = pp.create_transformer_from_parameters(net_ieee39, bus19, bus20, sn_mva=900, vn_hv_kv=345, vn_lv_kv=345, vkr_percent=0.63, max_loading_percent = 100,
                                          vk_percent=12.435967996099061, pfe_kw=0, i0_percent=0, shift_degree=0.0, name='Trafo 19-20')
    
    trasfo19_33 = pp.create_transformer_from_parameters(net_ieee39, bus19, bus33, sn_mva=900, vn_hv_kv=345, vn_lv_kv=345, vkr_percent=0.63, max_loading_percent = 100,
                                          vk_percent=12.79551874681132, pfe_kw=0, i0_percent=0, shift_degree=0.0, name='Trafo 2-30')
    
    trasfo20_34 = pp.create_transformer_from_parameters(net_ieee39, bus20, bus34, sn_mva=900, vn_hv_kv=345, vn_lv_kv=345, vkr_percent=0.81, max_loading_percent = 100,
                                          vk_percent=16.220237359545639, pfe_kw=0, i0_percent=0, shift_degree=0.0, name='Trafo 20-34')
    
    trasfo22_35 = pp.create_transformer_from_parameters(net_ieee39, bus22, bus35, sn_mva=900, vn_hv_kv=345, vn_lv_kv=345, vkr_percent=0.0, max_loading_percent = 100,
                                          vk_percent=12.869999999999999, pfe_kw=0, i0_percent=0, shift_degree=0.0, name='Trafo 22-35')
    
    trasfo23_36 = pp.create_transformer_from_parameters(net_ieee39, bus23, bus36, sn_mva=900, vn_hv_kv=345, vn_lv_kv=345, vkr_percent=0.0, max_loading_percent = 100,
                                          vk_percent=12.869999999999999, pfe_kw=0, i0_percent=0, shift_degree=0.0, name='Trafo 22-35')
    
    trasfo25_37 = pp.create_transformer_from_parameters(net_ieee39, bus25, bus37, sn_mva=900, vn_hv_kv=345, vn_lv_kv=345, vkr_percent=0.54, max_loading_percent = 100,
                                          vk_percent=20.88698159141239, pfe_kw=0, i0_percent=0, shift_degree=0.0, name='Trafo 25-37')
    # Loads
    pp.create_load(net_ieee39, bus3  , p_mw = 322   , q_mvar = 2.4  , scaling = LOAD[time], name='Load R3')
    pp.create_load(net_ieee39, bus4  , p_mw = 500   , q_mvar = 184  , scaling = LOAD[time], name='Load R4')
    pp.create_load(net_ieee39, bus7  , p_mw = 233.8 , q_mvar = 84   , scaling = LOAD[time], name='Load R7')
    pp.create_load(net_ieee39, bus8  , p_mw = 522   , q_mvar = 176  , scaling = LOAD[time], name='Load R8')
    pp.create_load(net_ieee39, bus12 , p_mw = 7.5   , q_mvar = 88  ,  scaling = LOAD[time], name='Load R12')
    pp.create_load(net_ieee39, bus15 , p_mw = 320   , q_mvar = 153  , scaling = LOAD[time], name='Load R15')
    pp.create_load(net_ieee39, bus16 , p_mw = 329   , q_mvar = 32.3 , scaling = LOAD[time], name='Load R16')
    pp.create_load(net_ieee39, bus18 , p_mw = 158   , q_mvar = 30   , scaling = LOAD[time], name='Load R18')
    pp.create_load(net_ieee39, bus20 , p_mw = 628  ,  q_mvar = 103  , scaling = LOAD[time], name='Load R20')
    pp.create_load(net_ieee39, bus21 , p_mw = 274   , q_mvar = 115  , scaling = LOAD[time], name='Load R21')
    pp.create_load(net_ieee39, bus23 , p_mw = 247.5 , q_mvar = 84.6 , scaling = LOAD[time], name='Load R23')
    pp.create_load(net_ieee39, bus24 , p_mw = 308.6 , q_mvar = -92  , scaling = LOAD[time], name='Load R24')
    pp.create_load(net_ieee39, bus25 , p_mw = 224   , q_mvar = 47.2 , scaling = LOAD[time], name='Load R25')
    pp.create_load(net_ieee39, bus26 , p_mw = 139   , q_mvar = 17   , scaling = LOAD[time], name='Load R26')
    pp.create_load(net_ieee39, bus27 , p_mw = 281   , q_mvar = 75.5 , scaling = LOAD[time], name='Load R27')
    pp.create_load(net_ieee39, bus28 , p_mw = 206  ,  q_mvar = 27.6 , scaling = LOAD[time], name='Load R28')
    pp.create_load(net_ieee39, bus29 , p_mw = 283.5 , q_mvar = 26.9 , scaling = LOAD[time], name='Load R29')
    pp.create_load(net_ieee39, bus31 , p_mw = 9.2   , q_mvar = 4.6  , scaling = LOAD[time], name='Load R31')
    pp.create_load(net_ieee39, bus39 , p_mw = 1104  , q_mvar = 250  , scaling = LOAD[time], name='Load R39')
    
    # Optional distributed energy recources
    n1 = PV[time]+(-PV[time]*0.1)
    n2 = PV[time]+(+PV[time]*0.1)
    PVN=(np.random.uniform(n1,n2))

    n3=PV[time]+(-PV[time]*0.2)
    n4=PV[time]+(+PV[time]*0.2)
    PVN_=(np.random.uniform(n3,n4))
    

    pp.create_sgen(net_ieee39, bus3  , 40*PVN , q_mvar=0,  name='PV 3', type='PV')
    pp.create_sgen(net_ieee39, bus8  , 40*PVN , q_mvar=0,  name='PV 8', type='PV')
    pp.create_sgen(net_ieee39, bus14 , 40*PVN , q_mvar=0,  name='PV 16', type='PV')
    pp.create_sgen(net_ieee39, bus25 , 40*PVN_, q_mvar=0,  name='PV 18', type='PV')
    pp.create_sgen(net_ieee39, bus30 , 40*PVN_, q_mvar=0,  name='PV 23', type='PV')
    pp.create_sgen(net_ieee39, bus31 , 40*PVN_, q_mvar=0,  name='PV 27', type='PV')


    #create generators        
    pp.create_gen(net_ieee39, bus30, p_mw = 250*LOAD[time]*0.1, min_p_mw= 0, max_p_mw = 250 , min_q_mvar = -58.084,  max_q_mvar = 153.18,   vm_pu = 1, controllable=True)
    #pp.create_gen(net_ieee39, bus31, p_mw = 522.728, q_mvar = 227.576, min_p_mw= 0, max_p_mw = 760 , min_q_mvar = -122.665, max_q_mvar = 429.804,  vm_pu = 1, controllable=True ,slack = True)
    pp.create_gen(net_ieee39, bus32, p_mw = 650*LOAD[time]*0.1, min_p_mw= 0, max_p_mw = 767 , min_q_mvar = -180.218, max_q_mvar = 446.722,  vm_pu = 1, controllable=True)
    pp.create_gen(net_ieee39, bus33, p_mw = 632*LOAD[time]*0.1, min_p_mw= 0, max_p_mw = 1068, min_q_mvar = -213.78,  max_q_mvar = 548.208,  vm_pu = 1, controllable=True)
    pp.create_gen(net_ieee39, bus34, p_mw = 508*LOAD[time]*0.1, min_p_mw= 0, max_p_mw = 982 , min_q_mvar = -188.028, max_q_mvar = 611.495,  vm_pu = 1, controllable=True)
    pp.create_gen(net_ieee39, bus35, p_mw = 650*LOAD[time]*0.1, min_p_mw= 0, max_p_mw = 987 , min_q_mvar = -234.972, max_q_mvar = 593.788,  vm_pu = 1, controllable=True)
    pp.create_gen(net_ieee39, bus36, p_mw = 560*LOAD[time]*0.1, min_p_mw= 0, max_p_mw = 932 , min_q_mvar = -249.132, max_q_mvar = 568.372,  vm_pu = 1, controllable=True)
    pp.create_gen(net_ieee39, bus37, p_mw = 540*LOAD[time]*0.1, min_p_mw= 0, max_p_mw = 882 , min_q_mvar = -216.122, max_q_mvar = 443.468,  vm_pu = 1, controllable=True)
    pp.create_gen(net_ieee39, bus38, p_mw = 830*LOAD[time]*0.1, min_p_mw= 0, max_p_mw = 1531, min_q_mvar = -356.889, max_q_mvar = 834.7751, vm_pu = 1, controllable=True)
    pp.create_gen(net_ieee39, bus39, p_mw =1000*LOAD[time]*0.1, min_p_mw= 0, max_p_mw = 1090, min_q_mvar = -173.261, max_q_mvar = 574.85,   vm_pu = 1, controllable=True)
    pp.create_ext_grid(net_ieee39, bus31 , max_p_mw=1000, min_p_mw=-1000)

    # 발전 비용 곡선

    pp.create_pwl_cost(net_ieee39, 0, "gen", [[0, 100, 557.41], [100, 250, 1704.91]])
    pp.create_pwl_cost(net_ieee39, 1, "gen", [[0, 250, 1244], [250, 767, 6093]])
    pp.create_pwl_cost(net_ieee39, 2, "gen", [[0, 350, 884.2], [350, 1068, 5674]])
    pp.create_pwl_cost(net_ieee39, 3, "gen", [[0, 330, 1977], [330, 982, 9492]])
    pp.create_pwl_cost(net_ieee39, 4, "gen", [[0, 330, 484.7], [330, 987, 2277]])
    pp.create_pwl_cost(net_ieee39, 5, "gen", [[0, 310, 1508], [310, 932, 5992]])
    pp.create_pwl_cost(net_ieee39, 6, "gen", [[0, 300, 1323], [300, 882, 5692]])
    pp.create_pwl_cost(net_ieee39, 7, "gen", [[0, 500, 1775], [500, 1531, 11043.5]])
    pp.create_pwl_cost(net_ieee39, 8, "gen", [[0, 360, 1882], [360, 1090, 8471]])
    pp.create_pwl_cost(net_ieee39, 0, "ext_grid", [[0, 250, 2851.6], [250, 760, 6330]])

    '''
    pp.create_pwl_cost(net_ieee39, bus30, "gen", [[0, 100, 557.41], [100, 250, 1704.91]])
    pp.create_pwl_cost(net_ieee39, bus31, "ext_grid", [[0, 250, 851.6], [250, 760, 4330]])
    pp.create_pwl_cost(net_ieee39, bus32, "gen", [[0, 250, 1244], [250, 767, 6093]])
    pp.create_pwl_cost(net_ieee39, bus33, "gen", [[0, 350, 884.2], [350, 1068, 5674]])
    pp.create_pwl_cost(net_ieee39, bus34, "gen", [[0, 330, 1977], [330, 982, 9492]])
    pp.create_pwl_cost(net_ieee39, bus35, "gen", [[0, 330, 484.7], [330, 987, 2277]])
    pp.create_pwl_cost(net_ieee39, bus36, "gen", [[0, 310, 1508], [310, 932, 5992]])
    pp.create_pwl_cost(net_ieee39, bus37, "gen", [[0, 300, 1323], [300, 882, 5692]])
    pp.create_pwl_cost(net_ieee39, bus38, "gen", [[0, 500, 1775], [500, 1531, 11043.5]])
    pp.create_pwl_cost(net_ieee39, bus39, "gen", [[0, 360, 1882], [360, 1090, 8471]])    '''
    pp.create_storage(net_ieee39, bus18, p_mw = 0, max_e_mwh = 2.5, soc_percent = 0.5, min_e_mwh = 0)


    pp.runopp(net_ieee39, delta=1e-3)

    busv_pu = np.array([net_ieee39.res_bus.vm_pu])
    line_percent = np.array([net_ieee39.res_line.p_to_mw]) 
    soc = np.array([net_ieee39.storage.soc_percent])
    p_mw = np.array([net_ieee39.storage.p_mw])


    #dcline_power = net_ieee39.dcline.p_mw
    cost = net_ieee39.res_cost
    gen = np.array(net_ieee39.res_gen)
    
    print(net_ieee39.res_bus.vm_pu)
    print(net_ieee39.res_load)
    print(net_ieee39.res_line.p_to_mw)
    #print('dcline',net_ieee39.dcline.p_mw)
    print('res_cost',cost)
    print(net_ieee39.res_gen)
    print(net_ieee39.res_ext_grid)
    
for i in range(24):
    opf(i)