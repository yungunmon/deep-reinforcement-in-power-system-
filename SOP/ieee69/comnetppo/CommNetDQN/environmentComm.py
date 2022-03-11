import numpy as np
import math

from matplotlib import pyplot as plt
from matplotlib import colors
from env_utils import clamp, resetAgentPos, resetUserPos, resetDronePos

import io
import PIL.Image
from torchvision.transforms import ToTensor
EP_LEN      =   40
N_USER      =   25
N_COMMAGENT =    4
N_DQNAGENT  =    0
N_DRONE     =    3
GRID        = 1200
QUALITY     =  np.array([2,3,6])
#  np.array([2, 3, 4, 6, 12]) # Coverage: 360p, 480p, 720p, 1080p, 2160p  
RADMAX      = len(QUALITY) - 1
RADIUS      = 1000   
COVERAGE    = 1 / QUALITY * RADIUS
VELOCITY    = 333 # 20 [km/h] -> 333.3 [m/min]
O_THRESHOLD = 0.6 

font = {'size'   :10}
plt.rc('font', **font)  # pass in the font dict as kwargs
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
class User:
    
    def __init__(self, id = None,x=None,y=None):
    
        self.id         = id
        self.x          = x
        self.y          = y
        self.connect    = {'connection': 0,
                           'quality': -1,
                           'drone':   -1}
    
    def reset_connection(self):
        
        self.connect    = {'connection': 0,
                           'quality': -1,
                           'drone':   -1}
        
    def is_support(self, agentList, droneList):
        
        self.reset_connection()
        
        for drone in droneList:
            if drone.isWorking:
                distance     = np.linalg.norm((self.x - drone.x , self.y, drone.y))
                drone_radius = COVERAGE[drone.radius]
                drone.isWorking
                if distance < drone_radius:

                    self.connect['connection'] = 1

                    if self.connect['quality'] < drone.quality:

                        self.connect['quality'] = drone.quality
                        self.connect['drone']   = drone.id
                        
        for drone in agentList:
            if drone.isWorking:
                distance     = np.linalg.norm((self.x - drone.x , self.y, drone.y))
                drone_radius = COVERAGE[drone.radius]
                drone.isWorking
                if distance < drone_radius:

                    self.connect['connection'] = 1

                    if self.connect['quality'] < drone.quality:

                        self.connect['quality'] = drone.quality
                        self.connect['drone']   = drone.id
                        
                        
        return [self.connect['connection'], self.connect['drone'], self.connect['quality']]

class Drone:
    
    def __init__(self, id = None, isAgent = False, radMax = 4, x = None, y = None):
        
        self.id = id
        self.isAgent = isAgent
        self.isWorking = 1
        self.batteryRemain = 1
        self.radMax = radMax
        self.x = x
        self.y = y
        if self.isAgent: self.radius = self.radMax
        else           : self.radius = 0
        self.coverage     = COVERAGE[self.radius]
        self.quality      = QUALITY[self.radius]
        self.avail_action = self._avail_action()
    def batterConsump(self, batteryConstant = 0.01, constant = 0.02):
        
        self.batteryRemain -= self.radius * batteryConstant + constant
        return self.radius * batteryConstant + constant

    def randomDamage(self): 
        
        if np.random.rand() < 0.03:
            self.isWorking  = 0
                
    def _avail_action(self):
        
        avail_action = np.ones(7)
        if self.x <= -GRID:
            avail_action[2] = 0
        if self.x >= GRID:
            avail_action[1] = 0
        if self.y <= -GRID:
            avail_action[4] = 0
        if self.y >= GRID:
            avail_action[3] = 0
        if self.radius == 0:
            avail_action[6] = 0
        if self.radius == self.radMax:
            avail_action[5] = 0
        return avail_action
            
    def transition(self,action):
        
        if action == 0: pass # 아무 액션도 하지 않음
        elif action == 1: self.x += VELOCITY
        elif action == 2: self.x -= VELOCITY
        elif action == 3: self.y += VELOCITY
        elif action == 4: self.y -= VELOCITY
        elif action == 5: self.radius += 1
        elif action == 6: self.radius -= 1
        
        # state space 맞추어주기.
        self.x = clamp(self.x, -GRID, GRID)
        self.y = clamp(self.y, -GRID, GRID)
        self.radius = clamp(self.radius, 0 , self.radMax)
        self.radius = int(self.radius)
        self.coverage = COVERAGE[self.radius]
        self.quality  = QUALITY[self.radius]
        
    def _step(self, action):
        
        self.transition(action)
        self.batterConsump()
        if self.isAgent == False: 
            self.randomDamage()
        else:
            self._avail_action()


class Utility:
    
    def __init__(self):
        
        self.T_resolution    = 0
        self.T_support_rate  = 0
        self.T_support_area  = 0
        self.T_support_user  = 0
        self.T_overlapped    = 0
        self.I_resolution    = np.zeros(N_COMMAGENT+N_DQNAGENT)
        self.I_numuser       = np.zeros(N_COMMAGENT+N_DQNAGENT)
        self.I_battery       = np.zeros(N_COMMAGENT+N_DQNAGENT)
        self.SUPPORT         = np.zeros(N_COMMAGENT+N_DQNAGENT+2)
        self.overlapped_thres= 0.3
        self.grid     = 60
        self.scale    = self.grid /GRID
        self.coverage = COVERAGE * self.scale
        self.location_pos        = np.array([[-1000,1000],[0, 1000],[1000, 1000],[-1000, 0],[0, 0],[1000, 0],[-1000,-1000],[0, -1000],[1000, -1000]]) 
        self.location_duplicated = np.zeros(9)
        
    def get_utils_info(self):

        return self.T_support_rate, self.T_support_area, self.T_resolution, self.T_overlapped
    
    def _calcuate_total_support(self,  userList, agentList, droneList):
        
        Qmap    = np.zeros((self.grid,self.grid)) 
        Smap    = np.zeros((self.grid,self.grid)).astype(bool)
        Surface = np.pi * 0 **2 
        UAVList = []
        UAVList.extend(droneList)
        UAVList.extend(agentList)
        
        for drone in UAVList:
            if drone.isWorking:
                X = np.arange(self.grid)
                Y = np.arange(self.grid)
                X, Y = np.meshgrid(X,Y)
                d_x = drone.x * self.scale 
                d_y = drone.y * self.scale
                d_r = self.coverage[drone.radius]
                s   = np.pi* d_r**2
                l_map =  (X-d_x)**2 + (Y-d_y)**2 <= s
                q_map = l_map * drone.quality

                Qmap = (Qmap >= q_map) * Qmap +  (q_map >= Qmap) * q_map
                Smap = Smap + l_map
                Surface += s
        TotalSurface = (Smap * 1).sum()
        
        if TotalSurface > 0:
            self.T_support_rate  = TotalSurface /  ((2*self.grid) ** 2)
            self.T_support_area  = TotalSurface / (self.scale) ** 2 
            self.T_resolution    = Qmap.sum()   / TotalSurface        # Average Resolution
            self.T_overlapped    = (Surface - TotalSurface) / Surface
        else: 
            self.T_support_rate  = 0
            self.T_support_area  = 0
            self.T_resolution    = 0
            self.T_overlapped    = 1
            
    def _calculate_individual_support(self, userList, agentList, droneList):
        for user in userList:
            isConnect, droneId, quality = user.is_support(agentList, droneList) 
            self.T_support_user += isConnect
            
            if isConnect:
            
                if droneId < N_COMMAGENT+N_DQNAGENT:
                    self.I_resolution[droneId]   += np.log(quality)
                    self.I_numuser[droneId]      += 1
    
        self.I_resolution    = self.I_resolution
        # self.I_numuser       = self.I_numuser # / N_USER * (N_COMMAGENT + N_DQNAGENT + N_DRONE) # 모두 서포트 가능하다고 가정했을 때, (최대 서포트 수 / 드론 개수)
        
    def _calculate_energy_consumption(self,agentList):
        
        for drone in agentList:
            self.I_battery[drone.id]  = drone.batteryRemain 
            
    def calculate_reward(self, userList, agentList, droneList):
        
        self.__init__() # 초기화 먼저 해줌.
        self._calcuate_total_support(userList, agentList, droneList)
        self._calculate_individual_support(userList, agentList, droneList)
        self._calculate_energy_consumption(agentList)
        self.SUPPORT[0] = self.T_support_user
        self.SUPPORT[1:5] = np.copy(self.I_numuser)
        self.SUPPORT[5] = self.T_support_user - self.I_numuser.sum()
        self.T_support_rate    =  self.T_support_user / N_USER
        Common_Reward          =  self.T_support_rate
        Individual_Reward      =  self.I_resolution
        Reward                 =  (Common_Reward  > 0.4) * Individual_Reward
        return Reward
    
class Environment:
 
    def __init__(self):
        
        # [1] Environment 
        self.EPLEN        = EP_LEN
        self.numUser      = N_USER
        self.numCommAgent = N_COMMAGENT
        self.numDQNAgent  = N_DQNAGENT
        self.numAgent     = self.numCommAgent + self.numDQNAgent
        self.numDrone     = N_DRONE
        self.GridRadius   = GRID
        self.AgentRadius  = self.GridRadius * 1 / 3
        self.RadMax       = RADMAX
        self.DroneRadius  = self.GridRadius * 1 / 2
        
        # [2] Environment --> RLAgent 
        self.common_obs        = None
        self.common_obs_prime  = None
        self.o            = None
        self.s            = None
        self.o_prime      = None
        self.s_prime      = None
        self.t            = 1
        
        # [3] Environment Initialization
        self.Initialize()
        
    def reset(self):
        self.Initialize()
    
    def get_available_action(self):
        available_action = []
        for i in range(self.numAgent):
            available_action.append(self.agentList[i]._avail_action())
        return np.array(available_action)
    
    def Initialize(self):
        self._initUser()
        self._initAgent()
        self._initDrone()
        pos= np.load('agent.npy')
        for i in range(self.numAgent):
            self.agentList[i].x = pos[i,0]
            self.agentList[i].y = pos[i,1]
        pos= np.load('user.npy')
        for i in range(self.numUser):
            self.userList[i].x = pos[i,0] * 4 / 7
            self.userList[i].y = pos[i,1] * 4 / 7
        pos= np.load('drone.npy')
        for i in range(self.numDrone):
            self.droneList[i].x = pos[i,0]
            self.droneList[i].y = pos[i,1]

        self.utility = Utility()
        self.common_obs = self.getCommonObs()
        self.common_obs_prime = np.copy(self.common_obs)
        self.o = self.getagentObs()
        self.o_prime = np.copy(self.o)
        self.s = self.getState()
        self.s_prime = np.copy(self.s)
        
        self.ava = self.get_available_action()
        self.ava_prime = self.get_available_action()
        
    
    def get_info(self):
        info = dict()
        info['episode_limit'] = self.EPLEN
        info['n_Comm_agents'] = self.numCommAgent
        info['n_DQN_agents']  = self.numDQNAgent
        info['n_agents']      = self.numCommAgent + self.numDQNAgent
        info['n_actions']     = 7
        info['state_dim']     = self.s.shape[-1]
        info['obs_dim']       = self.o.shape[-1]
        return info
    
    def _initUser(self):
        self.userList = []
        for i in range(self.numUser):
            x,y = resetUserPos(self.GridRadius)
            self.userList.append(User(id = i,
                                      x  = x,
                                      y  = y))

    def _initAgent(self):
        self.agentList = []
        for i in range(self.numAgent):
            x, y = resetAgentPos(self.AgentRadius)
            self.agentList.append(Drone(id      = i,
                                        x       = x,
                                        y       = y,
                                        isAgent = True,
                                        radMax  = self.RadMax))

    def _initDrone(self):
        self.droneList = []
        X, Y = resetDronePos(self.numDrone, self.DroneRadius)
        for i in range(self.numDrone):
            self.droneList.append(Drone(id = self.numAgent+ i,
                                        x = X[i],
                                        y = Y[i],
                                        isAgent = False,
                                        radMax = self.RadMax))

    def plot(self):
        colorlist = ['y','b','k','m','r']
        fig, ax = plt.subplots(figsize=(4,4),dpi=150)
        UX, UY = [],[]
        DX, DY, DR, DC = [],[],[],[]
        AX, AY, AR, AC = [],[],[],[]
        CX, CY, CR, CC = [],[],[],[]
        for i in range(self.numUser):
            UX.append(self.userList[i].x)
            UY.append(self.userList[i].y)

        for i in range(self.numDrone):
            if self.droneList[i].isWorking:
                DX.append(self.droneList[i].x)
                DY.append(self.droneList[i].y)
                DR.append(COVERAGE[self.droneList[i].radius])
                DC.append(colorlist[self.droneList[i].radius])
                
        for i in range(N_COMMAGENT):
            CX.append(self.agentList[i].x)
            CY.append(self.agentList[i].y)
            CR.append(COVERAGE[self.agentList[i].radius])
            CC.append(colorlist[self.agentList[i].radius])
            
        for i in range(N_COMMAGENT,self.numAgent):
            AX.append(self.agentList[i].x)
            AY.append(self.agentList[i].y)
            AR.append(COVERAGE[self.agentList[i].radius])
            AC.append(colorlist[self.agentList[i].radius])
        ax.scatter(UX, UY, s  = 30, c = 'r', marker = 's')
        if len(DX):
            ax.scatter(DX, DY, s  = 90, c = 'g', marker = 'o')
            for i in range(len(DX)):
                a_circle = plt.Circle((DX[i], DY[i]), DR[i], fill=False, color=DC[i],linewidth=2.5)
                ax.add_artist(a_circle)

        if len(CX):
            ax.scatter(CX, CY, s  = 90, c = 'purple', marker = '*')
            for i in range(N_COMMAGENT):
                a_circle = plt.Circle((CX[i], CY[i]), CR[i], fill=False, color=CC[i],linewidth=2.5)            
                ax.add_artist(a_circle)
                
        ax.scatter(AX, AY, s  = 90, c = 'b', marker = '*')
        
        for i in range(self.numAgent-N_COMMAGENT):
            a_circle = plt.Circle((AX[i], AY[i]), AR[i], fill=False, color=AC[i],linewidth=2.5)
            ax.add_artist(a_circle)
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        plt.xlim([-self.GridRadius, self.GridRadius])
        plt.ylim([-self.GridRadius, self.GridRadius])
        plt.draw()
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = PIL.Image.open(buf)
        plt.close()
        image = ToTensor()(image).unsqueeze(0)[0]
        return image

    def getCommonObs(self):
        
        # [1] 모든 요소에 대한 절대 위치         
        agent_pos = np.array([  [self.agentList[i].x,self.agentList[i].y]  for i in range(self.numAgent)]).flatten()  
        user_pos  = np.array([  [self.userList[u].x, self.userList[u].y ]  for u in range(self.numUser)]).flatten()  
        droneisWorking = np.array([self.droneList[k].isWorking for k in range(self.numDrone)]).flatten()  
        drone_pos = np.array([  [self.droneList[k].x, self.droneList[k].y] for k in range(self.numDrone)]).flatten() * np.hstack([droneisWorking, droneisWorking])
        Position  = np.hstack([agent_pos,user_pos,drone_pos])
        ## Normalize
        Position  = Position / self.GridRadius - 1
        
        # [2] 현재 작동하는지 정보 / Connection 정보
        # agentisWorking = np.array([self.agentList[i].isWorking for i in range(self.numAgent)]).flatten()
        userisConnect  = np.array([[user.connect['connection'],                              # 0 or 1
                                   user.connect['drone'] / (self.numAgent + self.numDrone), # 0 <= drone_id <=1
                                   user.connect['quality'] / QUALITY[-1]] \
                                  for user in self.userList]).flatten()        # 0 <= quality <= 1
        # IsWorking      = np.hstack([agentisWorking, userisConnect, droneisWorking])
        IsWorking      = np.hstack([userisConnect, droneisWorking])
        
        # [3] Coverage 정보
        agent_coverage = np.array([self.agentList[i].radius  for i in range(self.numAgent)]).flatten()
        drone_coverage = np.array([self.droneList[k].radius  for k in range(self.numDrone)]).flatten() * droneisWorking
        Coverage       = np.hstack([agent_coverage, drone_coverage])
        ## Normalize
        Coverage       = Coverage / self.RadMax
        
        # [4] Average support 정보, resolution 정보
        resolution    = self.utility.T_resolution 
        coverage_rate = self.utility.T_support_rate 
        support_rate  = self.utility.T_overlapped 
        Util          = np.array([resolution, coverage_rate, support_rate])
        Util          = np.append(Util, np.array([self.utility.I_resolution, self.utility.I_numuser]).flatten())
        
        # 모든 Observation을 concatenate
        CommonObs     = np.hstack([Position, IsWorking, Coverage, Util])
        return CommonObs
    
    def getState(self):
        
        state      = self.common_obs
        state      = np.append(state, [self.agentList[i].batteryRemain for i in range(self.numAgent)])
        for i in range(self.numAgent):
            for k in range(self.numAgent):
                x_diff = self.agentList[i].x - self.agentList[k].x 
                y_diff = self.agentList[i].y - self.agentList[k].y
                x_diff = x_diff / self.GridRadius - 1
                y_diff = y_diff / self.GridRadius - 1
                try:
                    dist = np.linalg.norm((x_diff,y_diff))
                except:
                    dist = 0
                state = np.append(state, [x_diff, y_diff, dist])
            for u in range(self.numUser):
                x_diff = self.agentList[i].x - self.userList[u].x
                y_diff = self.agentList[i].y - self.userList[u].y
                x_diff = x_diff / self.GridRadius - 1
                y_diff = y_diff / self.GridRadius - 1
                try:
                    dist = np.linalg.norm((x_diff,y_diff))
                except:
                    dist = 0
                state = np.append(state, [x_diff, y_diff, dist])
                
            for j in range(self.numDrone):
                if self.droneList[j].isWorking:
                    x_diff = self.agentList[i].x - self.droneList[j].x
                    y_diff = self.agentList[i].y - self.droneList[j].y
                    x_diff = x_diff / self.GridRadius - 1
                    y_diff = y_diff / self.GridRadius - 1
                    try:
                        dist = np.linalg.norm((x_diff,y_diff))
                    except:
                        dist = 0
                    state = np.append(state, [x_diff, y_diff, dist])
                else:
                    state = np.append(state, [0, 0, 0])
        return state
    
    def getagentObs(self):
        
        Obs = []
        obs_common = self.common_obs
        for i in range(self.numAgent):
            o_partial = []
            o_partial.append(self.agentList[i].x / self.GridRadius - 1)
            o_partial.append(self.agentList[i].y / self.GridRadius - 1)
            o_partial.append(self.agentList[i].batteryRemain)

            for k in range(self.numAgent):
                x_diff = self.agentList[i].x - self.agentList[k].x
                y_diff = self.agentList[i].y - self.agentList[k].y
                try:
                    o_partial.append(np.linalg.norm((x_diff,y_diff)) / self.GridRadius - 1)
                except:
                    o_partial.append(0)
                    
                
            for u in range(self.numUser):
                x_diff = self.agentList[i].x - self.userList[u].x
                y_diff = self.agentList[i].y - self.userList[u].y
                try:
                    o_partial.append(np.linalg.norm((x_diff,y_diff))  / self.GridRadius - 1)
                except:
                    o_partial.append(-1)
                    
            for j in range(self.numDrone):
                if self.droneList[j].isWorking:
                    x_diff = self.agentList[i].x - self.droneList[j].x
                    y_diff = self.agentList[i].y - self.droneList[j].y
                    try:
                        o_partial.append(np.linalg.norm((x_diff,y_diff))  / self.GridRadius - 1)
                    except:
                        o_partial.append(-1)
                else:
                    o_partial.append(0)
                
            o_partial = np.array(o_partial)
            o_idx = np.zeros(self.numAgent)
            o_idx[self.agentList[i].id] += 1 
            o = np.hstack([o_idx, obs_common, o_partial])
            
            Obs.append(np.copy(o))
        Obs = np.array(Obs)
        return Obs


    def step(self, actions):
        self.s = np.copy(self.s_prime)
        self.o = np.copy(self.o_prime)
        self.ava = self.get_available_action()
        for i in range(self.numAgent):
            self.agentList[i].transition(actions[i])
        for i in range(self.numDrone):
            self.droneList[i].randomDamage()
        self.getCommonObs()
        self.s_prime = self.getState()
        self.o_prime = self.getagentObs()
        
        rewards = self.utility.calculate_reward(self.userList, self.agentList, self.droneList) # 'userList', 'agentList', and 'droneList'
        self.t += 1
        
        self.ava_prime = self.get_available_action()
        if self.t >= self.EPLEN : done = False 
        else                    : done = True
        return self.o, self.s, self.ava, rewards, self.o_prime, self.s_prime, self.ava_prime, done
