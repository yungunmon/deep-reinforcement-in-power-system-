import numpy as np

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

def resetAgentPos(Radius):
    pos= np.load('agent.npy')
    x = pos[:,0]
    y = pos[:,1]
    return x, y

def resetUserPos(Radius):
    pos= np.load('user.npy')
    x = pos[:,0]
    y = pos[:,1]
    return x, y

def resetDronePos(numDrone, Radius):
    pos= np.load('user.npy')
    X = pos[:,0]
    Y = pos[:,1]
    return X, Y
