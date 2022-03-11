import numpy as np

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

def resetAgentPos(Radius):
    Radius = np.random.uniform(low = 0, high = Radius/3)
    theta=np.random.uniform(0, 2 * np.pi)
    x = Radius * np.cos(theta)
    y = Radius * np.sin(theta)
    return x, y

def resetUserPos(Radius):
    Radius = np.random.uniform(low = 0, high = Radius)
    theta=np.random.uniform(0, 2 * np.pi)
    x = Radius * np.cos(theta)
    y = Radius * np.sin(theta)
    return x, y

def resetDronePos(numDrone, Radius):
    Theta   =  np.linspace(0,2 * np.pi, numDrone+1) + np.random.uniform(0, 2 * np.pi / (numDrone+1))
    X       =  Radius * np.cos(Theta)
    Y       =  Radius * np.sin(Theta)
    return X, Y
