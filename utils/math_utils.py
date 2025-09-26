import numpy as np


def Rx(theta):
    """Rotation matrix around x-axis by angle theta (in radians)"""
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)]
    ])

def Ry(theta):
    """Rotation matrix around y-axis by angle theta (in radians)"""
    return np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [ 0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def Rz(theta):
    """Rotation matrix around z-axis by angle theta (in radians)"""
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])