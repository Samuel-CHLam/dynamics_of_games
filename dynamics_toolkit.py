import numpy as np
import matplotlib.pyplot as plt
from torch import tensor, dot, matmul, zeros, log
from scipy.special import xlogy

# constructing function for replicator dynamics
class oneD_replicator:
    def __init__(self, payoff):
        # payoff represents payoff matrices
        self.payoff = payoff
        self.size = payoff.shape[0]
        
    def replicator(self,x,t=None):
        return x * (self.payoff @ x - np.transpose(x) @ (self.payoff @ x))
    
    def replicator_tensor(self,x_raw,t=None, numpy_output=False):
        '''A tensor version of replicator function.'''
        A = tensor(self.payoff).double()
        x = x_raw.double()
        output = x*(matmul(A,x) - dot(x,matmul(A,x)))
        if numpy_output:
            return output.numpy()
        else:
            return output

class twoD_replicator:
    def __init__(self, payoff1, payoff2, mode=0, alpha1=0, alpha2=0):
        self.payoff1 = payoff1

        # mode represents what representation we are using. mode = 0 represents 1st representation, and mode = 1 represents 2nd representation
        if mode:
            self.payoff2 = payoff2.T
        else:
            self.payoff2 = payoff2
        
        self.shape = payoff1.shape

        # payoff represents the matrix [0, A; B, 0]
        self.payoff = np.block([[np.zeros((payoff1.shape[0],payoff1.shape[0])), payoff1], [payoff2, np.zeros((payoff1.shape[1], payoff1.shape[1]))]])

        # memories for replicator dynamics, in between 0 to 1
        self.memory1 = alpha1
        self.memory2 = alpha2

    def replicator(self,x,t=None):
        n = self.shape[0]
        m = self.shape[1]
        dx = np.zeros(m+n)
        dx[:n] = x[:n] * (self.payoff1@x[n:] - np.dot(x[:n], self.payoff1@x[n:]))
        dx[n:] = x[n:] * (self.payoff2@x[:n] - np.dot(x[n:], self.payoff2@x[:n]))
        return dx
    
    def adaptive(self,x,t=None):
        n = self.shape[0]
        m = self.shape[1]
        dx = np.zeros(m+n)
        dx[:n] = x[:n] * (self.payoff1@x[n:] - np.dot(x[:n], self.payoff1@x[n:]) + self.memory1 * (-np.log(x[:n]) + np.sum(xlogy(x[:n], x[:n]))))
        dx[n:] = x[n:] * (self.payoff2@x[:n] - np.dot(x[n:], self.payoff2@x[:n]) + self.memory2 * (-np.log(x[n:]) + np.sum(xlogy(x[n:], x[n:]))))
        return dx

    def replicator_tensor(self,x_raw,t=None):
        n = self.shape[0]
        m = self.shape[1]
        A = tensor(self.payoff1).double()
        B = tensor(self.payoff2).double()
        x = x_raw.double()
        dx = zeros(m+n)
        dx[:n] = x[:n] * (matmul(A, x[n:]) - dot(x[:n], matmul(A, x[n:])))
        dx[n:] = x[n:] * (matmul(B, x[:n]) - dot(x[n:], matmul(B, x[:n])))
        return dx
    
    def adaptive_tensor(self,x_raw,t=None):
        n = self.shape[0]
        m = self.shape[1]
        A = tensor(self.payoff1).double()
        B = tensor(self.payoff2).double()
        x = x_raw.double()
        dx = zeros(m+n)
        dx[:n] = x[:n] * (matmul(A, x[n:]) - dot(x[:n], matmul(A, x[n:])) + self.memory1 * (-log(x[:n]) + dot(x[:n], log(x[:n]))))
        dx[n:] = x[n:] * (matmul(B, x[:n]) - dot(x[n:], matmul(B, x[:n])) + self.memory2 * (-log(x[n:]) + dot(x[n:], log(x[n:]))))
        return dx

    

# Drawings

def draw_line(start, end, points=1000):
    '''
    Return array of points on a line segemnt.
    
    Parameters
    ----------
    start : list / array_like
        starting point
    end : list / array_like
        end point
    points : int, optional
        number of points included in the array, including start and end points
        
        
    Return
    ------
    array of point on the line, with row index being coordinate index and column index being index of points.
    '''
    
    # ts = 0 represents starting point, ts = 1 represents end point.
    ts = np.array([np.linspace(0,1,points)])
    
    start_reshaped = np.array(start).reshape((len(start),1))
    end_reshaped = np.array(end).reshape((len(end),1))
    
    return start_reshaped + ((end_reshaped - start_reshaped) @ ts)

def proj_3D_simplex(array_to_be_transformed):
    proj_mat = np.array(
    [[-1 * np.cos(30. / 360. * 2. * np.pi),np.cos(30. / 360. * 2. * np.pi), 0.],
     [-1 * np.sin(30. / 360. * 2. * np.pi),-1 * np.sin(30. / 360. * 2. * np.pi), 1.]])
    return proj_mat @ array_to_be_transformed

def proj_6D_2D(array_to_be_transformed):
    proj_mat = np.array(
    [[3.65, -1.35, 1.35, 5.35, 1.35, 1.45],
     [0.4, 0.4, 4.6, 1.9, -0.4, 4.4]])
    return proj_mat @ array_to_be_transformed

def initial_2D_simplex_figure(vertex_label = True):
    fig, ax = plt.subplots()
    ax.axis("off")

    PBd21 = proj_3D_simplex(draw_line([0,1,0], [1,0,0], 2))
    PBd32 = proj_3D_simplex(draw_line([0,0,1], [0,1,0], 2))
    PBd31 = proj_3D_simplex(draw_line([0,0,1], [1,0,0], 2))
    
    ax.plot(PBd21[0], PBd21[1], color='black', linewidth=3)
    ax.plot(PBd32[0], PBd32[1], color='black', linewidth=3)
    ax.plot(PBd31[0], PBd31[1], color='black', linewidth=3)
    
    if vertex_label:
        ax.text(-0.8660254-0.1, -0.5 +0.05 , "$e_1$",fontsize=12)
        ax.text(+0.8660254+0.05, -0.5 +0.05 , "$e_2$",fontsize=12)
        ax.text(0-0.03, 1 +0.1 , "$e_3$",fontsize=12)
    return fig, ax

def initial_6D_2D_figure(figsize=(12,12)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    oneD_vertices_list = [[1,0,0], [0,1,0], [0,0,1]]
    edges = [(0,1),(1,1),(1,2),(1,0),(2,0),(0,0),(0,1),(2,1),(2,0),(2,2),(1,2),(0,2),(2,2),(2,1),(1,1),(1,0),(0,0),(0,2),(0,1)]

    for i in range(len(edges)-1):
        start = oneD_vertices_list[edges[i][0]] + oneD_vertices_list[edges[i][1]]
        end = oneD_vertices_list[edges[i+1][0]] + oneD_vertices_list[edges[i+1][1]]
        proj_Bd = proj_6D_2D(draw_line(start, end, 2))
        ax.plot(proj_Bd[0], proj_Bd[1], color='black', linewidth=1)
    
    # if vertex_label:
    #     ax.text(-0.8660254-0.1, -0.5 +0.05 , "$e_1$",fontsize=12)
    #     ax.text(+0.8660254+0.05, -0.5 +0.05 , "$e_2$",fontsize=12)
    #     ax.text(0-0.03, 1 +0.1 , "$e_3$",fontsize=12)

    return fig, ax

# Misc

def simulate_close(original, max_diff, state=None):
    # max_diff represents maximum difference to first four coordinates
    
    if state:
        np.random.seed(state)
    
    noise = np.random.rand(len(original))
    noise = noise - np.mean(noise)
    return original + max_diff * noise

def localmax(arr):
    return arr[1:-1][(arr[2:] < arr[1:-1]) * (arr[1:-1] >= arr[:-2])]