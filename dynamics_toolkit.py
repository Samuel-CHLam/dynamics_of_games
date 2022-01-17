import numpy as np
import matplotlib.pyplot as plt
from torch import tensor, dot, matmul, zeros, log
from scipy.special import xlogy

# constructing function for replicator dynamics
class oneD_replicator:
    '''
    Class defining single-population games.

    Parameters
    ----------
    payoff : ndarray
        Payoff matrix.

    Attributes
    size : int
        Size of payoff matrix, representing the number of options for games.
    ----------
    '''
    def __init__(self, payoff):
        # payoff represents payoff matrices
        self.payoff = payoff
        self.size = payoff.shape[0]
        
    def replicator(self,x,t=None):
        '''
        The right hand side of single-population replicator dynamics.

        Parameters
        ----------
        x : ndarray
            Choice distribution vector.
        t : float, optional
            Time variable for running numerical simulation.

        Return
        ------
        ndarray

        See Also
        --------
        oneD_replicator.replicator_tensor : single-population replicator equation for tensors.
        '''
        return x * (self.payoff @ x - np.transpose(x) @ (self.payoff @ x))
    
    def replicator_tensor(self,x_raw,t=None, numpy_output=False):
        '''
        A tensor version of single-population replicator dynamics.

        Parameters
        ----------
        x_raw : Tensor
            Choice distribution tensor.
        t : float, optional
            Time variable for running numerical simulation.
        numpy_output : bool, optional
            If `true` return as an ndarray, if `false` return as a Tensor.

        Return
        ------
        ndarray if `numpy_output=True`, otherwise Tensor.

        See Also
        --------
        oneD_replicator.replicator : single-population replicator equation for numpy.ndarrays.
        '''
        A = tensor(self.payoff).double()
        x = x_raw.double()
        output = x*(matmul(A,x) - dot(x,matmul(A,x)))
        if numpy_output:
            return output.numpy()
        else:
            return output

class twoD_replicator:
    '''
    Class defining two-population games.

    Parameters
    ----------
    payoff1 : ndarray
        Payoff matrix for player 1.
    payoff2 : ndarray
        Payoff matrix for player 2.
    convention : bool
        If `True` then we use the 2nd convention for computation, if `False` then we use 1st convention. The conventions are defined in [1]_.
    alpha1 : float, optional
        Memory loss for player 1, as defined in [2]_.
    alpha2 : float, optional
        Memory loss for player 2, as defined in [2]_. 

    Attributes
    ----------
    shape : tuple
        Return the shape of payoff matrix for player 1. The first argument is the number of strategies for player 1, and the second argument is the number of strategies for player 2.
    payoff : ndarray
        Return the matrix [0 A; B 0] for computations in replicator dynamics.
    
    References
    ----------
    .. [1] Official dynamics of games notes by Prof. Sebestian van-Strien. Available at https://www.ma.imperial.ac.uk/~svanstri/teaching.php.
    .. [2] Y. Sato, E. Akiyama, and J. P Crutchfield, Stability and diversity in collective adaptation, Physica. D 210 (2005),
no. 1, 21–57
    '''

    def __init__(self, payoff1, payoff2, convention=False, alpha1=0, alpha2=0):
        self.payoff1 = payoff1

        # mode represents what representation we are using.
        if convention:
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
        '''
        The right hand side of two-population replicator dynamics.

        Parameters
        ----------
        x : ndarray
            Choice distribution vector.
        t : float, optional
            Time variable for running numerical simulation.    

        Return
        ------
        ndarray

        See Also
        --------
        twoD_replicator.adaptive : two-population adaptive reinforcement learning equation for ndarray.
        twoD_replicator.replicator_tensor : two-population replicator equation for tensors.
        twoD_replicator.adaptive_tensor : two-population adaptive reinforcement learning equation for Tensors.
        '''
        n = self.shape[0]
        m = self.shape[1]
        dx = np.zeros(m+n)
        dx[:n] = x[:n] * (self.payoff1@x[n:] - np.dot(x[:n], self.payoff1@x[n:]))
        dx[n:] = x[n:] * (self.payoff2@x[:n] - np.dot(x[n:], self.payoff2@x[:n]))
        return dx
    
    def adaptive(self,x,t=None):
        '''
        The right hand side of two-population adaptive reinforcement learning equation, as defined in [1]_.

        Parameters
        ----------
        x : ndarray
            Choice distribution vector.
        t : float, optional
            Time variable for running numerical simulation.

        Return
        ------
        ndarray

        See Also
        --------
        twoD_replicator.replicator : two-population replicator equation for ndarray.
        twoD_replicator.replicator_tensor : two-population replicator equation for tensors.
        twoD_replicator.adaptive_tensor : two-population adaptive reinforcement learning equation for Tensors.

        References
        ----------
        .. [1] Official dynamics of games notes by Prof. Sebestian van-Strien. Available at https://www.ma.imperial.ac.uk/~svanstri/teaching.php.        
        '''
        n = self.shape[0]
        m = self.shape[1]
        dx = np.zeros(m+n)
        dx[:n] = x[:n] * (self.payoff1@x[n:] - np.dot(x[:n], self.payoff1@x[n:]) + self.memory1 * (-np.log(x[:n]) + np.sum(xlogy(x[:n], x[:n]))))
        dx[n:] = x[n:] * (self.payoff2@x[:n] - np.dot(x[n:], self.payoff2@x[:n]) + self.memory2 * (-np.log(x[n:]) + np.sum(xlogy(x[n:], x[n:]))))
        return dx

    def replicator_tensor(self,x_raw,t=None,numpy_output=False):
        '''
        The right hand side of two-population replicator equation.

        Parameters
        ----------
        x_raw : Tensor
            Choice distribution vector.
        t : float, optional
            Time variable for running numerical simulation.
        numpy_output : bool, optional
            If `true` return as an ndarray, if `false` return as a Tensor.

        Return
        ------
        ndarray if `numpy_output=True`, otherwise Tensor

        See Also
        --------
        twoD_replicator.replicator : two-population replicator equation for ndarray.
        twoD_replicator.adaptive : two-population adaptive reinforcement learning equation for ndarray.
        twoD_replicator.adaptive_tensor : two-population adaptive reinforcement learning equation for tensors.        
        '''
        n = self.shape[0]
        m = self.shape[1]
        A = tensor(self.payoff1).double()
        B = tensor(self.payoff2).double()
        x = x_raw.double()
        dx = zeros(m+n)
        dx[:n] = x[:n] * (matmul(A, x[n:]) - dot(x[:n], matmul(A, x[n:])))
        dx[n:] = x[n:] * (matmul(B, x[:n]) - dot(x[n:], matmul(B, x[:n])))

        if numpy_output:
            return dx.numpy()
        else:
            return dx
    
    def adaptive_tensor(self,x_raw,t=None):
        '''
        The right hand side of two-population adaptive reinforcement learning equation, as defined in [1]_.

        Parameters
        ----------
        x_raw : Tensor
            Choice distribution vector.
        t : float, optional
            Time variable for running numerical simulation.

        Return
        ------
        ndarray if `numpy_output=True`, otherwise Tensor

        See Also
        --------
        twoD_replicator.replicator : two-population replicator equation for ndarray.
        twoD_replicator.adaptive : two-population adaptive reinforcement learning equation for ndarray.
        twoD_replicator.replicator_tensor : two-population replicator equation for tensors.

        References
        ----------
        .. [1] Y. Sato, E. Akiyama, and J. P Crutchfield, Stability and diversity in collective adaptation, Physica. D 210 (2005),
no. 1, 21–57.  
        '''
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
        number of points included in the array, including start and end points.
        
    Return
    ------
    ndarray
        array of point on the line, with row index being coordinate index and column index being index of points.
    '''
    
    # ts = 0 represents starting point, ts = 1 represents end point.
    ts = np.array([np.linspace(0,1,points)])
    
    start_reshaped = np.array(start).reshape((len(start),1))
    end_reshaped = np.array(end).reshape((len(end),1))
    
    return start_reshaped + ((end_reshaped - start_reshaped) @ ts)

def proj_3D_simplex(array_to_be_transformed):
    '''
    Project points on a 3D simplex onto a 2D triangle by using method in [1]_.
    
    Parameters
    ----------
    array_to_be_transformed : ndarray
        array of point on the line, with row index being coordinate index and column index being index of points.
        
    Return
    ------
    ndarray 
        array of point on the line, with row index being coordinate index and column index being index of points.

    References
    ----------
    .. [1] Official dynamics of games notes by Prof. Sebestian van-Strien. Available at https://www.ma.imperial.ac.uk/~svanstri/teaching.php.
    '''
    proj_mat = np.array(
    [[-1 * np.cos(30. / 360. * 2. * np.pi),np.cos(30. / 360. * 2. * np.pi), 0.],
     [-1 * np.sin(30. / 360. * 2. * np.pi),-1 * np.sin(30. / 360. * 2. * np.pi), 1.]])
    return proj_mat @ array_to_be_transformed

def proj_6D_2D(array_to_be_transformed):
    '''
    Project points on the product of two 3D simplices onto a 2D plane by using transformation [1]_ to replicate the figures in [2]_.
    
    Parameters
    ----------
    array_to_be_transformed : ndarray
        array of point on the line, with row index being coordinate index and column index being index of points.
        
    Return
    ------
    ndarray 
        array of point on the line, with row index being coordinate index and column index being index of points.

    References
    ----------
    .. [1] Official dynamics of games notes by Prof. Sebestian van-Strien. Available at https://www.ma.imperial.ac.uk/~svanstri/teaching.php.
    .. [2] Y. Sato, E. Akiyama, and J. P Crutchfield, Stability and diversity in collective adaptation, Physica. D 210 (2005),
no. 1, 21–57
    '''
    proj_mat = np.array(
    [[3.65, -1.35, 1.35, 5.35, 1.35, 1.45],
     [0.4, 0.4, 4.6, 1.9, -0.4, 4.4]])
    return proj_mat @ array_to_be_transformed

def initial_2D_simplex_figure(vertex_label = True):
    '''
    To initialise a figure with a triangle.
    
    Parameters
    ----------
    vertex_label : bool, optional
        If `True` then the vertices of the triangle are annotated.
        
    Return
    ------
    fig : matplotlib.figure 
        Figure that the triangle is living on.
    ax : matplotlib.axes
        Axes that the triangle is living on. 
    '''
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
    '''
    To initialise the figure used in [1]_.
    
    Parameters
    ----------
    figsize : tuple, optional
        Determine the size of figure.
        
    Return
    ------
    fig : matplotlib.figure 
        Figure that the boundaries are living on.
    ax : matplotlib.axes
        Axes that the boundaries are living on. 
    '''
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
    '''
    To simulate a new point next to certain original point.
    
    Parameters
    ----------
    original : [1,...] ndarray
        position of original point.
    max_diff : float
        maximum difference between the original and new point
    state : int, optional
        seed for random number generation
        
    Return
    ------
    [1,...] ndarray
        position of new point.
    '''
    # max_diff represents maximum difference to first four coordinates
    
    if state:
        np.random.seed(state)
    
    noise = np.random.rand(len(original))
    noise = noise - np.mean(noise)
    return original + max_diff * noise

def localmax(arr):
    '''
    Determine the local maxima of a sequence.
    
    Parameters
    ----------
    arr : [1,...] ndarray
        array for which the local maxima is computed.
        
    Return
    ------
    [1,...] ndarray
        array containing the local maxima.
    '''
    return arr[1:-1][(arr[2:] < arr[1:-1]) * (arr[1:-1] >= arr[:-2])]