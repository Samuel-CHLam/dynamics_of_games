import numpy as np
from scipy.integrate import odeint
from tqdm import tqdm
from torch.autograd.functional import jacobian
from torch import tensor

def MGS(A, return_matrix = True):
    '''
    input
    -----
    A : array
        matrix with *row* as basis elements
    return_matrix : Boolean (optional)
        control the format of output
    
    output 
    ------
    basis : list or array
        in list of vectors if return_matrix is False, otherwise matrix with *row* as basis elements
    lambda : list
        lengths of basis before normalisation
    '''
    
    # initialisation
    n,m = A.shape
    V = A
    Qlist = []
    Lambda = []
    
    # modified Gram Schmidt
    for i in range(n):
        Lambda.append(np.linalg.norm(V[i]))
        Qlist.append(V[i]/np.linalg.norm(V[i]))
        for j in range(i,n):
            V[j] -= np.dot(V[j], Qlist[-1]) * Qlist[-1]
    
    if return_matrix:
        Q = np.array(Qlist)
        return Q, Lambda
    
    return Qlist, Lambda

def tensor_to_numpy(tensor_func):
    '''converts np.array with float entries to tensor with float64 entries'''
    return lambda x: tensor_func(tensor(x)).numpy()

def compute_Jacobian(func):
    '''assuming func is tensor to tensor, output a function which compute Jacobian at a point.'''
    jacob = lambda x: jacobian(func,x)
    return tensor_to_numpy(jacob)


def Lyapunov(physical, initial_data, 
             step=100, interval=0.1, 
             physical_jacobian=None, physical_tensor_to_numpy=True, initial_directions=None, 
             show_x=False, return_end=False):
    
    # compute Jacobian matrix using automatic differentiation
    if physical_jacobian == None:
        physical_jacobian = compute_Jacobian(physical)
        
    # include a numpy version of physical
    if physical_tensor_to_numpy:
        numpy_physical = tensor_to_numpy(physical)
    else:
        numpy_physical = physical
    
    # compute variational function, with n = length of initial data
    def variational(y,t):
        n = len(initial_data)
        output = list(numpy_physical(y[:n]))
        for i in range(1, n+1):
            output += list(physical_jacobian(y[:n]) @ y[i*n: (i+1)*n])
        return np.array(output)
    
    # initialisations:
    
    # initialise initial directions as standard basis directions
    if initial_directions == None:
        initial_directions = list(np.eye(np.size(initial_data)))    
    
    # form initial vector
    x = np.array(list(initial_data) + list(np.array(initial_directions).reshape(-1)))
    
    if show_x:
        x_arr = [x]
    
    Big_log_N = np.zeros(len(initial_data))
    Lambda = []
    
    # compute Lyapunov exponent
    for i in tqdm(range(1,step+1), position=0, leave=True):
        new_raw_x = odeint(variational, x, t=[0, interval])[-1]
        new_matrix_x = new_raw_x.reshape(-1,len(initial_data))
        new_matrix_x[1:], Big_N = MGS(new_matrix_x[1:])
        x = new_matrix_x.reshape(-1)
        if show_x:
            x_arr.append(x)
        Big_log_N += np.log(Big_N)
        Lambda.append(Big_log_N / (i*interval))
    
    if return_end:
        if show_x:
            return Lambda[-1], x_arr
        else:
            return Lambda[-1]
    else:
        if show_x:
            return np.array(Lambda), np.array(x_arr)
        else:
            return np.array(Lambda)