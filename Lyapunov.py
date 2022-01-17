import numpy as np
from scipy.integrate import odeint
from tqdm import tqdm
from torch.autograd.functional import jacobian
from torch import tensor

def MGS(A, return_matrix = True):
    '''
    An implementation of modified Gram Schmidt.

    Parameters
    ----------
    A : array
        matrix with *row* as basis elements
    return_matrix : Boolean (optional)
        control the format of output
    
    Return 
    ------
    basis : list or array
        in list of vectors if return_matrix is `False`, otherwise matrix with *row* as basis elements
    lambda : list
        lengths of basis before normalisation

    References
    ----------
    .. [1] Golub, & Van Loan, C. F. (2013). Matrix computations (Fourth edition.). The Johns Hopkins University Press.
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
    '''
    Helper function to convert a function with tensor input to a function with ndarray input.
    
    Parameters
    ----------
    tensor_func : PyFunctionObject
        Function with tensor input.

    Return
    ------
    PyFunctionObject
        Function with ndarray input.
    '''
    return lambda x: tensor_func(tensor(x)).numpy()

def compute_Jacobian(func):
    '''
    Compute Jacobian of a function which has tensor input and tensor output using automatic differentiation.

    Parameters
    ----------
    func : PyFunctionObject
        Function for which Jacobian is computed.

    Return
    ------
    PyFunctionObject
        Jacobian of a function. This is a function with ndarray input and output.

    References
    ----------
    .. [1] Official dynamics of games notes by Prof. Sebestian van-Strien. Available at https://www.ma.imperial.ac.uk/~svanstri/teaching.php.
    '''
    jacob = lambda x: jacobian(func,x)
    return tensor_to_numpy(jacob)


def Lyapunov(physical, initial_data, 
             step=100, interval=0.1, 
             physical_jacobian=None, physical_tensor_to_numpy=True, initial_directions=None, 
             show_x=False, return_end=False):
    '''
    Compute the finite time Lyapunov spectrum of dynamical system \dot{x} = f(x) by solving the first variational function.

    Parameters
    ----------
    physical : PyFunctionObject
        Function f(x) of the dynamical system for which the Lyapunov spectrum is computed.
    initial_data : ndarray
        Initial condition of the dynamical system
    step : int, optional
        Number of steps of simulation
    interval : float, optional
        Stepsize for each timestep.
    physical_jacobian : PyFunctionObject, optional
        Function of the Jacobian of f(x). If not supplied then the Jacobian is computed using `compute_Jacobian`.
    physical_tensor_to_numpy : bool, optional
        `True` indicates that the function supplied for `physical` is a function with tensor input.
    initial_directions : ndarray, optional
        Initial directions for the first variational equation, arranged as matrix with columns as directions. If not supplied then the directions are assumed to be the directions of standard basis vectors.
    show_x : bool, optional
        If `True` then the trajectory of the first variational function is computed and returned.
    return_end : bool, optional
        If `True`, only the final values of Lyapunov spectrum is returned, otherwise the Lyapunov spectra at different timestep are returned as a matrix.

    Return
    ------
    lambda : ndarray
        If `return_end=True`, only the final values of Lyapunov spectrum is returned, otherwise the Lyapunov spectra at different timestep are returned as a matrix.
    x_arr : ndarray or None
        Return the trajectory of the first variational function is computed and returned when `show_x=True`.

    Notes
    -----
    After solving the first variational equation for a small time step, the directions are orthogonalised to ensure numerical stability.

    References
    ----------
    .. [1] J. C. Vallejo, Predictability of chaotic dynamics a finite-time lyapunov exponents approach, 2nd ed. 2019., Springer
Series in Synergetics, Springer International Publishing, Cham, 2019.
    .. [2] K Ramasubramanian and M. S Sriram, A comparative study of computation of lyapunov spectra with different
algorithms, Physica. D 139 (2000), no. 1, 72–86.
    '''
    
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

        # solving the variational equation
        new_raw_x = odeint(variational, x, t=[0, interval])[-1]

        # perform MGS to obtain Lyapunov exponents and new directions
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