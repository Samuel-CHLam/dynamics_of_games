# Permanance and Chaos of Game and Reinforcement Dynamics
Project for MATH97069 Dynamics of Games.

[Final Write-Up](https://www.overleaf.com/read/vgzpxvmmpncm) | [Presentation Slides](https://www.overleaf.com/read/dyjjtqjbpwjg) | [Presentation Video](https://imperiallondon-my.sharepoint.com/:v:/g/personal/chl4817_ic_ac_uk/EU0yuAPZ40BMrsuUqg47losBsuJz6_gpMvS8QKKRBq7PqA?e=ascYvM) (Imperial credential required.)

Please clone the entire file for assessment, since most of the main files require code from the associated Python modules. The main files I have used include:
- [Computation of Lyapunov Exponent of Lorenz System and Hypercycle](https://github.com/Samuel-CHLam/dynamics_of_games/blob/main/Computation%20of%20Lyapunov%20Exponent%20of%20Lorenz%20System%20and%20Hypercycle.ipynb), and
- [Lyapunov for RSP (Rock-Paper-Scissor)](https://github.com/Samuel-CHLam/dynamics_of_games/blob/main/Lyapunov_RSP.ipynb)

Most functions from the modules include a docstring. If you want to view it, simply import the module and type `<your function name>?` in your console. For instance, if you type `Lyapunov?`, you will get the docstring of the function `Lyapunov`, which should be the following:

```
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
```

Please email me at [chun.lam18@imperial.ac.uk](mailto:chun.lam18@imperial.ac.uk) for any enquiries. Thank you for viewing/marking my project.