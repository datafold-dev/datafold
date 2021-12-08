import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp

from datafold.pcfold.kernels import RadialBasisKernel, _apply_kernel_function_numexpr


class InvertedPendulum:
    '''
    Model the physics of an inverted pendulum on a cart
    controlled by electric motor.
    '''
    def __init__(self, 
                # Pendulum parameters
                pendulum_mass      = .0905, # kgg
                cart_mass          = 1.12,  # k
                g                  = 9.81,  # m/s^2
                tension_force_gain = 7.5,   # -
                pendulum_length    = .365,  # m
                cart_friction      = 6.65): # -
        self.pendulum_mass      = pendulum_mass     
        self.cart_mass          = cart_mass         
        self.g                  = g                 
        self.tension_force_gain = tension_force_gain
        self.pendulum_length    = pendulum_length   
        self.cart_friction      = cart_friction
        self.reset()
    
    def reset(self):
        '''
        Restore to neutral position at 0 time.
        '''
        self.state = np.array([[0, 0, np.pi, 0]])
        self.last_time = 0
    
    def _f(self, t, state, control_input):
        # inverted pendulum physics
        x, xdot, theta, thetadot = state
        f1 = xdot
        f3 = thetadot
        # Sabin version - math error or coding error?
        #f4 = ((self.cart_mass + self.pendulum_mass)*self.g*np.sin(theta) 
        #      + self.cart_friction*np.cos(theta)*xdot
        #      - self.pendulum_mass*self.pendulum_length*thetadot**2*np.sin(theta)*np.cos(theta)) \
        #     / (self.pendulum_length*(self.cart_mass+self.pendulum_mass*np.sin(theta)**2))
        #f2 = (self.tension_force_gain*control_input 
        #      - self.cart_friction*xdot 
        #      - self.pendulum_mass*self.pendulum_length*f4**2*np.sin(theta)
        #      - self.pendulum_mass*self.g*np.cos(theta)*np.sin(theta)) \
        #     / (self.cart_mass+self.pendulum_mass*np.sin(theta)**2)
        # Sabin Ref.[16] version
        f2 = (self.tension_force_gain*control_input
              - self.cart_friction*xdot
              - self.pendulum_mass*self.pendulum_length*thetadot**2*np.sin(theta)) \
             /(self.cart_mass + self.pendulum_mass*np.sin(theta)**2)
        f4 = (self.tension_force_gain*control_input
              - self.cart_friction*xdot
              - self.pendulum_mass*self.pendulum_length*thetadot**2*np.sin(theta)*np.cos(theta)
              + (self.cart_mass + self.pendulum_mass)*self.g*np.sin(theta)) \
             /(self.pendulum_length - self.pendulum_mass*self.pendulum_length*np.cos(theta)**2)
        return np.array((f1, f2, f3, f4))
    
    def _check_state(self, state):
        if state is None:
            # use last state if none given
            state = self.state
        else:
            # make sure state is the right shape
            try:
                state = state.reshape(4,1)
            except ValueError as e:
                 raise ValueError('State should have size 4.') from e
        return state
    
    def step(self, time_step, state = None, control_input = 0, current_time = None):
        '''
        Return the next state
        
        time_step - length of single time step
        state - state to step from (default last state of the pendulum)
        control_input - applied control force (default 0)
        current_time - time to step from (default last time of the pendulum)
        '''
        state = self._check_state(state)
        t0 = self.last_time if current_time is None else current_time
        self.sol = solve_ivp(fun = self._f,
                             args = (control_input,),
                             t_span = (t0, t0+time_step),
                             y0 = state.ravel(),
                             method = 'RK45',
                             t_eval = np.atleast_1d(t0 + time_step),
                             vectorized = True)
        self.state = self._check_state(self.sol.y)
        self.last_time = self.sol.t[-1]
        return self.state
    
    def trajectory(self, time_step, num_steps, control_func, initial_state = None, t0 = None):
        '''
        Compute a trajectory in state space
        
        time_step - length of single time step in the output
        initial_state - initial condition
        num_steps - number of time steps in the output
        control_func - f(t, state) callable returning control input
        '''
        if not callable(control_func):
            raise TypeError("control_func needs to be a function of time and the state")
        state = self._check_state(initial_state)
        t0 = self.last_time if t0 is None else t0
        tf = t0+time_step*(num_steps+1)
        self.sol = solve_ivp(fun = lambda t,y: self._f(t, y, control_func(t, y)),
                             t_span = (t0, tf),
                             y0 = state.ravel(),
                             method = 'RK45',
                             t_eval = np.arange(t0, tf, time_step),
                             vectorized = True)
        self.state = self._check_state(self.sol.y[:,-1])
        self.last_time = self.sol.t[-1]
        
        return self.sol.y


# TODO: if actually used, move to datafold.pcfold.kernels
class InverseQuadraticKernel(RadialBasisKernel):
    r"""Inverse quadratic radial basis kernel.

    .. math::
        K = (\frac{1}{2\varepsilon} \cdot D + 1)^{-1}

    where :math:`D` is the squared Euclidean distance matrix.

    See also super classes :class:`RadialBasisKernel` and :class:`PCManifoldKernel`
    for more functionality and documentation.

    Parameters
    ----------
    epsilon
        kernel scale
    """

    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon
        super(InverseQuadraticKernel, self).__init__(distance_metric="sqeuclidean")

    def eval(self, distance_matrix):
        """Evaluate the kernel on pre-computed distance matrix.

        Parameters
        ----------
        distance_matrix
            Matrix of pairwise distances of shape `(n_samples_Y, n_samples_X)`.

        Returns
        -------
        Union[np.ndarray, scipy.sparse.csr_matrix]
            Kernel matrix of same shape and type as `distance_matrix`.
        """

        self.epsilon = self._check_bandwidth_parameter(
            parameter=self.epsilon, name="epsilon"
        )

        return _apply_kernel_function_numexpr(
            distance_matrix,
            expr="1.0 / (1.0 / (2*eps) * D + 1.0)",
            expr_dict={"eps": self.epsilon},
        )