import numpy as np

"""
    File containing the integrator implementations
    notice that these are all made to update the entire system at once,
    i.e. they require all the positions, velocities, forces and masses of 
    the entire system
"""

def integrator_euler(x, v, f, m, delta_t):
    """
    Applies the euler time-integration

    Input:
        x: Positions
        v: Velocities
        f: Force 
        m: Mass
        delta_t: Timestep
    Output:
        x_new: new position
        v_new: new velocities
    """

    x_new = x + delta_t*v + (delta_t**2 / 2)*np.true_divide(f, m[:, np.newaxis])
    v_new = v + (delta_t) * np.true_divide(f, m[:, np.newaxis])

    return x_new, v_new

def integrator_verlet_pos(x, x_old, f, m, delta_t):
    """
    Applies verlet algorithm to the position

    Input:
        x: position
        x_old: position one timestep before
        f: force on particle
        m: mass
        delta_t: timestep
    Output:
        x_new: new position
    """

    x_new = 2*x - x_old + (delta_t**2)*np.true_divide(f, m[:, np.newaxis])

    return x_new

def integrator_verlet_vel(x_new, x_old, delta_t):
    """
    Applies verlet algorithm to the velocity

    Input:
        x_new: position one timestep after
        x_old: position one timestep before
        f: force on particle
        m: mass
        delta_t: timestep
    Output:
        v_new: new velocity
    """

    v_new = (1/(2* delta_t))*(x_new - x_old)

    return v_new

def integrator_velocity_verlet_pos(x, v, f, m, delta_t):
    """
    Applies velocity_verlet time-integration on x

    Input:
        x: Positions
        v: Velocities
        f: Force 
        m: Mass
        delta_t: Timestep
    Output:
        x_new: new position
    """

    x_new = x + delta_t*v + (delta_t**2 / 2)*np.true_divide(f, m[:, np.newaxis])

    return x_new

def integrator_velocity_verlet_vel(v, f, f_new, m, delta_t):
    """
    Applies velocity_verlet time-integration on x

    Input:
        x: Positions
        v: Velocities
        f: Force 
        m: Mass
        delta_t: Timestep
    Output:
        v_new: new velocity
    """

    v_new = v + (delta_t/2)*np.true_divide(f_new+f, m[:, np.newaxis])

    return v_new