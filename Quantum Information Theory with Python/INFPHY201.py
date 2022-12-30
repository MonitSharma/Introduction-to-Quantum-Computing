# Numpy
import numpy as np
np.set_printoptions(suppress = True, linewidth=180)

# Scipy: Matrix exponential
from scipy.linalg import expm

# Qutip: Bloch sphere and Quantum object
try:
    from qutip import Bloch, Qobj
except ImportError:
    print('Could not import Bloch and Qobj from qutip library. Did you install it?')

# Matplotlib
import matplotlib.pyplot as plt

# IPython clear_output (for the live_plot function)
from IPython.display import clear_output





def plot_position(state: np.array) -> None:
    """
    Plots the probability distribution of the position of a quantum system.
    It assumes that the position is centered in zero and that \delta x = 1.
    
    Arguments:
        state array(complex): the quantum state vector
    """
    probs = np.abs(state)**2

    fig, ax = plt.subplots()
    x = np.arange(0, len(probs))
    if len(probs) < 10:
        ax.set_xticks(x)
    ax.bar(x, probs, 1)
    ax.set_title('Position of quantum system')
    ax.set_ylabel('Probability')
    plt.show()
    
    
def psi_n(L: int, dim: int, n: int) -> (np.array, np.array):
    """
    Returns the `n`-th energy eigenfunction of a particle in a square box of length `L`
    which has been discretized into `dim` parts.
    
    Arguments:
        L (int): the physical length of the box
        dim (int): the number of segments we are discretizing the box into
        n (int): the index of the energy eigenfunction
    
    Returns:
        x (np.array(float)): the array of coordinates of the segments
        psi (np.array(complex)): the n-th energy eigenfunction
    """
    x = np.linspace(0, L, dim)
    kn = (n+1)*np.pi/L
    
    psi = np.sin(kn*x) # unnormalized
    psi = psi/np.linalg.norm(psi) + 0.0j
    return x, psi


def inner_prod(v, w):
    """
    Computes the inner product of the two vectors v and w
    taking care of conjugating v.
    
    Arguments:
        v (np.array(complex)): ket |v>
        w (np.array(complex)): ket |w>
    
    Returns:
        (complex): the inner product <v|w>
    """
    return np.sum(np.conj(v) * w)


def random_state(d):
    """
    Creates a random state in d dimensions.
    Warning: this state is not uniformly random.
    
    Arguments:
        d (int): the dimension of the state space
        
    Returns:
        (np.array(complex)): a random state in d dimensions
    """
    x = np.random.normal(size=d)
    y = np.random.normal(size=d)
    state = x + 1j*y
    return state/np.linalg.norm(state)


def live_plot(data_dict, figsize=(7,5), title=''):
    """
    Creates a new figure and overwrites the current output with it.
    If called repeatedly it creates an animation in place.
    
    Arguments:
        data_dict (dict): a dictionary with the data of each line
        figsize ((int,int)): a tuple with the size of the figure (default to (7,5))
        title (str): the title of the plot (default to '')
    """
    clear_output(wait=True)
    plt.figure(figsize=figsize)
    for label,data in data_dict.items():
        plt.plot(data, label=label)
    plt.title(title)
    plt.xlabel(f'x')
    plt.legend(loc='center left')
    plt.show();