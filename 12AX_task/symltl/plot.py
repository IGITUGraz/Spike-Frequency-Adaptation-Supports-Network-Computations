import matplotlib.ticker as ticker
import numpy as np

# Covert data units to points
def height_from_data_units(height, axis, reference='y', value_range=None):
    """
    Convert a linewidth in data units to linewidth in points.

    Parameters
    ----------
    height: float
        Linewidth in data units of the respective reference-axis
    axis: matplotlib axis
        The axis which is used to extract the relevant transformation
        data (data limits and size must not change afterwards)
    reference: string
        The axis that is taken as a reference for the data width.
        Possible values: 'x' and 'y'. Defaults to 'y'.

    Returns
    -------
    linewidth: float
        Linewidth in points
    """
    fig = axis.get_figure()
    if reference == 'x':
        length = fig.bbox_inches.width * axis.get_position().width
        if value_range is None:
            value_range = np.diff(axis.get_xlim())
    elif reference == 'y':
        length = fig.bbox_inches.height * axis.get_position().height
        if value_range is None:
            value_range = np.diff(axis.get_ylim())
    # Convert length to points
    length *= 72
    # Scale height to value range
    ms = height * (length / value_range)
    # print(height, length, value_range, ms)
    return ms


def plot_spikes(spikes, ax):
    n_neurons = spikes.shape[0]
    neurons = np.arange(n_neurons) + 1
    sps = spikes * (neurons.reshape(len(spikes), 1))
    sps[sps == 0.] = -10
    sps -= 1
    marker_size = height_from_data_units(0.9, ax, value_range=n_neurons)
    for neuron in range(n_neurons):
        ax.plot(range(spikes.shape[1]), sps[neuron, :], marker='|', linestyle='none', color='black',
                markersize=marker_size, markeredgewidth=0.7)
    ax.set(ylim=(-1, n_neurons + 1))
    ax.yaxis.set_major_locator(ticker.FixedLocator([0, n_neurons]))
