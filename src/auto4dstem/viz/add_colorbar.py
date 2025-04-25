import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def add_colorbar(im, ax, size="5%", pad=0.05):
    """function to add colorbar to subplots

    Args:
        im (matplotlib image): image that colorbar comes from
        ax (matplotlib subplot ax): where to attach the colorbar
        size (str, optional): size of the colorbar. Defaults to "5%".
        pad (float, optional): pad of the colorbar. Defaults to 0.05.
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    plt.colorbar(im, cax=cax)
