import matplotlib.pyplot as plt
from auto4dstem.viz.label_style import apply_figure_labels


def display_diffraction_image(  
        data, clim=[0, 1], cmap="viridis", **kwargs
    ):
        """Function to pick one image for visualization.

        Args:
            index (int, optional): Index of the image to pick. Defaults to 0.
            clim (list, optional): Color range for plt.imshow. Defaults to [0, 1].
            cmap (str, optional): Color map for plt.imshow. Defaults to 'viridis'.
            add_label (bool, optional): Whether to add a label to the figure. Defaults to True.
            label_style (str, optional): Style of the label. Defaults to 'wb'.
        """

        # visualize image
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.imshow(data, cmap=cmap, clim=clim)
        
        # apply figure labels
        apply_figure_labels(ax, **kwargs)
        