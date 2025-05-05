import matplotlib.pyplot as plt
from auto4dstem.calculations.noise import PoissonNoise
from auto4dstem.viz.label_style import apply_figure_labels

from m3util.viz.axes import remove_all_ticks
from m3util.util.kwargs import filter_kwargs


def display_diffraction_image(data, clim=[0, 1], cmap="viridis", **kwargs):
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
    remove_all_ticks()
    ax.imshow(data, cmap=cmap, clim=clim)

    # apply figure labels
    apply_figure_labels(ax, **kwargs)


def display_noisy_diffraction(
    data,
    folder_path,
    noise_profile=PoissonNoise,
    noise_level=[0],
    clim=[0, 1],
    cmap="viridis",
    **kwargs,
):
    """function to visualize poisson noise scaling images

    Args:
        noise_level (list, optional): list of noise level. Defaults to [0].
        clim (list, optional): color range of plot. Defaults to [0,1].
        file_name (str, optional): name of saved figure. Defaults to ''.
        cmap (str, optional): color map of imshow. Defaults to '1'.
        add_label (bool, optional): determine if add label to figure. Defaults to True.
        label_style (str, optional): determine label style. Defaults to 'wb'
    
    kwargs:
        dpi (int, optional): dpi of saved figure. Defaults to 600.
        save_format (str, optional): format of saved figure. Defaults to 'svg'.
        file_prefix (str, optional): prefix of saved figure. Defaults to ''.
    """
    
    kwargs.setdefault("dpi", 600)
    kwargs.setdefault("save_format", "svg")
    kwargs.setdefault("file_prefix", "")
    kwargs.setdefault("label_figs", True)

    
    fig, ax = plt.subplots(1, len(noise_level), figsize=(4 * len(noise_level), 4))

    # Ensure ax is always an array-like structure
    if len(noise_level) == 1:
        ax = [ax]

    # filter kwargs
    filtered_kwargs = filter_kwargs(noise_profile, kwargs)
    
    # generate noise
    noise_generator = noise_profile(**filtered_kwargs)

    # add poisson noise on image
    for i, background_weight in enumerate(noise_level):
        # generate string of noise
        bkg_str = format(int(background_weight * 100), "02d")

        # generate noise
        noise_generator.background_weight = background_weight
        int_noisy = noise_generator.generate(data)

        ax[i].title.set_text(f"{bkg_str} Percent")
        ax[i].imshow(int_noisy, cmap=cmap, clim=clim)


    printer = Printer(base_path=folder_path, **kwargs)
    printer.save_figure(fig, f"{file_prefix}_generated_{noise_level}_noise", **kwargs)
    
    

        # apply figure labels
        apply_figure_labels(ax[i], number=i, **kwargs)
        
    

    # clean x,y tick labels
    remove_all_ticks()
    fig.tight_layout()

    # save figure
    plt.savefig(
        f"{folder_path}/{file_name}_generated_{noise_level}_noise.{save_format}",
        dpi=dpi,
    )
