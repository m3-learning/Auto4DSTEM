from m3util.viz.text import labelfigs

def apply_figure_labels(ax, **kwargs):
    
    # default label style
    kwargs["style"] = kwargs.get("style", "wb")
    kwargs["number"] = kwargs.get("number", 0)
    kwargs["loc"] = kwargs.get("loc", "tl")
    kwargs["size"] = kwargs.get("size", 10)
    kwargs["inset_fraction"] = kwargs.get("inset_fraction", (0.1, 0.1))
    add_label = kwargs.get("add_label", True)
    
    if add_label:
        labelfigs(
                ax,
                **kwargs,
            )