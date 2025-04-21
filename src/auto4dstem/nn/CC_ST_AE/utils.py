import torch 

def enforce_transformation_boundary(coord, radius, max_val) -> float:
    """Helper function to adjust a coordinate to stay within bounds

    Args:
        coord (float): The coordinate value to adjust
        radius (int): The radius to check bounds against
        max_val (int): The maximum allowed value

    Returns:
        float: The adjusted coordinate value
    """
    if coord - radius < 0:
        return coord - (coord - radius)
    if coord + radius > max_val:
        return coord - ((coord + radius) - max_val)
    return coord


def get_coordinate_range(coord: float, radius: int) -> tuple[int, int]:
    """Calculate start and end coordinates for a given center coordinate and radius

    Args:
        coord (float): Center coordinate
        radius (int): Radius to extend from center

    Returns:
        tuple[int, int]: Start and end coordinates
    """
    return (int(coord - radius), int(coord + radius))


def map_and_load_pkl_weights(model, 
                            pkl_path, 
                            state = 'net', 
                            strict=True, 
                            device =torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                            ):
    """
    Load weights from a .pkl file and map them into a PyTorch model.

    Args:
        model (torch.nn.Module): Target model instance.
        pkl_path (str): Path to the .pkl weight file.
        state (callable): name of the state in pkl file.
        strict (bool): Whether to enforce strict matching of keys.
        device (torch.device): device to map the model weights
    Returns:
        model (torch.nn.Module): Model with loaded weights.
    """
    # Load the .pkl file
    pretrained_data = torch.load(pkl_path, map_location = device)

    if isinstance(pretrained_data, dict):
        if state in pretrained_data:
            pretrained_state = pretrained_data[state]
    else:
        pretrained_state = pretrained_data

    # Extract the weights as list (ordered)
    pretrained_weights = list(pretrained_state.values())
    model_keys = list(model.state_dict().keys())

    if len(pretrained_weights) != len(model_keys):
        raise ValueError(f"Layer count mismatch: {len(pretrained_weights)} (pretrained) vs {len(model_keys)} (model)")

    # Create ordered mapping
    mapped_state = {
        k: v if isinstance(v, torch.Tensor) else torch.tensor(v)
        for k, v in zip(model_keys, pretrained_weights)
    }

    model.load_state_dict(mapped_state, strict=strict)