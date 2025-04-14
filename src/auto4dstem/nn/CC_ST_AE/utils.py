########################################################
# Helper functions for reverse affine transform
########################################################


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