def box_transform(z, z_min, z_max):
    """Projects z from box [z_min, z_max] to [0, 1]

    :param z: an array of z values
    :param z_min: float, the minimum value of the z box
    :param z_max: float, the maximum value of the z box
    :returns: z projected onto [0, 1]
    :rtype: array

    """

    return (z - z_min) / (z_max - z_min)
