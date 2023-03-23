import numpy as np


def box_transform(z, z_min, z_max):
    """Projects z from box [z_min, z_max] to [0, 1]

    :param z: an array of z values
    :param z_min: float, the minimum value of the z box
    :param z_max: float, the maximum value of the z box
    :returns: z projected onto [0, 1]

    """

    return (z - z_min) / (z_max - z_min)


def make_grid(n_grid, z_min, z_max):
    """Create grid of equally spaced points

    :param n_grid: integer number of grid points
    :param z_min: float, the minimum value of the z box
    :param z_max: float, the maximum value of the z box
    :returns: a grid of n_grid equally spaced points between z_min and z_max

    """
    return np.linspace(z_min, z_max, n_grid).reshape((n_grid, 1))


def params_dict_optim_decision(params, multi_output=False):
    """
    Ingest parameter dictionary and determines whether to do CV optimization.
    If one of the parameter has a list of length above 1 as values
    then automatically format the dictionary for GridSearchCV.

    :param params: dictionary of model parameters
    :param multi_output: boolean flag, whether the optimization would need
        to be performed in MultiOutputRegressor
    :returns: a dictionary of parameters and a boolean flag of whether CV-opt
        is going to be performed. If CV-optimization is set to happen then
        the paramater dictionary is correctly format.
    """

    # Determines whether there are any list in the items of the dictionary
    opt_flag = False
    for k, value in params.items():
        if type(value) == tuple:
            raise ValueError(
                "Parameter values need to be lists or np.array, not tuple."
                "Current issues with parameter %s" % (k)
            )
        if type(value) == list or type(value) == np.ndarray:
            opt_flag = True
            break

    # Format the dictionary if necessary - put int, str and float into a list
    # with one element
    out_param_dict = {} if opt_flag else params.copy()
    if opt_flag:
        for k, value in params.items():
            out_value = value.tolist() if type(value) == np.ndarray else value
            out_value = [out_value] if type(out_value) != list else out_value
            out_key = "estimator__" + k if multi_output else k
            out_param_dict[out_key] = out_value

    return out_param_dict, opt_flag


def params_name_format(params, str_rem):
    """
    Changes all the key in dictionaries to remove a specific word from each key (``estimator__``).
    This is because in order to GridsearchCV on MultiOutputRegressor one needs to
    use ``estimator__`` in all parameters - but once the best parameters are fetched
    the name needs to be changed.

    :param params: dictionary of model parameters
    :param str_rem: word to be removed
    :returns: dictionary of parameters in which the word has been removed in keys
    """
    out_dict = {}
    for k, v in params.items():
        new_key = k.replace(str_rem, "") if str_rem in k else k
        out_dict[new_key] = v
    return out_dict
