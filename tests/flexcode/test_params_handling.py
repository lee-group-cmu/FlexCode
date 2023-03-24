import numpy as np
import pytest

from flexcode.helpers import params_dict_optim_decision


def test_params_transform():
    dict1 = {"k": [1, 2, 3, 4, 5, 6]}
    dictout1a, flag1a = params_dict_optim_decision(dict1)

    assert dictout1a == dict1
    assert flag1a == True

    dictout1b, flag1b = params_dict_optim_decision(dict1, True)
    assert dictout1b == {"estimator__k": [1, 2, 3, 4, 5, 6]}
    assert flag1b == True

    ############################################################

    dict2 = {"k": 1}
    dictout2a, flag2a = params_dict_optim_decision(dict2)

    assert dictout2a == dict2
    assert flag2a == False

    dictout2b, flag2b = params_dict_optim_decision(dict2, True)
    assert dictout2b == dict2
    assert flag2b == False

    #############################################################

    dict3 = {"k": [1, 2, 3, 4, 5, 6], "obj": "linear", "eta": 0.3}
    dictout3a, flag3a = params_dict_optim_decision(dict3)

    assert dictout3a == {"k": [1, 2, 3, 4, 5, 6], "obj": ["linear"], "eta": [0.3]}
    assert flag3a == True

    dictout3b, flag3b = params_dict_optim_decision(dict3, True)
    assert dictout3b == {
        "estimator__k": [1, 2, 3, 4, 5, 6],
        "estimator__obj": ["linear"],
        "estimator__eta": [0.3],
    }
    assert flag3b == True

    ############################################################

    dict4 = {"k": 1, "obj": "linear", "eta": 0.3}
    dictout4a, flag4a = params_dict_optim_decision(dict4)

    assert dictout4a == dict4
    assert flag4a == False

    dictout4b, flag4b = params_dict_optim_decision(dict4, True)
    assert dictout4b == dict4
    assert flag4b == False

    ############################################################

    dict5 = {"k": (1, 2, 3)}
    with pytest.raises(Exception):
        _ = params_dict_optim_decision(dict5)

    #############################################################

    dict6 = {"k": [1, 2, 3, 4, 5, 6], "obj": "linear", "eta": np.linspace(0.3, 0.5, 3)}
    dictout6a, flag6a = params_dict_optim_decision(dict6)

    assert dictout6a == {"k": [1, 2, 3, 4, 5, 6], "obj": ["linear"], "eta": [0.3, 0.4, 0.5]}
    assert flag6a == True

    dictout6b, flag6b = params_dict_optim_decision(dict6, True)
    assert dictout6b == {
        "estimator__k": [1, 2, 3, 4, 5, 6],
        "estimator__obj": ["linear"],
        "estimator__eta": [0.3, 0.4, 0.5],
    }
    assert flag6b == True
