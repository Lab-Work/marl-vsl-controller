import numpy as np
from config.globals import *
import torch


def min_max_norm(feature, value):
    """
    The min-max normalization transformation.
    :params feature: a string to indicate the traffic characteristic type, i.e., speed, volume or occupancy.
    :params value: the raw value to be normalized.
    """
    speed_min = spd_min
    speed_max = spd_max
    volume_min = vol_min
    volume_max = vol_max
    occupancy_min = occ_min
    occupancy_max = occ_max
    speed_limit_min = sl_min
    speed_limit_max = sl_max
    if feature == 'speed':
        norm_value = (value - speed_min) / (speed_max - speed_min)
    elif feature == 'volume':
        norm_value = (value - volume_min) / (volume_max - volume_min)
    elif feature == 'occupancy':
        norm_value = (value - occupancy_min) / (occupancy_max - occupancy_min)
    elif feature == 'speed_limit':
        norm_value = (value - speed_limit_min) / (speed_limit_max - speed_limit_min)
    else:
        assert False, "feature should be one of speed, volume and occupancy"
    return np.clip(norm_value, 0, 1)


def action_to_speed(action):
    """
    A mapping from the selected action to the corresponding speed limit.
    :param action: the selected action by the agent, which should be an element from [0,1,2,3,4]
    """
    if action == 0:
        speed_limit = 30
    elif action == 1:
        speed_limit = 40
    elif action == 2:
        speed_limit = 50
    elif action == 3:
        speed_limit = 60
    else:
        speed_limit = 70

    return speed_limit


def speed_to_action(speed_limit):
    """
    A mapping from speed limit to the corresponding action.
    :param speed_limit: the speed limit, which should be an element from [30,40,50,60,70]
    """
    if speed_limit == 30:
        action = 0
    elif speed_limit == 40:
        action = 1
    elif speed_limit == 50:
        action = 2
    elif speed_limit == 60:
        action = 3
    else:
        action = 4
    return action


def get_available_action_set(pre_action):
    """
    Generate the valid action set based on the downstream agent's selected action. This set is used for invalid
    action masking so that we can ensure the step-down rules are guaranteed.
    :param pre_action: the action selected by the downstream agent, which should be an element from [0,1,2,3,4]
    """
    available_act_set = [0, 0, 0, 0, 0]
    if pre_action == 0:
        available_act_set[0] = 1
        available_act_set[1] = 1
    elif pre_action == 1 or pre_action == 2:
        available_act_set[: pre_action + 2] = [1] * (pre_action + 2)
    else:
        available_act_set = [1, 1, 1, 1, 1]

    return torch.tensor(available_act_set)
