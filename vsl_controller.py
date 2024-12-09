from mappo.algorithm.mappo_policy import MAPPOPolicy as Policy
from mappo.config import get_config
import gym
import pandas as pd
from util import *
from safety_guards import *



# Get MAPPO configurable parameters
parser = get_config()
args = parser.parse_args(args=['--use_feature_normalization'])
observation_space = gym.spaces.Box(low=-np.inf, high=+np.inf, shape=(5,), dtype=np.float32)
action_space = gym.spaces.Discrete(5)
device = torch.device("cpu")

# Initialize model
policy = Policy(args, observation_space, action_space, device)

# Load the learned model
actor_path = ('./mappo/model/actor.pt')
policy_actor_state_dict = torch.load(actor_path)
policy.actor.load_state_dict(policy_actor_state_dict)
ai_model = policy.actor

# Get state input dataset of Monday, April 22, 2024
dataset_all = pd.read_pickle('dataset/dataset.pkl')

# Extract the state input information to reproduce the control outputs
input_dataset = dataset_all.loc[:, ['time_index', 'mm', 'down_spd', 'down_occ', 'up_spd', 'up_occ']]
input_dataset.loc[:, 'final_control_output'] = None

# Initialize an empty output dataset
output_dataset = pd.DataFrame()

# Simulate getting input datastream and generate VSL outputs over time
for time_idx in input_dataset.time_index.unique():
    datastream = input_dataset[input_dataset.time_index == time_idx].sort_values('mm').reset_index(
        drop=True)  # the top one is the most downstream one

    # Set the preceding (downstream) agent's speed limit of the most downstream agent as the default maximum, i.e., 70 mph
    pre_speed_limit_raw = sl_max
    # At each timestep, loop through all gantries starting from the most downstream one
    for index, row in datastream.iterrows():
        # -------------------- STEP 1: Data Preprocessing (This step has been skipped in this tutorial) --------------------
        # -------------------- STEP 2: MARL Policy Evaluation and Speed Matching Correction --------------------
        # get observation data (input data)
        down_spd_raw = row['down_spd']
        down_occ_raw = row['down_occ']
        up_spd_raw = row['up_spd']
        up_occ_raw = row['up_occ']

        down_spd_norm = min_max_norm(feature='speed', value=down_spd_raw)
        down_occ_norm = min_max_norm(feature='occupancy', value=down_occ_raw)
        up_spd_norm = min_max_norm(feature='speed', value=up_spd_raw)
        up_occ_norm = min_max_norm(feature='occupancy', value=up_occ_raw)
        pre_action_norm = min_max_norm(feature='speed_limit', value=pre_speed_limit_raw)

        observation = torch.tensor([down_spd_norm, down_occ_norm, up_spd_norm, up_occ_norm, pre_action_norm])

        # Get available action set
        available_action_set = get_available_action_set(pre_action=speed_to_action(pre_speed_limit_raw))

        # Generate action/speed limit by MARL-based policy (here we have the invalid action masking layer)
        ai_action = ai_model(observation, available_actions=available_action_set, deterministic=True).item()

        # Speed-Matching Correction
        sm_corrected = speed_matching_correction(ai_action=ai_action, down_speed_raw=down_spd_raw,
                                                 down_occupancy_raw=down_occ_raw,
                                                 pre_speed_limit_raw=pre_speed_limit_raw)

        if sm_corrected:
            datastream.loc[datastream.index[index], 'final_control_output'] = action_to_speed(sm_corrected)
            pre_speed_limit_raw = action_to_speed(sm_corrected)
        else:
            datastream.loc[datastream.index[index], 'final_control_output'] = action_to_speed(ai_action)
            pre_speed_limit_raw = action_to_speed(ai_action)

    # -------------------- STEP 3: Maximum Speed Limit Correction --------------------
    for index, row in datastream.iterrows():
        datastream.loc[datastream.index[index], 'final_control_output'] = max_speed_limit_correction(
            input_speed_limit=row.final_control_output, mm=row.mm, mm_to_maxspeedlimit=mm_to_maxsl)

    # -------------------- STEP 4: Bounce Correction --------------------
    speed_limit_list = datastream.final_control_output.to_list()
    datastream.loc[:, 'final_control_output'] = bounce_correction(speed_limit_list=speed_limit_list)

    # Concatenate the datastream to form output_dataset
    output_dataset = pd.concat([output_dataset, datastream], ignore_index=True)

output_dataset.to_pickle('dataset/control_outputs_dataset.pkl')
print(f'Generate dataset with control outputs at dataset/control_outputs_dataset.pkl')

