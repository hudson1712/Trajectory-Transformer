import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import pickle
from transformers import DecisionTransformerConfig, DecisionTransformerModel
import json
import utilities
import uuid

#configuration = DecisionTransformerConfig()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
evaluation_mode = True

# Read config from json config_file/config.json and convert to decision transformer config
config_file_path = 'config_file/config.json'
with open(config_file_path, 'r') as f:
    config = json.load(f)

# Define Model and Data parameters
state_dim = 29  # depends on your specific environment
action_dim = 1  # depends on your specific environment
max_length = 48  # maximum sequence length/context size
hidden_size = 64 # dimension of the hidden embeddings, quadratic in model parameter size
num_heads = 2 # number of attention heads
num_layers = 2 # number of transformer layers

# Define Training parameters
num_epochs = 100 # training epochs
batch_size = 128 # number of trajectories per batch
early_stopping = 1 # set to 1 if you want to stop training when the validation loss stops improving

#Adjust the configuration
config['state_dim'] = state_dim
config['act_dim'] = action_dim
config['max_ep_len'] = max_length
config['hidden_size'] = hidden_size
config['n_layer'] = num_layers
config['n_head'] = num_heads
config['n_positions'] = 3*max_length*2 #minimum number of positions, can be extended for larger contexts

#Save config file
with open('config_file/config.json', 'w') as f:
    json.dump(config, f)

decision_transformer_config = DecisionTransformerConfig(**config)

data_suffix = 'RockHoFNope03B'
#data_suffix = 'dec-jan'

with open('data/trajectories_'+data_suffix+'.pkl', 'rb') as f:
    trajectories = pickle.load(f)
with open('data/masks_'+data_suffix+'.pkl', 'rb') as f:
    masks = pickle.load(f)

print(trajectories.shape, masks.shape)

#Send data to tensors on device
states = torch.tensor(trajectories[:, :, :state_dim], dtype=torch.float32).to(device)
actions = torch.tensor(trajectories[:, :, -3], dtype=torch.float32).unsqueeze(-1).to(device) # depends on context, current structure is state, action, reward, return 
rewards = torch.tensor(trajectories[:, :, -2], dtype=torch.float32).unsqueeze(-1).to(device)
returns_to_go = torch.tensor(trajectories[:, :, -1], dtype=torch.float32).unsqueeze(-1).to(device)
attention_mask = torch.tensor(masks[:, :], dtype=torch.float32).to(device) # mask for padding
timesteps = torch.arange(max_length).unsqueeze(0).repeat(trajectories.shape[0], 1).to(device) # time steps for each trajectory

if not evaluation_mode:
    #Define Model and Optimiser 
    model = DecisionTransformerModel(decision_transformer_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    #Create a dataloader for batching the data
    train_loader = DataLoader(TensorDataset(states, actions, rewards, returns_to_go, timesteps, attention_mask), batch_size=batch_size, shuffle=True)

    #Define loss function
    loss_function = nn.MSELoss()

    #Test by getting an element of the dataloader
    for states, actions, rewards, returns_to_go, timesteps, att_mask in train_loader:
        print(states.shape, actions.shape, rewards.shape, returns_to_go.shape, att_mask.shape)
        break

    #Train the model
    best_model = utilities.train_trajectory_transformer(model, optimizer, train_loader, loss_function, num_epochs)
    torch.save(best_model.state_dict(), 'models/best_model_current.pt')

if evaluation_mode:
    #Load the best model
    model = DecisionTransformerModel(decision_transformer_config).to(device)
    model.load_state_dict(torch.load('models/model-2_layer_standardised_b=128.pt'))

    #Build a train loader for evaluation
    train_loader = DataLoader(TensorDataset(states, actions, rewards, returns_to_go, timesteps, attention_mask), batch_size=1, shuffle=False)

    #Evaluate the model
    state, action, reward = utilities.evaluate_trajectory_transformer(model, train_loader, target_return = -2)

    print(state, action, reward)
