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

#configuration = DecisionTransformerConfig()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Read config from json config_file/config.json and convert to decision transformer config
config_file_path = 'config_file/config.json'
with open(config_file_path, 'r') as f:
    config = json.load(f)

# Define Model and Data parameters
state_dim = 27  # depends on your specific environment
action_dim = 1  # depends on your specific environment
max_length = 48  # maximum sequence length/context size
hidden_size = 64 # dimension of the hidden embeddings, quadratic in model parameter size
num_heads = 2 # number of attention heads
num_layers = 1 # number of transformer layers

# Define Training parameters
num_epochs = 10 # training epochs
batch_size = 256 # number of trajectories per batch
early_stopping = 1 # set to 1 if you want to stop training when the validation loss stops improving

#Adjust the configuration
config['state_dim'] = state_dim
config['act_dim'] = action_dim
config['max_ep_len'] = max_length
config['hidden_size'] = hidden_size
config['n_layer'] = num_layers
config['n_head'] = num_heads

decision_transformer_config = DecisionTransformerConfig(**config)

with open('data/trajectories_48.pkl', 'rb') as f:
    trajectories = pickle.load(f)
with open('data/masks_48.pkl', 'rb') as f:
    masks = pickle.load(f)

#Define Model and Optimiser 
model = DecisionTransformerModel(decision_transformer_config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

print(trajectories.shape, masks.shape)

#Send data to tensors on device
states = torch.tensor(trajectories[:, :, :state_dim], dtype=torch.float32).to(device)
actions = torch.tensor(trajectories[:, :, -3], dtype=torch.float32).unsqueeze(-1).to(device) # depends on context, current structure is state, action, reward, return 
rewards = torch.tensor(trajectories[:, :, -2], dtype=torch.float32).unsqueeze(-1).to(device)
returns_to_go = torch.tensor(trajectories[:, :, -1], dtype=torch.float32).unsqueeze(-1).to(device)
attention_mask = torch.tensor(masks[:, :], dtype=torch.float32).to(device) # mask for padding
timesteps = torch.arange(max_length).unsqueeze(0).repeat(trajectories.shape[0], 1).to(device) # time steps for each trajectory

#Create a dataloader for batching the data
data_loader = DataLoader(TensorDataset(states, actions, rewards, returns_to_go, timesteps, attention_mask), batch_size=batch_size, shuffle=True)

#Test by getting an element of the dataloader
for states, actions, rewards, returns_to_go, timesteps, att_mask in data_loader:
    print(states.shape, actions.shape, rewards.shape, returns_to_go.shape, att_mask.shape)
    break

#Define loss function
loss_function = nn.MSELoss()

training_losses = []
best_loss = 0

model.train()
for epoch in range(num_epochs):
    total_loss = 0
    i=0
    for states, actions, rewards, returns_to_go, timesteps, att_mask in data_loader:

        # Forward pass
        state_preds, action_preds, return_preds = model(
            states=states, 
            actions=actions, 
            rewards=rewards,
            returns_to_go=returns_to_go, 
            timesteps=timesteps, 
            attention_mask=att_mask,
            return_dict = False
        )

        if i == 100:
            #display the first state and the prediction
            print((states[0][:], state_preds[0][:]))
            print((actions[0][:], action_preds[0][:]))
        
        # Compute loss (e.g., mean squared error for continuous actions)
        loss = loss_function(action_preds, actions)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        i+=1
        
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(data_loader)}')
    training_losses.append(total_loss/len(data_loader))

#Save the model and loss curve to a csv
#Generate unique id for the model
import uuid
model_id = uuid.uuid4().hex
torch.save(model.state_dict(), 'models/model_'+model_id+'.pt')
training_losses_df = pd.DataFrame({'training_loss': training_losses})
training_losses_df.to_csv('data/training_loss-'+model_id+'.csv')