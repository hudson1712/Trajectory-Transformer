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
def read_config(file_path):
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config

config_file_path = 'config_file/config.json'
config = read_config(config_file_path)

decision_transformer_config = DecisionTransformerConfig(**config)

# Define model and data parameters
state_dim = 27  # depends on your specific environment
action_dim = 1  # depends on your specific environment
max_length = 48  # maximum sequence length/context size
hidden_size = 128 # dimension of the hidden embeddings, quadratic in complexity
num_epochs = 10 # training epochs
batch_size = 32 # number of trajectories per batch

with open('trajectories.pkl', 'rb') as f:
    trajectories = pickle.load(f)
with open('masks.pkl', 'rb') as f:
    masks = pickle.load(f)

model = DecisionTransformerModel(decision_transformer_config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(trajectories.shape, masks.shape)

#Send data to tensors on device
states = torch.tensor(trajectories[:, :, :state_dim], dtype=torch.float32).to(device)
actions = torch.tensor(trajectories[:, :, -3], dtype=torch.float32).unsqueeze(-1).to(device)
rewards = torch.tensor(trajectories[:, :, -2], dtype=torch.float32).unsqueeze(-1).to(device)
returns_to_go = torch.tensor(trajectories[:, :, -1], dtype=torch.float32).unsqueeze(-1).to(device)
attention_mask = torch.tensor(masks[:, :], dtype=torch.float32).to(device)
timesteps = torch.arange(max_length).unsqueeze(0).repeat(trajectories.shape[0], 1).to(device)

#Create a dataloader for batching the data
data_loader = DataLoader(TensorDataset(states, actions, rewards, returns_to_go, timesteps, attention_mask), batch_size=batch_size, shuffle=True)

#Test by getting an element of the dataloader
for states, actions, rewards, returns_to_go, timesteps, att_mask in data_loader:
    print(states.shape, actions.shape, rewards.shape, returns_to_go.shape, att_mask.shape)
    break

loss_function = nn.MSELoss()

training_losses = []

for epoch in range(num_epochs):
    model.train()
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
        
        # Compute loss (e.g., mean squared error for continuous actions)
        loss = loss_function(action_preds, actions)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        #print(loss.item())
        
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(data_loader)}')
    training_losses.append(total_loss/len(data_loader))

#Save the model and loss curve to a csv
torch.save(model.state_dict(), 'model_32.pt')
training_losses_df = pd.DataFrame({'training_loss': training_losses})
training_losses_df.to_csv('training_loss_10.csv')