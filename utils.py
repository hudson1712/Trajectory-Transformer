import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import uuid

def train_trajectory_transformer(model, optimizer, train_loader, loss_function, num_epochs):
    training_losses = []
    best_loss = np.inf
    best_model = None
    
    model_id = uuid.uuid4().hex
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        i=0
        for states, actions, rewards, returns_to_go, timesteps, att_mask in train_loader:

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
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader)}')
        training_losses.append(total_loss/len(train_loader))
        if total_loss <= best_loss:
            best_loss = total_loss
            best_model = model
            torch.save(best_model.state_dict(), 'models/best_model'+model_id+'.pt')
    
    #torch.save(model.state_dict(), 'models/model_'+model_id+'.pt')
    training_losses_df = pd.DataFrame({'training_loss': training_losses})
    training_losses_df.to_csv('loss/training_loss-'+model_id+'.csv')
    print(f'Best loss: {best_loss} in epoch {training_losses.index(best_loss)}')
    return best_model