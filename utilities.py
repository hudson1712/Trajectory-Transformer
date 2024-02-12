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
            # Compute loss (e.g., mean squared error for continuous actions)
            loss = loss_function(action_preds, actions)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            i+=1
        
        total_loss /= len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss}')
        training_losses.append(total_loss)
        if total_loss <= best_loss:
            best_loss = total_loss
            best_model = model
            torch.save(best_model.state_dict(), 'models/model-'+model_id+'.pt')
    
    #torch.save(model.state_dict(), 'models/model_'+model_id+'.pt')
    training_losses_df = pd.DataFrame({'training_loss': training_losses})
    training_losses_df.to_csv('loss/training_loss-'+model_id+'.csv')
    print(f'Best loss: {best_loss} in epoch {training_losses.index(best_loss)}')
    return best_model

def evaluate_trajectory_transformer(model, test_loader, target_return = 10e6):
    
    # R , s , a , t , done = [ target_return ] , [ env . reset ()] , [] , [1] , False
    # while not done : # autoregressive generation / sampling
    # # sample next action
    # action = DecisionTransformer (R , s , a , t )[ -1] # for cts actions
    # new_s , r , done , _ = env.step ( action )
    # # append new tokens to sequence
    # R = R + [ R [ -1] - r] # decrement returns -to -go with reward
    # s , a , t = s + [ new_s ] , a + [ action ] , t + [ len ( R )]
    # R , s , a , t = R [ - K :] , ... # only keep context length of K

    # Implement the above psuedocode from the paper:
    model.eval()
    with torch.no_grad():
        for states, actions, rewards, returns_to_go, timesteps, att_mask in test_loader:
            #Adjust the mask to be all ones
            #att_mask = torch.ones_like(att_mask)

            #Add the target return to every element of the rewards to go
            returns_to_go = returns_to_go + target_return
            print(returns_to_go)

            state_preds, action_preds, return_preds = model(
                    states=states, 
                    actions=actions, 
                    rewards=rewards,
                    returns_to_go=returns_to_go,
                    timesteps=timesteps, 
                    attention_mask=att_mask,
                    return_dict = False
                )

    return state_preds, action_preds, return_preds
