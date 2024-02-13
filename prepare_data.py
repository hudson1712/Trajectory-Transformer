import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Optional
from torch import Tensor
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'

day = 24*60*60
week = 7*day
year = (365.2425)*day

def get_dataframe(filename, savepath, sequence_length=48):
    
    #load csv with datetime format and filter the data
    df = pd.read_csv(filename)
    
    df['datetime'] = pd.to_datetime(df['adset_datetime'])
    df = df[(df['datetime'] >= pd.to_datetime('2023-01-01'))]
    #df['hour_of_day']
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols] #reorder the datetime to the first column
    
    #compute useful metrics for dataframe
    df['display_revenue'] = df[['dynamicAllocation_revenue_with_forecast', 
                                'prebid_won_revenue', 'direct_revenue']].apply(sum, axis=1)
    df['total_revenue'] = df['display_revenue'] + df['aniview_adjusted']
    df['profit'] = df['total_revenue'] - df['spend']
    
    #Filter out quizzes and bidcapped campaigns
    df = df[(df['campaign_name'].str.contains('Quiz|quiz|BIDCAP')==False)]
    labels = df['adset_name'].unique()
    
    #Drop irrelevant columns
    df = df.drop(columns=['parent_adset','adset_datetime','fb_spend','ay_revenue',
                          'prebid_won_revenue','dynamicAllocation_revenue_with_forecast',
                          'direct_revenue','video_revenue','video_impressions',
                          'display_revenue','aniview_adjusted','net_video_revenue',
                          'impressions','created_at','updated_at','campaign_name','id'])
    
    df.dropna(inplace=True)
    
    

    for campaign in labels:
        try:
            #Get all campaigns that share the same adset name/slug
            cdf = df[(df["adset_name"] == campaign)]
            
            #compute the start date and get first x days of campaign
            cdf = cdf.sort_values('datetime')
            #cdf.dropna(subset = ['spend'], inplace=True)
            if cdf.empty == True: continue
            start_datetime = cdf.datetime.iloc[0]
            #cdf = cdf[cdf.adset_datetime <= (start_datetime + pd.Timedelta(days=days_to_observe))]
            cdf['hours_since_launch'] = (cdf['datetime'] - start_datetime)
            cdf['time_of_day'] = cdf['datetime'].dt.hour
            #print(cdf['time_of_day'])
            cdf['hours_since_launch'] = cdf['hours_since_launch'].apply(lambda x: x.value) / (10**9 * 3600) #convert to hours
            
            #combine duplicates into one dataset
            cdf = cdf.groupby(['hours_since_launch'], as_index = False)[['profit','spend','total_revenue','budget','time_of_day','ay_impressions','ay_sessions','ay_pageviews']].agg(
                    profit=('profit','sum'), spend=('spend','sum'), total_revenue=('total_revenue','sum'),
                    budget=('budget','sum'), time_of_day=('time_of_day','mean'), ay_impressions=('ay_impressions','sum'),
                    ay_sessions=('ay_sessions','sum'), ay_pageviews=('ay_pageviews','sum'))
            
            #compute cumulative metrics over period
            cdf['c_profit'] =   cdf['profit'].cumsum()/100
            cdf['c_rev'] =      cdf['total_revenue'].cumsum()/100
            cdf['c_spend'] =    cdf['spend'].cumsum()/100
            
            #Consider only the time when a campaign is actively spending
            cdf = cdf[(cdf['spend'] > 0.01)]
            if cdf.empty == True: continue
        
            #Reset the index
            cdf.reset_index(inplace=True, drop=True)
            
            #Pad and then cut the data to have a fixed sequence length
            cdf = cdf.assign(mask=1)
            cdf = cdf.reindex(range(sequence_length), fill_value=0)
            cdf = cdf[:sequence_length]
            
            cdf.to_csv(savepath+campaign)
            
        except KeyboardInterrupt:
            print('stopped')  
            exit(0)
        #except:
            #print('exception')
        
    return df

def prep_data(df, cdf, conversions_savepath=None, savepath=None, description_path='', display_variables=0):
    """
    Prepare the data for training by concatenating the hourly data with the hourly conversions and then normmalising/scaling data.
    Creates columns for actions (% Budget change from previous step) and the rewards (absolute profit).

    Inputs:
        df: dataframe containing hourly data, joined with budget table from SQL database
        cdf: dataframe containing hourly cumulative conversions
    Outputs:
        df: dataframe containing all the data normalised and ready for sequencing
    """
    #Process conversions
    cdf['adset_datetime'] = pd.to_datetime(cdf['hour'])
    cdf.groupby(['campaign_name','adset_datetime'])
    for campaign in cdf['campaign_name'].unique():
        cdf.loc[cdf['campaign_name'] == campaign, 'hourly_conversions'] = cdf.loc[cdf['campaign_name'] == campaign, 'conversions'].diff()
        #if at 00:00:00 set hourly conversions to conversions
        cdf.loc[cdf['adset_datetime'].dt.hour == 0, 'hourly_conversions'] = cdf.loc[cdf['adset_datetime'].dt.hour == 0, 'conversions']
        #Replace NaN and negative values in the hourly conversions with 0
        cdf['hourly_conversions'].fillna(0, inplace=True)
        cdf['hourly_conversions'] = cdf['hourly_conversions'].apply(lambda x: 0 if x < 0 else x)
        cdf.loc[cdf['campaign_name'] == campaign, 'cum_conversions'] = cdf.loc[cdf['campaign_name'] == campaign, 'hourly_conversions'].cumsum()

    cdf.to_csv(conversions_savepath, index=False)

    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', None)
    #print(cdf.groupby(['campaign_name','adset_datetime'])['hourly_conversions'].sum())

    #Drop na values in the campaign name
    df.dropna(subset=['campaign_name'], inplace=True)
    df = df[~df['campaign_name'].str.contains('-A12|test', regex=True)]

    df['budget'] = df['budget.1']
    df.drop(columns=['id','fb_spend','budget.1','created_at','updated_at','id.1','datetime','campaign_name.1','adset_name.1','created_at.1','updated_at.1'], inplace=True)
    df['revenue'] = df['ay_revenue'] + df['net_video_revenue']

    #Group by campaign_name and compute the fractional change in budget between each hour, filling with 1 for the first hour
    df['action_budget_change'] = df.groupby('campaign_name')['budget'].transform(lambda x: x.pct_change().fillna(0))
    #Shift the action_budget_change back by 1 hour and fill with 0 for na
    df['action_budget_change'] = df.groupby('campaign_name')['action_budget_change'].transform(lambda x: x.shift(-1).fillna(0))
    df['reward_profit'] = df['revenue'] - df['spend']

    # Convert adset_datetime to numerical timestamp integer in nanoseconds
    #df['adset_datetime'] = pd.to_datetime(df['adset_datetime'].astype(str).str[:-2], format='%Y-%m-%d %H:%M:%S')
    df['adset_datetime'] = pd.to_datetime(df['adset_datetime'].astype(str), format='%Y-%m-%d %H:%M:%S')
    df = df.merge(cdf, how='left', on=['adset_datetime','campaign_name'])
    df['adset_datetime'] = df['adset_datetime'].astype(np.int64) // 10**9

    #Compute temporal encodings
    df.dropna(subset=['budget', 'spend', 'ay_revenue'], inplace=True)
    df['hour_cos'] = np.cos(df['adset_datetime'] * (2. * np.pi / day))
    df['hour_sin'] = np.sin(df['adset_datetime'] * (2. * np.pi / day))
    df['day_cos'] = np.cos(df['adset_datetime'] * (2. * np.pi / week))
    df['day_sin'] = np.sin(df['adset_datetime'] * (2. * np.pi / week))
    df['month_cos'] = np.cos(df['adset_datetime'] * (2. * np.pi / year))
    df['month_sin'] = np.sin(df['adset_datetime'] * (2. * np.pi / year))

    #Compute additional metrics
    df['CTR'] = df['ay_sessions'] / df['impressions']
    df['CPM'] = df['spend'] / df['impressions']
    df['CPR'] = df['spend'] / df['hourly_conversions']
    df['RPS'] = df['revenue'] / df['ay_sessions']

    #Replace all inf and NaN with 0
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    #From each campaign_name, extract the RPS event, the country code (US, GB or CA), the domain (EK, GS, SM or HH) and the device (if not AND then iOS)
    df['event'] = df['campaign_name'].str.extract(r'(RPS\d+)')
    df['country'] = df['campaign_name'].str.extract(r'(-US|-CA)')
    df['domain'] = df['campaign_name'].str.extract(r'(-EK|-GS|-SM|-HH)')
    df['device'] = df['campaign_name'].str.extract(r'(-AND)')

    #if spend is 0 set the active flag to 0
    df['active'] = df['spend'].apply(lambda x: 0 if x == 0 else 1)

    #Encode the above categorical features
    df = pd.get_dummies(df, columns=['event', 'country', 'domain', 'device'])
    #Convert all boolean columns to 0 and 1
    df = df.replace({True: 1, False: 0})
    #print(df)

    #log transform the numerical features and scale by 10 to bring into range 0,1 (No variables should be greater than ~10^5)
    df['budget'] = np.log10(df['budget']/100+1) / 10
    df['revenue'] = np.log10(df['revenue']+1) / 10
    df['spend'] = np.log10(df['spend']+1) / 10
    df['impressions'] = np.log10(df['impressions']+1) / 10
    df['ay_sessions'] = np.log10(df['ay_sessions']+1) / 10
    df['ay_pageviews'] = np.log10(df['ay_pageviews']+1) / 10
    df['hourly_conversions'] = np.log10(df['hourly_conversions']+1) / 10
    df['cum_conversions'] = np.log10(df['cum_conversions']+1) / 10
    df['reward_profit'] = df['revenue'] - df['spend']

    #Ensure all categorical columns are present, creating them if not
    for col in ['event_RPS3', 'event_RPS5', 'event_RPS8', 'country_-US', 'country_-CA', 'domain_-EK', 'domain_-GS', 'domain_-SM', 'domain_-HH', 'device_-AND']:
        if col not in df.columns:
            df[col] = 0

    #Define the order of state variables and action variables with associated aggregation methods
    agg_dict = {
        'event_RPS3': 'first',
        'event_RPS5': 'first',
        'event_RPS8': 'first',
        'country_-US': 'first',
        'country_-CA': 'first',
        'domain_-EK': 'first',
        'domain_-GS': 'first',
        'domain_-SM': 'first',
        'domain_-HH': 'first',
        'device_-AND': 'first',
        'active': 'first',
        'budget': 'first',
        'cum_conversions': 'sum',
        'revenue': 'sum', 
        'spend': 'sum', 
        'impressions': 'sum', 
        'ay_sessions': 'sum', 
        'ay_pageviews': 'sum', 
        'hourly_conversions': 'sum',
        'CTR': 'mean',
        'CPM': 'mean',
        'CPR': 'mean',
        'RPS': 'mean',
        'hour_cos': 'mean',
        'hour_sin': 'mean',
        'day_cos': 'mean',
        'day_sin': 'mean',
        'month_cos': 'mean',
        'month_sin': 'mean',
        'action_budget_change': 'mean',
        'reward_profit': 'sum'
    }
    
    #Save the data.describe to a csv for normalisation constants
    #df.describe().to_csv('data/'+description_path+'.csv')
    data_describe = pd.read_csv('data/data_describe_dec-jan.csv', index_col=0)

    for col in ['budget', 'cum_conversions', 'revenue', 'spend', 'impressions', 'ay_sessions', 'ay_pageviews', 'hourly_conversions', 'CTR', 'CPM', 'CPR', 'RPS']:
        #Standardise the variables using the described data
        df[col] = (df[col] - data_describe.loc['mean', col]) / data_describe.loc['std', col]

    #create a violin plot of the above metrics
    if display_variables:
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df[['budget', 'action_budget_change', 'reward_profit', 'revenue', 'spend', 'impressions', 'ay_sessions', 'ay_pageviews', 'hourly_conversions', 'CTR', 'CPM', 'RPS', 'CPR', 'cum_conversions']])
        plt.xlabel('Metrics')
        plt.xticks(rotation=90)
        plt.ylabel('Values')
        plt.ylim(-2,2)
        plt.title('Violin Plot of Metrics')
        plt.show()
    
    #Group by campaign_name and adset_datetime and aggregate the above metrics
    df = df.groupby(['campaign_name','adset_datetime']).agg(agg_dict).reset_index()
    #save the data
    df.to_csv(savepath)
    return df

def create_sequences(data, sequence_length=48):
    """
    Create sequences from the input data with optional parameters for sequence length. 
    Returns sequences and masks as numpy arrays.

    Input: dataframe with columns: campaign_name, adset_datetime, [--states--], action_budget_change, reward_profit
    Output: numpy arrays of sequences and masks
    """
    state_columns = [col for col in data.columns if col not in ('Unnamed: 0', 'campaign_name', 'adset_datetime', 'action_budget_change', 'reward_profit')]
    action_column = 'action_budget_change'
    returns_to_go = 'returns_to_go'
    reward_column = 'reward_profit'
    sequences = []
    masks = []
    for _, group in data.groupby('campaign_name'):
        # Extract states, actions and rewards
        states_actions_rewards = group[state_columns + [action_column, reward_column]].values

        #Exclude short sequences from the training data
        if len(group) < 5:
            continue

        returns_to_go = np.flip(np.cumsum(np.flip(group[reward_column].values)))
        #Append the returns to go to the states and actions
        df = np.concatenate((states_actions_rewards, returns_to_go.reshape(-1, 1)), axis=1)
        
        # If the campaign is shorter than the sequence length, pad it and provide a mask
        if len(group) < sequence_length:
            # Pad the states and actions
            pad_length = sequence_length - len(group)
            df_padded = np.pad(df, ((0, pad_length), (0, 0)), 'constant', constant_values=0)

            # Create the mask
            mask = [1] * len(group) + [0] * pad_length  # 1 for actual data, 0 for padding
            sequences.append(df_padded)
            masks.append(mask)
            continue
        
        # If the campaign is longer than the sequence length, create overlapping sequences with full masks
        for start_idx in range(0, len(group) - sequence_length + 1):
            end_idx = start_idx + sequence_length
            sequences.append(df[start_idx:end_idx])
            masks.append([1] * sequence_length)

    return np.array(sequences), np.array(masks)

def create_trajectories_file(hourly_data, conversions_data, savepath_id='1', sequence_length=48, to_csv=False):
    """
    Calculate the states actions and rewards for the transformer model.
    Create trajectories file from the given data and save the sequences to a CSV file and the trajectories to a pickle file.
    """
    #data = pd.read_csv('ash_0712-2901__.csv', index_col=0)
    #data = pd.read_csv('data/ash_0712-2901__.csv', index_col=0)
    data = prep_data(hourly_data, conversions_data, display_variables=True, description_path=savepath_id)
    
    state_columns = [col for col in data.columns if col not in ('Unnamed: 0', 'campaign_name', 'adset_datetime', 'action_budget_change', 'reward_profit')]
    action_column = 'action_budget_change'
    reward_column = 'reward_profit'

    states = data[state_columns]
    actions = data[action_column].values.reshape(-1, 1)  # reshaping for the scaler
    rewards = data[reward_column]

    # Add normalized actions back to the states dataframe for convenience
    states = pd.DataFrame(states, columns=state_columns)
    states[action_column] = actions
    states[reward_column] = rewards

    # Group by campaign and sort by datetime
    data['adset_datetime'] = pd.to_datetime(data['adset_datetime'], unit='s')  # Assuming 'adset_datetime' is a UNIX timestamp
    #grouped_data = data.groupby('campaign_name').apply(lambda x: x.sort_values('adset_datetime'))

    sequences, masks = create_sequences(data, sequence_length=sequence_length)
    
    #print(sequences)
    #Save the sequences to a csv file
    if to_csv:
        with open('data/sequences_'+savepath_id+'.csv', 'w') as f:
            for sequence in sequences:
                np.savetxt(f, sequence, delimiter=',')

    #Save trajectories to a file then import
    with open('data/trajectories_'+savepath_id+'.pkl', 'wb') as f:
        pickle.dump(sequences, f)
    with open('data/masks_'+savepath_id+'.pkl', 'wb') as f:
        pickle.dump(masks, f)

    return sequences, masks