#!/usr/bin/env python
# coding: utf-8

# Multinomial HMM

# In[1]:


import numpy as np
from hmmlearn import hmm
from matplotlib import cm, pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import pandas as pd
from pandas import DataFrame, Series


# In[2]:


import seaborn as sns


# In[3]:


import sys
import re


# In[4]:


weatherStateMap   = {'sunny' : 0, 'rainy' : 1, 'foggy' : 2}
weatherStateIndex = {0 : 'sunny', 1 : 'rainy', 2 : 'foggy'}


# In[5]:


# observation map
weatherObsMap   = {'no' : 0, 'yes' : 1}
weatherObsIndex = {0 : 'no', 1 : 'yes'}


# In[6]:


# prior probability on weather states
# P(sunny) = 0.5  P(rainy) = 0.25  P(foggy) = 0.25
weatherProb = [0.5, 0.25, 0.25]


# In[7]:


# transition probabilities
#                    tomorrrow
#    today     sunny  rainy  foggy
#    sunny      0.8    0.05   0.15
#    rainy      0.2    0.6    0.2 
#    foggy      0.2    0.3    0.5
weatherTProb = [ [0.8, 0.05, 0.15], [0.2, 0.6, 0.2], [0.2, 0.3, 0.5] ]


# In[8]:


# conditional probabilities of evidence (observations) given weather
#                          sunny  rainy  foggy 
# P(umbrella=no|weather)    0.9    0.2    0.7
# P(umbrella=yes|weather)   0.1    0.8    0.3
weatherEProb = [ [0.9, 0.2, 0.7], [0.1, 0.8, 0.3] ]


# In[9]:


# Get markov edges
def get_markov_edges(df):
    # Create a dictionary
    edges = {}
    # Loop columns
    for column in df.columns:
        # Loop rows
        for row in df.index:
            edges[(row,column)] = df.loc[row,column]
    # Return edges
    return edges
# Viterbi algorithm for shortest path
# https://github.com/alexsosn/MarslandMLAlgo/blob/master/Ch16/HMM.py
def viterbi(pi, a, b, obs):
    
    nStates = np.shape(b)[0]
    T = np.shape(obs)[0]
    
    path = np.zeros(T)
    delta = np.zeros((nStates, T))
    phi = np.zeros((nStates, T))
    
    delta[:, 0] = pi * b[:, obs[0]]
    phi[:, 0] = 0
    for t in range(1, T):
        for s in range(nStates):
            delta[s, t] = np.max(delta[:, t-1] * a[:, s]) * b[s, obs[t]] 
            phi[s, t] = np.argmax(delta[:, t-1] * a[:, s])
    
    path[T-1] = np.argmax(delta[:, T-1])
    for t in range(T-2, -1, -1):
        path[t] = phi[int(path[t+1]),int(t+1)]
        
    return path, delta, phi
# The main entry point for this module
def main():
    # Observation states
    # The director can have an umbrella or not have an umbrella (equally likely)
    observation_states = ['Umbrella', 'No umbrella']
    # Create hidden states with probabilities (250 rainy days per year)
    p = [0.32, 0.68]
    #p = [0.5, 0.5]
    #p = [0.7, 0.3]
    hidden_states = ['Sunny', 'Rainy']
    state_space = pd.Series(p, index=hidden_states, name='states')
    # Print hidden states
    print('--- Hidden states ---')
    print(state_space)
    print()
    # Create a hidden states transition matrix with probabilities
    hidden_df = pd.DataFrame(columns=hidden_states, index=hidden_states)
    hidden_df.loc[hidden_states[0]] = [0.75, 0.25]
    hidden_df.loc[hidden_states[1]] = [0.25, 0.75]
    # Print transition matrix
    print('--- Transition matrix for hidden states ---')
    print(hidden_df)
    print()
    print(hidden_df.sum(axis=1))
    print()
    # Create matrix of observations with sensor probabilities
    observations_df = pd.DataFrame(columns=observation_states, index=hidden_states)
    observations_df.loc[hidden_states[0]] = [0.1, 0.9]
    observations_df.loc[hidden_states[1]] = [0.8, 0.2]
    # Print observation matrix
    print('--- Sensor matrix ---')
    print(observations_df)
    print()
    print(observations_df.sum(axis=1))
    print()
    # Create graph edges and weights
    hidden_edges = get_markov_edges(hidden_df)
    observation_edges = get_markov_edges(observations_df)
    # Print edges
    print('--- Hidden edges ---')
    print(hidden_edges)
    print()
    print('--- Sensor edges ---')
    print(observation_edges)
    print()
    # Observations
    observations_map = {0:'Umbrella', 1:'No umbrella'}
    observations = np.array([1,1,1,0,1,1,1,0,0,0])
    observerations_path = [observations_map[v] for v in list(observations)]
    # Get predictions with the viterbi algorithm
    path, delta, phi = viterbi(p, hidden_df.values, observations_df.values, observations)
    state_map = {0:'Sunny', 1:'Rainy'}
    state_path = [state_map[v] for v in path]
    state_delta = np.amax(delta, axis=0)
    # Print predictions
    print('--- Predictions ---')
    print(pd.DataFrame().assign(Observation=observerations_path).assign(Prediction=state_path).assign(Delta=state_delta))
    print()
# Tell python to run main method
if __name__ == "__main__": main()


# In[10]:


# reading in data 
state_seq = []
observation_seq = []
f = open('weather-test1-1000.txt','r')
line = f.readline()
while line:
    line=line.rstrip().split(',')
    # encoding: sunny:0, rainy:1, foggy:2
    if line[0] == 'sunny':
        state_seq.append(0)
    elif line[0] == 'rainy':
        state_seq.append(1)
    else:
        state_seq.append(2)
    # encoding: yes:0, no:1
    if line[1] == 'yes':
        observation_seq.append(0)
    else:
        observation_seq.append(1)
    line = f.readline()


# In[11]:


# building a data frame 
row_names = ['weather','observations']
dats = list(zip(state_seq,observation_seq))
df = pd.DataFrame(dats , columns = row_names)
df['obv_no'] = range(1, len(df) + 1)
df.set_index('obv_no')


# In[12]:


df.dtypes


# In[13]:


# visualising the data 
plt.figure(figsize=(16, 30))
g = sns.displot(data=df, x="weather", kind="hist", height = 7)


# In[14]:


plt.figure(figsize=(16, 30))
g = sns.displot(data=df, x="observations", kind="hist", height = 7)


# In[15]:


plt.figure(figsize=(16, 30))
g = sns.displot(data= df , x="weather",y = 'observations', kind="kde", height = 7)

