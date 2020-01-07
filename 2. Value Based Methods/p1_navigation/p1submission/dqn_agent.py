import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

##Config variables
LR = 5e-4                #Learning Rate
BUFFER_SIZE = int(1e5)   #Replay Buffer Size
BATCH_SIZE = 64          #Replay Memory Batch Size
UPDATE_EVERY = 4         #Fixed Qtarget update intervals
GAMMA = 0.99             #Discount factor
TAU = 1e-3               #Target network update rate


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    
    def __init__(self, state_size, action_size, seed):
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.t_step = 0
        
        ##Active Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        
        ##Fixed QTarget Network
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        
        ##Optimiser
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        
        ##Instantiate replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
        ##Initialise time step (every Update_every steps)
            
    
    def act(self, state, eps=0.):
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        ##Eval mode
        self.qnetwork_local.eval()
        
        ##TODO: Is no_grad necessary if we are in eval mode?
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
            
        ##Train mode
        self.qnetwork_local.train()
        
        ##Epsilon greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
        
    def step(self, state, action, reward, next_state, done):
        ##Add experience to memory
        self.memory.add(state, action, reward, next_state, done)
        
        ##Learn every UPDATE_EVERY steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
         
    
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        
        ##Max Q for next state from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        ##Calculate Q Target for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1-dones))
        
        ##Get expected Q values from active model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
    
        ##Calculate loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        ##Back Prop and optim step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        ##Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        
            
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
            
            
class ReplayBuffer:
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
       
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)