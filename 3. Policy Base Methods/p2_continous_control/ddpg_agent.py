import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  #reply buffer size
BATCH_SIZE = 512        #minibatch_size
GAMMA = 0.99            #discount factor
TAU = 1e-3              #soft update mix rate
LR_ACTOR = 1e-4         #actor learning rate
LR_CRITIC = 1e-3        #critic learning rate
WEIGHT_DECAY = 0        #L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, state_size, action_size, random_seed):
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        
        ##Actor Networks
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr = LR_ACTOR)
        
        ##Critic Networks        
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr = LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        ##Noise Process
        self.noise = OUNoise(action_size, random_seed)
        
        ##Replay Memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
        
    def step(self, state, action, reward, next_state, done):
        ##add memory to replay buffer
        self.memory.add(state, action, reward, next_state, done)
        
        ##Learn if we have sufficient memories
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)
            
            
    def act(self, state, eps, add_noise=True):
        ##Move state to device
        state = torch.from_numpy(state).float().to(device)
        
        ##Run actor forward and get action
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
            
        ###Add noise if desired with dampening
        if add_noise:
            action += self.noise.sample()*eps
           
        ##Epsilon greedy exploration
        #if random.random() < eps:
        #    action = (np.random.random(self.action_size) * 2) - 1
            
        ##Clip action values
        return np.clip(action, -1, 1)
    
    
    def reset(self):
        self.noise.reset()
    
    
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        
        
        ##Update Critic
        ##Get next_state targets
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        
        ##Calculate state targets
        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))
        
        ##Compute loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        ## Minimise loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        
        ##Update Actor
        ##Compute actions and loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        ##Minimise loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        
        ##Update target Networks
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)
        
        
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)
        
        
        
class OUNoise:
    def __init__(self, size, seed, mu=0, theta=0.15, sigma=0.2):
        self.mu = mu*np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()
        
    def reset(self):
        self.state = copy.copy(self.mu)
        
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
    
    
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
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        