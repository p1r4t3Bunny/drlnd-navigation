import numpy as np
import random
from replay_buffer import ReplayBuffer, NaivePrioritizedReplayBuffer
from defaults import *
from model import QNetwork, DuelingQNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

class DqnAgent():
    """Deep Q-Network agent that interacts with and learns from the environment."""

    def __init__(self, id, state_size, action_size, seed, use_double=False, use_prio=False, use_dueling=False):
        """Initialize an Agent object.
        
        Params
        ======
            id (int): id used to identify the agent
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            double (boolean): Use Double DQN algorithm
            use_prio (boolean): Use Prioritized Experience Replay
            use_dueling (boolean): Use Dueling DQN algorithm
        """
        self.state_size = state_size
        self.action_size = action_size
        self.id = id

        self.use_double = use_double
        self.use_prio = use_prio
        self.use_dueling = use_dueling
        self.seed = random.seed(seed)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Q-Network
        if use_dueling:
            self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed).to(self.device)
            self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed).to(self.device)
        else:
            self.qnetwork_local = QNetwork(state_size, action_size, seed).to(self.device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed).to(self.device)
            
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        
        # Replay memory
        if use_prio:
            self.memory = NaivePrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, PRIO_ALPHA, PRIO_EPSILON)
        else:
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done, beta=PRIO_BETA):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                if self.use_prio:
                    experiences, weights = self.memory.sample(beta)
                    states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
                    actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
                    rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
                    next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
                    dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
                    weights = torch.from_numpy(np.vstack(weights)).float().to(self.device)

                    experiences = (states, actions, rewards, next_states, dones)
                    self.learn(experiences, GAMMA, weights)
                else:
                    experiences = self.memory.sample()

                    states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
                    actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
                    rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
                    next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
                    dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

                    experiences = (states, actions, rewards, next_states, dones)
                    self.learn(experiences, GAMMA)



    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()).item()
        else:
            return random.choice(np.arange(self.action_size)).item()

    def learn(self, experiences, gamma, weights=None):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
            weights (array_like): list of weights for compensation the non-uniform sampling (used only
                                    with prioritized experience replay)
        """
        states, actions, rewards, next_states, dones = experiences

        if self.use_double:
            # Evaluate the greedy policy according to the local network
            target_next_indices_local = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)

            # Get max predicted Q values (for next states) from target model
            Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, target_next_indices_local)
        else:
            # Get max predicted Q values (for next states) from target model
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        if self.use_prio:
            td_error = Q_expected - Q_targets
            loss = (td_error) ** 2
                
            loss = loss * weights
            loss = loss.mean()

            self.memory.update_priorities(np.hstack(td_error.detach().cpu().numpy()))

        else:
            loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def getId(self):
        """ Return the ID of the agent """
        return self.id 

    def summary(self):
        """ Return a brief summary of the agent"""
        s = 'DQN Agent {}: Double: {}, PER: {}, Dueling: {}\n'.format(self.id, self.use_double, self.use_prio, self.use_dueling)
        s += self.qnetwork_local.__str__()
        s += '\nMemory size: {} \nBatch size: {}\nGamma: {}\nLR: {}\nTau: {}\nUpdate every: {}'.format(BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY)
        return s