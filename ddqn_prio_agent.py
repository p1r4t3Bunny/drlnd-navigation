from model import QNetwork

from replay_buffer import PrioritizedReplayBuffer 
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd 


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 0.001              # for soft update of target parameters
LR = 0.0005               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

PRIO_ALPHA = 0.6
PRIO_BETA = 0.4


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DdqnPrioAgent():
    """Double Deep Q-Network agent that interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        
        # Replay memory
        self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
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
                experiences, weights = self.memory.sample(beta)

                states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
                actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
                rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
                next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
                dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

                experiences = (states, actions, rewards, next_states, dones)

                weights = torch.from_numpy(np.vstack(weights)).float().to(device)

                self.learn(experiences, GAMMA, weights)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()).item()
        else:
            return random.choice(np.arange(self.action_size)).item()

    def learn(self, experiences, gamma, weights):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
            weights (array_like): list of weights for compensation the non-uniform sampling
        """
        states, actions, rewards, next_states, dones = experiences

        # Evaluate the greedy policy according to the local network
        target_next_indices_local = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).gather(1, target_next_indices_local)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss (weighted MSE)
        td_error = Q_expected - Q_targets
        loss = (td_error) ** 2
            
        loss = loss * weights
        loss = loss.mean()

        self.memory.update_priorities(np.hstack(td_error.detach().cpu().numpy()))

        # Minimize the loss and save updated priorities
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

    def summary(self):
        s = 'DDQN\n'
        s += self.qnetwork_local.__str__()
        s += '\nMemory size: {} \nBatch size: {}\nGamma: {}\nLR: {}\nTau: {}\nUpdate every: {}'.format(BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY)
        return s
