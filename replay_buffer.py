import numpy as np

from collections import namedtuple, deque

PRIO_ALPHA = 0.6
PRIO_BETA = 0.4
PRIO_EPSILON = 1e-5

class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, alpha=PRIO_ALPHA):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        np.random.seed(seed)
        self.alpha = alpha
        self.indices = []
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""

        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        if len(self.priorities) > 0:
            max_prio = max(self.priorities)
            self.priorities.append(max_prio)
        else:
            self.priorities.append(1.0)

        assert(len(self.memory)==len(self.priorities))
    
    def sample(self, beta=PRIO_BETA):
        """Sample a batch of experiences from memory according to the priorities"""

        N = len(self.priorities)
        # sample the indices
        probs = np.array(self.priorities) ** self.alpha
        probs /= sum(probs)
        self.indices = np.random.choice(range(len(self.priorities)), size=self.batch_size, p=probs)
        weights  = (N * probs[self.indices]) ** (-beta)
        weights /= max(weights)
        return ([self.memory[i] for i in self.indices], weights)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def update_priorities(self, priorities):
        """Update the priorities of the selected experiences"""
        for i,p in zip(self.indices, priorities):
            self.priorities[i] = abs(p) + PRIO_EPSILON