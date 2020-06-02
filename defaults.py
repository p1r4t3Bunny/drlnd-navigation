BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 0.001              # for soft update of target parameters
LR = 0.0005               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

PRIO_ALPHA = 0.6
PRIO_BETA = 0.4
PRIO_EPSILON = 1e-5