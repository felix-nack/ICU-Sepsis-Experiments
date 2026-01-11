import numpy as np
import gymnasium as gym
import icu_sepsis
import torch
import torch.nn as nn
import random
from collections import namedtuple
from collections import deque

Transition = namedtuple('Transition', ('state', 'action', 'action_mask','reward', 'next_state', 'next_action_mask', 'done'))
TransitionSARSA = namedtuple('TransitionSARSA', ('state', 'action', 'action_mask','reward', 'next_state', 'next_action','next_action_mask', 'done'))

NUM_ACTIONS = 25 
NUM_STATES = 722
NUM_ENVS = 4



def make_env(seed, env_type = None):
    def thunk():
        env = gym.make('Sepsis/ICU-Sepsis-v2')
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def many_hot(indices, vector_length):
    """
    Create a many-hot encoded vector in PyTorch.
    
    :param indices: Tensor of indices to set to 1.
    :param vector_length: Total length of the output vector.
    :return: Many-hot encoded vector.
    """
    vector = torch.zeros(vector_length)
    vector[indices] = 1
    return vector

def get_mask(info, num_envs = 1, n_actions = NUM_ACTIONS):
    allowed_actions = info['admissible_actions']
    masks = np.zeros((num_envs, n_actions))
    for i, allowed_action in enumerate(allowed_actions):
        masks[i] = many_hot(allowed_action, n_actions)
    return torch.Tensor(masks)

def encode_state(obs, n_states = NUM_STATES):
    obs = torch.Tensor(obs)
    return nn.functional.one_hot(obs.long(), n_states).float()

def encode_state_cont(obs, n_states):
    obs = torch.Tensor(obs).float().reshape(-1)
    return obs
    
def layer_init(layer, optimistic=False, bias_const=0.0):
    """Initialize layer weights and bias.
    
    Args:
        layer: Neural network layer to initialize
        optimistic: If True, use Xavier initialization and positive bias for optimistic Q-values
        bias_const: Constant value for bias initialization (used when optimistic=True)
    """
    if optimistic:
        # Use Xavier uniform initialization for better gradient flow
        torch.nn.init.xavier_uniform_(layer.weight)
        # Set positive bias to encourage exploration through optimistic Q-estimates
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, bias_const)
    else:
        # Original initialization: constant zero
        torch.nn.init.constant_(layer.weight, 0.0)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, 0.0)
    return layer



def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)



class ReplayBuffer:
    def __init__(self, capacity, n_step: int = 1, gamma: float = 0.99):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

        self.n_step = int(n_step)
        self.gamma = float(gamma)

        # for n-step return construction
        self.n_step_buffer = deque(maxlen=self.n_step if self.n_step > 0 else 1)

    def _to_scalar(self, x):
        """Convert 0-d/1-element containers to a Python scalar (int/float/bool)."""
        try:
            # torch tensor
            if isinstance(x, torch.Tensor):
                if x.numel() == 1:
                    return x.item()
                return x
        except Exception:
            pass

        try:
            # numpy array
            if isinstance(x, np.ndarray):
                if x.size == 1:
                    return x.item()
                return x
        except Exception:
            pass

        # python list/tuple like [val]
        if isinstance(x, (list, tuple)) and len(x) == 1:
            return x[0]

        return x

    def _push_transition(self, state, action, action_mask, reward, next_state, next_action_mask, done):
        """Push a single (already processed) transition to the main buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = Transition(state, action, action_mask, reward, next_state, next_action_mask, done)
        self.position = (self.position + 1) % self.capacity


    def _get_n_step_info(self):
        """
        From current n_step_buffer, compute:
        R_n, next_state_n, next_action_mask_n, done_n
        """
        R = 0.0
        next_state_n = None
        next_action_mask_n = None
        done_n = False

        for i, trans in enumerate(self.n_step_buffer):
            r_i = float(trans.reward)
            R += (self.gamma ** i) * r_i
            next_state_n = trans.next_state
            next_action_mask_n = trans.next_action_mask
            done_n = bool(trans.done)
            if done_n:
                break

        return R, next_state_n, next_action_mask_n, done_n

    def push(self, *args):
        """
        Saves a transition.
        Expected args order:
        (state, action, action_mask, reward, next_state, next_action_mask, done)
        """
        state, action, action_mask, reward, next_state, next_action_mask, done = args

        # Normalize common 1-element arrays (important for speed and consistency)
        action = self._to_scalar(action)
        reward = self._to_scalar(reward)
        done = self._to_scalar(done)

        # Convert to clean python scalars where possible
        if not isinstance(action, (torch.Tensor, np.ndarray)):
            try:
                action = int(action)
            except Exception:
                pass

        try:
            reward = float(reward)
        except Exception:
            pass

        done = bool(done)

        if self.n_step <= 1:
            self._push_transition(state, action, action_mask, reward, next_state, next_action_mask, done)
            return

        # n-step path
        self.n_step_buffer.append(Transition(state, action, action_mask, reward, next_state, next_action_mask, done))

        # If buffer not full yet and episode not done, wait
        if len(self.n_step_buffer) < self.n_step and not done:
            return

        # If we have enough for one n-step transition, or episode ended: create at least one
        R_n, next_state_n, next_action_mask_n, done_n = self._get_n_step_info()
        first = self.n_step_buffer[0]
        self._push_transition(first.state, first.action, first.action_mask, R_n, next_state_n, next_action_mask_n, done_n)

        # Move window forward by one
        self.n_step_buffer.popleft()

        # If episode ended, flush remaining partial transitions
        if done:
            while len(self.n_step_buffer) > 0:
                R_n, next_state_n, next_action_mask_n, done_n = self._get_n_step_info()
                first = self.n_step_buffer[0]
                self._push_transition(first.state, first.action, first.action_mask, R_n, next_state_n, next_action_mask_n, done_n)
                self.n_step_buffer.popleft()

    def sample(self, batch_size):
        """Samples a batch of experiences."""
        transitions = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*transitions))

        states = torch.stack(batch.state).reshape((batch_size, -1))
        actions = torch.tensor(batch.action, dtype=torch.int64).reshape((batch_size, -1))
        action_masks = torch.stack(batch.action_mask).reshape((batch_size, -1))
        rewards = torch.tensor(batch.reward, dtype=torch.float32).reshape((batch_size, -1))
        next_states = torch.stack(batch.next_state).reshape((batch_size, -1))
        next_action_masks = torch.stack(batch.next_action_mask).reshape((batch_size, -1))
        dones = torch.tensor(batch.done, dtype=torch.float32).reshape((batch_size, -1))

        return Transition(states, actions, action_masks, rewards, next_states, next_action_masks, dones)

    def __len__(self):
        return len(self.buffer)

    

def calculate_discounted_return(reward_list, gamma = 0.99):
    discounted_return = 0
    for reward in reversed(reward_list):
        discounted_return = reward + gamma * discounted_return
    return discounted_return
