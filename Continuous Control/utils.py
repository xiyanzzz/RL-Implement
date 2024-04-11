import torch
import collections
import random
import numpy as np
import torch.nn as nn


class ReplayBuffer:
    def __init__(self, BUFFER_SIZE, BATCH_SIZE):
        self.buffer_size = BUFFER_SIZE
        self.batch_size = BATCH_SIZE
        self.buffer = collections.deque(maxlen=BUFFER_SIZE)

    def add_experience(self, *experience):
        self.buffer.append(experience) # tuple: (state, action, reward, next_state, done)
        '''
        (np.ndarray(2,).float,  np.ndarray(1,).float, float, np.ndarray(2,).float, bool)
        '''
    
    def get_batch(self):
        transitions = random.sample(self.buffer, self.batch_size)
        batch_s, batch_a, batch_r, batch_s_, batch_done = zip(*transitions)

        batch_s_tensor = torch.tensor(batch_s, dtype=torch.float32)
        batch_a_tensor = torch.tensor(batch_a, dtype=torch.float32)
        batch_r_tensor = torch.tensor(batch_r, dtype=torch.float32).unsqueeze(-1)
        batch_s__tensor = torch.tensor(batch_s_, dtype=torch.float32)
        batch_done_tensor = torch.tensor(batch_done, dtype=torch.float32).unsqueeze(-1)
        
        return batch_s_tensor, batch_a_tensor, batch_r_tensor, batch_s__tensor, batch_done_tensor
    
    def get_size(self):
        return len(self.buffer)


class OrnsteinUhlenbeckActionNoise:
    '''reference: https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py#L31'''
    def __init__(self, mu, sigma, theta=.15, dt=1, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


def moving_average(a, window_size):
    '''Smoothing training curves'''
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def init_weights(m):
    if type(m) in (nn.Linear, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, np.sqrt(float(2)))
        if m.bias is not None:
            m.bias.data.fill_(0)


class Memory:
    """Storing the memory of the trajectory (s, a, r ...)."""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.next_states = []

    def to_tensor(self):
        states = torch.tensor(self.states, dtype=torch.float)
        actions = torch.tensor(self.actions, dtype=torch.float)
        next_states = torch.tensor(self.next_states, dtype=torch.float)
        rewards = torch.tensor(self.rewards, dtype=torch.float)
        masks = torch.tensor(self.masks, dtype=torch.float)
        return states, actions, next_states, rewards, masks

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.next_states = []


def compute_advantage(td_delta, gamma, lambda_):
    '''GAE'''
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lambda_ * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


def trajectories_data_generator(states, actions, returns, log_probs, values, advantages, batch_size, num_epochs):
    """data-generator."""
    data_len = states.size(0) # data_len == rollout_len
    for _ in range(num_epochs):
        for _ in range(data_len // batch_size):
            ids = np.random.choice(data_len, batch_size)
            yield states[ids], actions[ids], returns[ids], log_probs[ids], values[ids], advantages[ids]



    