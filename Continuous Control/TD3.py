import copy
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import utils
import matplotlib.pyplot as plt
import gym

'''change means differences from DDPG code'''

# Configuration
class Config(object):
    """Object to hold the config requirements for an agent/game"""
    def __init__(self):
        self.algor_name = 'TD3'
        # Env
        self.env_name = None
        self.env_seed = None
        # Buffer
        self.BUFFER_SIZE = None
        self.BATCH_SIZE = None
        self.MINIMAL_SIZE = None
        # Agent
        self.HIDDEN_DIM = None
        self.actor_lr = None
        self.critic_lr = None
        self.tau = None # soft-update factor for target network
        self.gamma = None
        self.model_path = None # model save/load path
        # Noise
        self.OU_Noise = False
        self.mu = None
        self.sigma = None
        self.theta = None
        # Train
        self.episode_num = None
        self.step_num = None
        # Evaluate
        self.is_load = False
        # For TD3
        '''change: extra parameters'''
        self.policy_freq = None # requency of delayed policy updates
        self.policy_noise = None # Noise added to target policy during critic update
        self.noise_clip = None # Range to clip target policy noise

# Actor Network
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.action_bound = action_bound
        self.actorNN = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.actorNN(state) * self.action_bound # constrain the action to requirentment of env


# Critic Network
class QValueNet(nn.Module):
    '''change: another Q-value Network'''
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QValueNet, self).__init__()
        self.crticNN1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.crticNN2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        cat = torch.cat([state, action], dim = 1)
        q1 = self.crticNN1(cat)
        q2 = self.crticNN2(cat)
        return q1, q2

'''cahneg'''
class TD3Agent:
    def __init__(self, STATE_DIM, ACTION_DIM, ACTION_BOUND, config:Config):

        self.ACTION_DIM = ACTION_DIM
        self.ACTION_BOUND = ACTION_BOUND

        # Update
        self.update_count = 0
        self.policy_noise = config.policy_noise
        self.noise_clip = config.noise_clip
        self.policy_freq = config.policy_freq
        self.gamma = config.gamma
        self.tau = config.tau

        # Buffer
        self.replay_buffer = utils.ReplayBuffer(config.BUFFER_SIZE, config.BATCH_SIZE)

        # Noise
        self.Noise_OU = config.OU_Noise
        if config.OU_Noise:
            self.ou_noise = utils.OrnsteinUhlenbeckActionNoise(config.mu, config.sigma, theta=config.theta)
        else:
            self.sigma = config.sigma # Gaussian Noise
            
        self.actor = PolicyNet(STATE_DIM, ACTION_DIM, config.HIDDEN_DIM, ACTION_BOUND) # TODO
        self.critic = QValueNet(STATE_DIM, ACTION_DIM, config.HIDDEN_DIM)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        if config.is_load:
            self.load_pretrained_model(config.model_path)

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action = self.actor(state).item()
        if self.Noise_OU:
            action += self.ou_noise() # add ou_noise
        else:
            action += self.sigma * np.random.randn(self.ACTION_DIM) # add gaussian_noise
        return np.clip(action, -self.ACTION_BOUND, self.ACTION_BOUND)
    
    def update(self):
        b_s, b_a, b_r, b_ns, b_done = self.replay_buffer.get_batch()
        
        # update critic networks
        '''change: Add clipped noise to taeget policy action'''
        action_noise = (torch.randn_like(b_a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        next_action = (self.target_actor(b_ns) + action_noise).clamp(-self.ACTION_BOUND, self.ACTION_BOUND)
        '''change: Choose the smaller q-value to compute td target'''
        next_q_values1, next_q_values2 = self.target_critic(b_ns, next_action)
        next_q_values = torch.min(next_q_values1, next_q_values2)

        target_values = b_r + self.gamma * next_q_values * (1-b_done)
        '''change: Count the both q'''
        current_q_values1, current_q_values2 = self.critic(b_s, b_a)
        critic_loss = F.mse_loss(target_values, current_q_values1) + F.mse_loss(target_values, current_q_values2)

        self.critic_optimizer.zero_grad() # ^_^
        critic_loss.backward()
        self.critic_optimizer.step()
        '''change: Delay policy update'''
        self.update_count += 1
        if self.update_count % self.policy_freq == 0:
            # upadte actor network
            '''change: Choose one critic between two'''
            actor_loss = -self.critic(b_s, self.actor(b_s))[0].mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update target networks
            self.soft_update(self.critic, self.target_critic)
            self.soft_update(self.actor, self.target_actor)

            self.update_count = 0

    
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()): # .parameters()是generator
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def load_pretrained_model(self, model_path):
        self.actor.load_state_dict(torch.load(model_path))

    def save_trained_model(self, model_parameters, model_path):
        torch.save(model_parameters, model_path)


# Evaluation
def test(env_name:str, agent:TD3Agent)->None:
    epsiode_num = 100 # render or not
    env_test = gym.make(env_name)
    #env_test = gym.make(env_name, render_mode='human')
    return_list = []
    for i in range(epsiode_num):
        s, _ = env_test.reset()
        done = False
        return_episode = 0
        for step_i in range(500):
            a = agent.actor(torch.tensor(s, dtype=torch.float32)).detach().numpy()
            s, r, done, _, _ = env_test.step(a)
            return_episode += r
            if done or step_i == 499:
                print(f'episode.{i+1}: {return_episode}')
                return_list.append(return_episode)
                break
    print('================================================')
    print(f"average return on {epsiode_num} episodes of testing: {np.mean(return_list):.2f}")
    print('================================================')


# Train function
def train(env_name:str, agent:TD3Agent, config:Config)->TD3Agent:
    
    # env_train = gym.make(env_name, render_mode='human')
    env_train = gym.make(env_name)

    return_list = []
    for episode_i in range(1, config.episode_num + 1):
        episode_return = 0
        s, _ = env_train.reset(seed=config.env_seed)
        done = False
        for _ in range(config.step_num):
        # while not done:
            a = agent.get_action(s)
            s_, r, done, _, _ = env_train.step(a)
            agent.replay_buffer.add_experience(s, a, r, s_, done)
            s = s_
            episode_return += r

            if agent.replay_buffer.get_size() >= config.MINIMAL_SIZE:
                agent.update()

            if done: break

        return_list.append(episode_return)
        if episode_i % 10 == 0:
            print(f"Episode: {episode_i}, Avg.10_most_recent Return: {np.mean(return_list[-10:]):.2f}")

    agent.save_trained_model(agent.actor.state_dict(), config.model_path)
    # plot trainning curves 
    episodes_list = list(range(len(return_list)))
    mv_return = utils.moving_average(return_list, 9)
    plt.plot(episodes_list, return_list, label="episode_return")
    plt.plot(episodes_list, mv_return, label="mv_episode_return")
    plt.legend(loc=0)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'{config.algor_name} on {env_name}')
    plt.savefig(f"./Figures/{config.algor_name} on {env_name}.png")
    plt.show()

    return agent


if __name__ == "__main__":

    '''choose env'''
    env_name =  'MountainCarContinuous-v0'
    # env_name =  'Pendulum-v1'

    config = Config()
    '''if true: load pretrained model and test; else: train a raw model'''
    config.is_load = True

    if env_name == 'MountainCarContinuous-v0':
        # Env
        config.env_seed = 0
        # Buffer
        config.BUFFER_SIZE = 10000
        config.BATCH_SIZE = 64
        config.MINIMAL_SIZE = 1000
        # Agent
        config.HIDDEN_DIM = 64
        config.actor_lr = 3e-4
        config.critic_lr = 3e-3
        config.tau = 0.005
        config.gamma = 0.9 # 0.98
        config.model_path = "./Models/TD3-MountainCarContinuous.pth" # change if need
        # Noise # 参数参考自: https://github.com/samhiatt/ddpg_agent/tree/master
        config.OU_Noise = True
        config.mu = np.array([0.]) # []
        config.sigma = 0.25
        config.theta = 0.05
        # Train
        config.episode_num = 200
        config.step_num = 500
        '''change'''
        config.policy_noise = 0.2 # Noise added to target policy during critic update
        config.noise_clip = 0.5 # Range to clip target policy noise
        config.policy_freq = 2 # requency of delayed policy updates

    # elif env_name == 'Pendulum-v1':
    #     config.env_seed = 0
    #     # Buffer
    #     config.BUFFER_SIZE = 10000
    #     config.BATCH_SIZE = 64
    #     config.MINIMAL_SIZE = 1000
    #     # Agent
    #     config.HIDDEN_DIM = 64
    #     config.actor_lr = 3e-4
    #     config.critic_lr = 3e-3
    #     config.tau = 0.005
    #     config.gamma = 0.98
    #     config.model_path = "./Models/TD3-Pendulum.pth"
    #     # Noise
    #     config.OU_Noise = False
    #     config.sigma = 0.01 # standard deviation of Gaussian noise whose mean = 0
    #     # Train
    #     config.episode_num = 200
    #     config.step_num = 200
    #     '''change'''
    #     config.policy_noise = 0.1 # Noise added to target policy during critic update
    #     config.policy_freq = 2 # requency of delayed policy updates
    #     config.noise_clip = 0.1 # Range to clip target policy noise
    
    np.random.seed(config.env_seed)
    torch.manual_seed(config.env_seed)

    env = gym.make(env_name)
    STATE_DIM = env.observation_space.shape[0]
    ACTION_DIM = env.action_space.shape[0]
    ACTION_BOUND = env.action_space.high[0]

    agent = TD3Agent(STATE_DIM, ACTION_DIM, ACTION_BOUND, config=config)
    env.close()

    if config.is_load:
        test(env_name, agent) # pretrained
    else:
        agent = train(env_name, agent, config) # untrained
        test(env_name, agent) # trained

'''average return on 100 episodes of testing: 93.90'''