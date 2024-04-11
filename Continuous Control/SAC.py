import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import utils
import matplotlib.pyplot as plt
import gym

# Configuration
class Config(object):
    """Object to hold the config requirements for an agent/game"""
    def __init__(self):
        self.algor_name = 'SAC'
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
        self.model_path = None
        # Train
        self.episode_num = None
        self.step_num = None
        # SAC
        self.alpha = None
        self.target_entropy = None
        self.alpha_lr = None
        self.refactor_reward = False
        # OU Noise
        self.OU_Noise = False
        self.mu = None
        self.sigma = None
        self.theta = None
        # Evaluate
        self.is_load = False

# Actor Network
class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim) # n 个 mu 值
        self.fc_std = nn.Linear(hidden_dim, action_dim) # n 个 std 值
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x)) # 光滑的ReLU，确保std > 0
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # rsample()是重参数化采样 记r.v.: u ~ mu(u|s)
        log_prob = dist.log_prob(normal_sample) # 计算采样点u的概率，取log
        action = torch.tanh(normal_sample) # 采样动作规范到(-1, 1)内, 即 a' = tanh(u) ~ pi'(a'|s)
        # 计算a'的对数概率, 即log(pi'(a'|s))
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-7)
        # a = c * a' ~ pi(a|s)
        action = action * self.action_bound
        # 计算a的对数概率, 即log(pi(a|s))
        log_prob -= torch.log(torch.tensor(self.action_bound)) # 注意下维度
        return action, log_prob

# Critic Network
class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out1 = torch.nn.Linear(hidden_dim, 1)

        self.fc3 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        # two Q networks
        q1 = F.relu(self.fc1(cat))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc_out1(q1)

        q2 = F.relu(self.fc3(cat))
        q2 = F.relu(self.fc4(q2))
        q2 = self.fc_out2(q2)
        return q1, q2

# Agent
class SACAgent:
    ''' 处理连续动作的SAC算法 '''
    def __init__(self, STATE_DIM, ACTION_DIM, ACTION_BOUND, config:Config):
        self.actor = PolicyNet(STATE_DIM, config.HIDDEN_DIM, ACTION_DIM, ACTION_BOUND)
        self.critic = QValueNet(STATE_DIM, config.HIDDEN_DIM, ACTION_DIM)
        self.target_critic = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        if config.is_load:
            self.load_pretrained_model(model_path=config.model_path)

        self.log_alpha = torch.tensor(np.log(config.alpha), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.target_entropy = config.target_entropy
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],lr=config.alpha_lr)

        # Buffer
        self.replay_buffer = utils.ReplayBuffer(config.BUFFER_SIZE, config.BATCH_SIZE)
        
        # Noise
        self.OU_Noise = config.OU_Noise
        if config.OU_Noise:
            self.ou_noise = utils.OrnsteinUhlenbeckActionNoise(config.mu, config.sigma, theta=config.theta)
        
        # Update
        self.gamma = config.gamma
        self.tau = config.tau
        self.refactor_reward = config.refactor_reward

        self.ACTION_BOUND = ACTION_BOUND



    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action = self.actor(state)[0].detach().numpy()
        # for Mountain Car to be aggressive
        if self.OU_Noise:
            action += self.ou_noise() # array([])
            action = np.clip(action, -self.ACTION_BOUND, self.ACTION_BOUND)
        return action
    
    def calc_target(self, rewards, next_states, dones):
        with torch.no_grad():
            next_actions, log_prob = self.actor(next_states)
            entropy = -log_prob
            q1_value, q2_value = self.target_critic(next_states, next_actions)
            next_value = torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy
            td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target
    
    def update(self):
        b_s, b_a, b_r, b_ns, b_done = self.replay_buffer.get_batch()

        # update critic networks
        if self.refactor_reward:
            b_r = (b_r + 8.0) / 8.0 # only for pendulum

        target_values = self.calc_target(b_r, b_ns, b_done)
        current_q_values1, current_q_values2 = self.critic(b_s, b_a)
        critic_loss = F.mse_loss(target_values, current_q_values1) + F.mse_loss(target_values, current_q_values2)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update policy network
        new_action, log_prob = self.actor(b_s)
        entropy = -log_prob
        q_value1, q_value2 = self.critic(b_s, new_action)
        actor_loss = (-self.log_alpha.exp() * entropy - torch.min(q_value1, q_value2)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update alpha
        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # update target networks
        self.soft_update(self.critic, self.target_critic)
    
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def load_pretrained_model(self, model_path):
        self.actor.load_state_dict(torch.load(model_path))

    def save_trained_model(self, model_parameters, model_path):
        torch.save(model_parameters, model_path)


# Evaluate
def test(env_name:str, agent:SACAgent)->None:
    epsiode_num = 100 # render or not
    env_test = gym.make(env_name)
    # env_test = gym.make(env_name, render_mode='human')
    return_list = []
    for i in range(epsiode_num):
        s, _ = env_test.reset()
        done = False
        return_episode = 0
        for step_i in range(1000):
            a = agent.actor(torch.tensor(s, dtype=torch.float32))[0].detach().numpy()
            s, r, done, _, _ = env_test.step(a)
            return_episode += r
            if done or step_i == 999:
                print(f'episode {i+1}: {return_episode}')
                return_list.append(return_episode)
                break
    print('===============================================')
    print(f"average return on {epsiode_num} episodes of testing: {np.mean(return_list):.2f}")
    print('===============================================')


def train(env_name:str, agent:SACAgent, config:Config) -> SACAgent:

    env_train = gym.make(env_name)
    # env_train = gym.make(env_name, render_mode='human')

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
            print("Episode: {}, Avg.10_most_recent Return: {:.2f}".format(episode_i, np.mean(return_list[-10:])))

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
    '''if true: load pretrained model and test; else: train a model from 0'''
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
        config.alpha_lr = 3e-4
        config.tau = 0.005
        config.gamma = 0.9 # 0.98
        config.alpha = 0.01
        config.target_entropy = -1 # -env.action_space.shape[0]
        config.model_path = "./Models/SAC-MountainCarContinuous.pth"
        # Train
        config.episode_num = 200
        config.step_num = 500
        config.refactor_reward = False
        # Noise # 参数参考自: https://github.com/samhiatt/ddpg_agent/tree/master
        config.OU_Noise = True
        config.mu = np.array([0.]) # []
        config.sigma = 0.25
        config.theta = 0.05
 

    elif env_name == 'Pendulum-v1':
        # Env & Algorithm
        config.env_seed = 0
        # Buffer
        config.BUFFER_SIZE = 10000
        config.BATCH_SIZE = 64
        config.MINIMAL_SIZE = 1000
        # Agent
        config.HIDDEN_DIM = 64
        config.actor_lr = 3e-4
        config.critic_lr = 3e-3
        3e-4
        config.tau = 0.005
        config.gamma = 0.99
        config.alpha = 0.01
        config.target_entropy = -1
        config.model_path = "./Models/SAC-Pendulum.pth"
        # Train
        config.episode_num = 200
        config.step_num = 200
        config.refactor_reward = True
        # Noise 
        config.Noise = False
    
    np.random.seed(config.env_seed)
    torch.manual_seed(config.env_seed)

    env = gym.make(env_name)
    STATE_DIM = env.observation_space.shape[0]
    ACTION_DIM = env.action_space.shape[0]
    ACTION_BOUND = env.action_space.high[0]

    agent = SACAgent(STATE_DIM, ACTION_DIM, ACTION_BOUND, config=config)
    env.close()

    if config.is_load:
        test(env_name, agent) # pretrained
    else:
        agent = train(env_name, agent, config) # untrained
        test(env_name, agent) # trained

'''average return on 10 episodes of testing: 93.42'''

