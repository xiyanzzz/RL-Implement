import gym
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import utils

'''The main reference for the implementation is: https://github.com/mandrakedrink/PPO-pytorch/tree/master '''

class Config(object):
    """Object to hold the config requirements for an agent/game"""
    def __init__(self):
        self.algor_name = 'PPO'
        # Env
        self.env_name = None
        self.env_seed =  None
        # Agent
        self.HIDDEN_DIM_A = None
        self.HIDDEN_DIM_C = None
        self.actor_lr = None
        self.critic_lr = None
        self.gamma = None
        self.lambda_ = None
        self.entropy_coef = None
        self.epsilon =  None
        self.epochs = None
        self.batch_size = None
        self.model_path = None
        # Train
        self.max_rollout_num = None # 50000 # 若solved_reward不为空, 则可以尽可能大，只要满足条件会退出训练
        self.rollout_len = None # 1000个时间步，能跑多少回合跑多少回合
        self.solved_reward = None # 10步左右就能到达，但速度不够快
        self.min_completed_episode_num =  None
        # Evaluate
        self.is_load = False



'''网络结构: 隐藏层数不同, std限制在了(1/e, e)内, 加了权重初始化'''
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound, hidden_dim=32):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.action_bound * torch.tanh(self.fc_mu(x))
        log_std = torch.tanh(self.fc_std(x))

        std = torch.exp(log_std)
        dist = Normal(mu, std)
        action = dist.sample()
        return action, dist
    
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
 
class PPOAgent:
    def __init__(self, STATE_DIM, ACTION_DIM, ACTION_BOUND, config:Config):
        '''初始化'''
        self.actor = PolicyNet(STATE_DIM, ACTION_DIM, ACTION_BOUND, hidden_dim=config.HIDDEN_DIM_A).apply(utils.init_weights)
        self.critic = ValueNet(STATE_DIM, hidden_dim=config.HIDDEN_DIM_C).apply(utils.init_weights)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        if config.is_load:
            self.load_pretrained_model(model_path=config.model_path)

        self.gamma = config.gamma
        self.lambda_ = config.lambda_

        # self.memory = Memory()
        self.epochs = config.epochs
        # self.batch_size = config.batch_size
        self.entropy_coef = config.entropy_coef
        self.epsilon = config.epsilon
    
        self.action_bound = ACTION_BOUND

    def get_action(self, state): # TODO
        state = torch.tensor(state, dtype=torch.float)
        action, dist = self.actor(state)
        return action.detach().numpy(), dist

    
    def update(self, memory):
        actor_losses, critic_losses = [], []

        states, actions, next_states, rewards, masks = memory.to_tensor()

        # 1.估计优势函数 - GAE
        td_target = rewards + self.gamma * self.critic(next_states) * masks
        td_delta = td_target - self.critic(states)
        advantages = utils.compute_advantage(td_delta, self.gamma, self.lambda_)

        # 2. 存档theta_old, 即ratio的分母取对数
        action_dists =  self.actor(states)[1]
        old_log_probs = action_dists.log_prob(actions).detach() # detach()很重要，因为目标参数只能是theta'

        old_td_targets = td_target.detach()

        # 3. 以theta_old为基础, 多次更新
        for i_epoche in range(self.epochs):
            # print(i_epoche)

            # 4. tehta'，即ratio的分子取对数
            action_dists =  self.actor(states)[1]
            cur_log_probs = action_dists.log_prob(actions)

            # 5. 计算ratio
            ratio = torch.exp(cur_log_probs - old_log_probs) # 第一次循环必定为1

            # compute entropy
            entropys = action_dists.entropy().mean() # 熵正则项, 该技巧适用于多种算法

            # 6. 计算actor损失L(theta'): 截断, 比较取min
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.mean(torch.min(surr1, surr2)) - entropys * self.entropy_coef

            # 7. 计算critic损失
            cur_values = self.critic(states)
            critic_loss = torch.mean(F.mse_loss(cur_values , old_td_targets))

            # 8. 更新
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            actor_loss.backward()
            critic_loss.backward()

            self.actor_optimizer.step()
            self.critic_optimizer.step()

            # log
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        return np.mean(actor_losses), np.mean(critic_losses)
    
    def load_pretrained_model(self, model_path):
        self.actor.load_state_dict(torch.load(model_path))

    def save_trained_model(self, model_parameters, model_path):
        torch.save(model_parameters, model_path)

  
# Evaluate
def test(env_name:str, agent:PPOAgent)->None:
    epsiode_num = 100 # render or not
    env_test = gym.make(env_name)
    # env_test = gym.make(env_name, render_mode='human')
    return_list = []
    for i in range(epsiode_num):
        s, _ = env_test.reset()
        done = False
        return_episode = 0
        for step_i in range(1000):
            a = agent.actor(torch.tensor(s, dtype=torch.float32))[0].numpy()
            s, r, done, _, _ = env_test.step(a)
            return_episode += r
            if done or step_i == 999:
                print(f'episode {i+1}: {return_episode}')
                return_list.append(return_episode)
                break
    print('===============================================')
    print(f"average return on {epsiode_num} episodes of testing: {np.mean(return_list):.2f}")
    print('===============================================')


def train(env_name:str, agent:PPOAgent, config:Config) -> PPOAgent:

    env_train = gym.make(env_name)
    # env_train = gym.make(env_name, render_mode='human')

    s, _ = env_train.reset(seed=config.env_seed)
    done = False
    
    memory = utils.Memory()

    return_list = []
    actor_losses_list = []
    critic_losses_list = []
    episode_return = 0
    for _ in range(config.max_rollout_num):
        '''Do not reset until done'''
        for _ in range(config.rollout_len):
            '''很好地避免了on-policy对过往稀少成功经验的丢弃浪费'''
            a, _ = agent.get_action(s)
            s_, r, done, _, _ = env_train.step(a)

            memory.states.append(s)
            memory.actions.append(a)
            memory.next_states.append(s_)
            memory.rewards.append([r])
            memory.masks.append([1-done])

            s = s_
            episode_return += r
            if done:
                return_list.append(episode_return)
                episode_return = 0
                # reset env
                s, _ = env_train.reset(seed=config.env_seed)

        if config.solved_reward is not None:
            '''主动break, 也能保存一个不错的的agent参数, 并限制回合数须大于等于500回合(否则10回合就能solved但效果不是很好)'''
            if len(return_list) >= config.min_completed_episode_num and np.sum(return_list[-10:]) > config.solved_reward:
                print("Congratulations, it's solved! ^_^")
                break
        # update
        actor_losses, critic_losses = agent.update(memory)

        actor_losses_list.append(actor_losses)
        critic_losses_list.append(critic_losses)

        memory.clear_memory()

        
        # 打印训练信息
        completed_episode_num = len(return_list)
        if  completed_episode_num % 10 == 0 and completed_episode_num >= 10:
            print(f"Episode: {completed_episode_num}, Avg.10_most_recent Return: {np.mean(return_list[-10:]):.2f}")

    agent.save_trained_model(agent.actor.state_dict(), config.model_path)
    # plot trainning curves
    episodes_list = list(range(len(return_list)))
    mv_return = utils.moving_average(return_list, 9)
    plt.plot(episodes_list, return_list, label="abs_episode_return") # 和之前算法不同，这里的episode是绝对的，即一定done
    plt.plot(episodes_list, mv_return, label="mv_abs_episode_return")
    plt.legend(loc=0)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'{config.algor_name} on {env_name}')
    plt.savefig(f"./Figures/{config.algor_name} on {env_name}.png")
    plt.show()

    rollout_list = list(range(len(actor_losses_list)))
    f, (ax1, ax2) = plt.subplots(2,1,sharex='col')
    ax1.plot(rollout_list, actor_losses_list)
    ax1.set_title('actor loss')
    ax2.plot(rollout_list, critic_losses_list)
    ax2.set_title('critic loss')
    plt.savefig(f"./Figures/{config.algor_name}_on_{env_name}_loss_curves.png")
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
        config.env_name = "MountainCarContinuous-v0"
        config.env_seed = 0 #555
        # Agent
        config.HIDDEN_DIM_A = 32
        config.HIDDEN_DIM_C = 64
        config.actor_lr = 1e-3
        config.critic_lr = 5e-3
        config.gamma = 0.95 # key 0.99/0.95 0.95/0.98
        config.lambda_ = 0.98
        config.entropy_coef = 0.01 # 0.003
        config.epsilon = 0.2
        config.epochs = 64
        config.batch_size = 1000
        config.model_path = "./Models/PPO-MountainCarContinuous.pth"
        # Train
        config.max_rollout_num = 10000 # 50000 # 若solved_reward不为空, 则可以尽可能大，只要满足条件会退出训练
        config.rollout_len = 1000 # 1000个时间步，能跑多少回合跑多少回合
        config.solved_reward = 92
        config.min_completed_episode_num = 500


    # elif env_name == 'Pendulum-v1':

    
    np.random.seed(config.env_seed)
    torch.manual_seed(config.env_seed)

    env = gym.make(env_name)
    STATE_DIM = env.observation_space.shape[0]
    ACTION_DIM = env.action_space.shape[0]
    ACTION_BOUND = env.action_space.high[0]

    agent = PPOAgent(STATE_DIM, ACTION_DIM, ACTION_BOUND, config=config)
    env.close()

    if config.is_load:
        test(env_name, agent) # pretrained
    else:
        agent = train(env_name, agent, config) # untrained
        test(env_name, agent) # trained

'''average return on 100 episodes of testing: 92.05'''



