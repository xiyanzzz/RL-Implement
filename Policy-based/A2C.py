
import torch
from torch import nn 
from torch.distributions import Categorical
import numpy as np
import gym
import matplotlib.pyplot as plt
# from tqdm import tqdm
import copy

class ActorNN(nn.Module):
    def __init__(self, input_dim, output_dim, hiden_layer = 256):
        super(ActorNN, self).__init__()
        self.actorNN = nn.Sequential(
            nn.Linear(input_dim, hiden_layer),
            nn.ReLU(),
            nn.Linear(hiden_layer, output_dim),
            nn.Softmax(dim=-1)
        )
    def forward(self, state):
        dist = Categorical(self.actorNN(state))
        return dist

class CriticNN(nn.Module):
    def __init__(self, input_dim, output_dim, hiden_layer = 256):
        super(CriticNN, self).__init__()
        self.criticNN = nn.Sequential(
            nn.Linear(input_dim, hiden_layer),
            nn.ReLU(),
            nn.Linear(hiden_layer, 1)
        )
    def forward(self, state):
        value = self.criticNN(state)

        return value
    
    
class Agent:
    def __init__(self, input_dim, output_dim, device):
        self.device = device
        #==========================#
        self.gamma = torch.tensor(0.99).float().to(self.device)
        self.lr_actor = torch.tensor(3e-4).float().to(self.device)
        self.lr_critic = torch.tensor(5e-4).float().to(self.device)
        #==========================#
        self.actor = ActorNN(input_dim, output_dim).to(self.device)
        self.critic = CriticNN(input_dim, output_dim).to(self.device)
        self.target = copy.deepcopy(self.critic)

        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)

    # 加载预训练模型
    def load_pretrained_model(self, model_path="./Models/cartpole-A2C_multi-step.pth"):
        self.actor.load_state_dict(torch.load(model_path))
    # 保存模型参数
    def save_trained_model(self, model_parameters, model_path="./Models/cartpole-multi-step.pth"):
        torch.save(model_parameters, model_path)

def get_default_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def test(agent, device, is_sample = True):
    env_test = gym.make("CartPole-v1", render_mode="human") # gym>=0.25.0
    s, _ = env_test.reset()
    done = False
    while not done:
        dist = agent.actor(torch.as_tensor(s, dtype=torch.float32).to(device))
        if is_sample:
            a = dist.sample().item()
        else:
            a = torch.argmax(dist.probs).item()
        s, _, done, _, _ = env_test.step(a)
        # env_test.render() # 不需要手动render
    env_test.close()


def train(env_train, input_dim, output_dim, is_test, device):

    agent = Agent(input_dim=input_dim, output_dim=output_dim, device=device)

    #=============================#
    n_episodes = 5000 # 2000
    PRINT_FREQUENCY = 20 # 20
    UPDATE_FREQUENCY = 20 # 10
    TARGET_UPDATE_FREQUENCY = 50 # 50
    #=============================#

    reward_episode_list = []
    reward_ep20avg_list = []
    best_model_parameters = copy.deepcopy(agent.actor.state_dict())
    max_reward = 0
    update_count = 0
    for episode_i in range(1, n_episodes+1):
        s, _ = env_train.reset()
        done = False
        step_i = 0
        #episode_reward = 0
        entropy = 0
        
        reward_list = []
        reward_update = []
        while not done:
            step_i += 1
            s = torch.as_tensor(s, dtype=torch.float32).to(device)
            if not reward_update: s_0 = s

            # 前向，获取动作
            dist = agent.actor(s)
            a = dist.sample() # <Categorical>对象，dist拥有很多<Attributes> 如dist.probs查看概率 / a为tensor([*])
            if not reward_update: a_0 = a
            # interact
            s_, r, done, _ , _ = env_train.step(a.item())

            r = torch.tensor([r],dtype=torch.float).unsqueeze(-1).to(device)
            reward_update.append(r)
            s = s_
            # if step_i >= 2000: done = True # 手动结束

            if step_i % UPDATE_FREQUENCY == 0 or done:
                reward_list.extend(reward_update)
                reward_till_step_i = torch.cat(reward_list).detach().sum().cpu()

                # 保存模型参数
                if not done and reward_till_step_i > max(max_reward, 10000):
                    best_model_parameters = copy.deepcopy(agent.actor.state_dict())
                    max_reward = reward_till_step_i
                    # 主动中断
                    if reward_till_step_i > 120000:
                        agent.save_trained_model(best_model_parameters)
                        print("Endless episode may occur. Model is saved and train is terminated!")
                        return

                # 更新
                value = agent.critic(s_0)
                next_value = agent.target(torch.as_tensor(s_, dtype=torch.float32).to(device))
                log_prob = dist.log_prob(a_0).unsqueeze(-1)

                return_k_step = [(agent.gamma**i) * num for i,num in enumerate(reward_update)] # 为每一步奖励加折扣
                return_k_step = torch.cat(return_k_step)
                mask = torch.tensor([1-done],dtype=torch.float).unsqueeze(-1).to(device)
                advantage = return_k_step.sum() + agent.gamma**len(reward_update) * next_value * mask - value

                critic_loss = 0.5 * advantage.pow(2)
                actor_loss = -log_prob * advantage.detach() # 策略网络损失
                
                # loss = actor_loss + 5 * critic_loss + 0.001 * entropy

                agent.optim_critic.zero_grad()
                critic_loss.backward()
                agent.optim_critic.step()

                agent.optim_actor.zero_grad()
                actor_loss.backward()
                agent.optim_actor.step()

                reward_update.clear()
                update_count +=1
                if update_count % TARGET_UPDATE_FREQUENCY == 0:
                    agent.target.load_state_dict(agent.critic.state_dict())

                # 统计、打印
                if done:
                    reward_episode_list.append(reward_till_step_i)
                    if episode_i % PRINT_FREQUENCY == 0:
                        reward_ep20avg_list.append(np.mean(reward_episode_list[-PRINT_FREQUENCY:]))
                        print("Episode: {}, Avg. Reward: {}".format(episode_i, reward_ep20avg_list[-1]))

    # 保存最高分的模型参数
    agent.save_trained_model(best_model_parameters) # 保存模型参数
    print("Training is over! The best episode reward is {}".format(max_reward))

    # 作图
    ep_axis = range(PRINT_FREQUENCY, n_episodes+1, PRINT_FREQUENCY)
    reward_avg_list = [np.mean(reward_episode_list[:i-1]) for i in ep_axis]
    # plt.plot(range(1, n_episodes+1), reward_episode_list, label="episode_rewards")
    plt.plot(ep_axis, reward_ep20avg_list, label=f"Avg.{PRINT_FREQUENCY}_most_recent ep_rewards")
    plt.plot(ep_axis, reward_avg_list, label=f"Avg. ep_rewards")
    plt.legend(loc=0)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('./Figures/A2C_multi-step.png')
    plt.show()



    if is_test:
        test(agent, device)
                    

#device = torch.device(get_default_device()) # 定义device
device = 'cpu' # MacM1pro mps < cpu

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    #==========================#
    is_load_model = False # 训练模式
    is_load_model = True # 注释则训练，取消注释则加载已有模型

    is_test = True
    #==========================#

    if is_load_model: # 是否从指定路径中加载模型参数
        agent = Agent(input_dim, output_dim, device)
        env.close()
        agent.load_pretrained_model(model_path="./Models/cartpole-multi-step-endless.pth")
        if is_test:
            test(agent, device, is_sample = True)

    else:
        train(env, input_dim, output_dim, is_test=False, device='cpu')

'''
Episode: 1870, Avg. Reward: 4314.7001953125 ?
Episode: 1890, Avg. Reward: 262.6000061035156
Episode: 1910, Avg. Reward: 43152.69921875
Episode: 1920, Avg. Reward: 34.900001525878906
Episode: 1950, Avg. Reward: 3316.39990234375
...停不下来woc
Episode: 1960, Avg. Reward: 151230.796875
Episode: 1970, Avg. Reward: 383.0
很不稳定，一次高峰后就立马寄


待完成工作：
1. 作图
2. 保存奖励分最高的模型
'''
