import torch
from torch import nn 
from torch.distributions import Categorical
import numpy as np
import gym
import matplotlib.pyplot as plt
# from tqdm import tqdm
import copy

class PolicyValueNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer = 256):
        super(PolicyValueNN, self).__init__()
        self.policyNN = nn.Sequential(
            nn.Linear(input_dim, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, output_dim),
            nn.Softmax(dim=-1)
        )
        self.valueNN = nn.Sequential(
            nn.Linear(input_dim, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, 1)
        )
    def forward(self, state):
        value = self.valueNN(state)
        dist = Categorical(self.policyNN(state))
        return dist, value

    
class Agent:
    def __init__(self, input_dim, output_dim, device):
        self.device = device
        #==========================#
        self.gamma = torch.tensor(0.99).float().to(self.device)
        self.lr = torch.tensor(5e-4).float().to(self.device)
        #==========================#
        self.model = PolicyValueNN(input_dim, output_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    # 加载预训练模型
    def load_pretrained_model(self, model_path="./Models/cartpole-REINFb.pth"):
        self.model.load_state_dict(torch.load(model_path))
    # 保存模型参数
    def save_trained_model(self, model_parameters, model_path="./Models/cartpole-REINFb.pth"):
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
        dist, _ = agent.model(torch.as_tensor(s, dtype=torch.float32).to(device))
        if is_sample:
            a = dist.sample().item()
        else:
            a = torch.argmax(dist.probs).item()
        s, _, done, _, _ = env_test.step(a)
        # env_test.render() # 不需要手动render
    env_test.close()



def train(env_train, input_dim, output_dim, is_test, device):

    agent = Agent(input_dim=input_dim, output_dim=output_dim, device=device)
    #======================#
    n_episodes = 2000
    PRINT_FREQUENCY = 20
    #======================#

    reward_episode_list = []
    reward_ep20avg_list = []
    max_episode_reward = 0 # 训练单回合最大累计奖励
    best_model_parameters = copy.deepcopy(agent.model.state_dict())
    for episode_i in range(1, n_episodes+1):
        s, _ = env_train.reset()
        done = False
        step_i = 0
        #episode_reward = 0
        # entropy = 0
        

        log_prob_list = []
        value_list = []
        reward_list = []
        # mask_list = []
        # next_value_list = []
        return_list = []
        

        while not done:
            step_i += 1
            # 前向，获取动作与状态价值
            dist, value = agent.model(torch.as_tensor(s, dtype=torch.float32).to(device))
            a = dist.sample() # <Categorical>对象，dist拥有很多<Attributes> 如dist.probs查看概率 / a为tensor([])

            # interact
            s_, r, done, _ , _ = env_train.step(a.item())

            # collect
            log_prob = dist.log_prob(a).unsqueeze(-1) # 等效 torch.log(dist.probs[a]), 单独对a的概率取对数(计算梯度), tensor标量转[]
            # entropy += dist.entropy() # 对分布求熵，计算梯度
            # _, next_value = agent.model(torch.as_tensor(s_, dtype=torch.float32).to(device))


            #target_value = reward + agent.gamma * (torch.FloatTensor(1-done).to(device)) * next_value
            #advantage = 

            log_prob_list.append(log_prob.unsqueeze(-1))
            value_list.append(value.unsqueeze(-1))
            # next_value_list.append(next_value.unsqueeze(-1))
            reward_list.append(torch.tensor([r],dtype=torch.float).unsqueeze(-1).to(device))
            # mask_list.append(torch.tensor([1-done],dtype=torch.float).unsqueeze(-1).to(device))

            #episode_reward += r
            s = s_

            if done:
                reward_episode_list.append(torch.cat(reward_list).detach().sum().cpu())
                if episode_i % PRINT_FREQUENCY == 0:
                    reward_ep20avg_list.append(np.mean(reward_episode_list[-PRINT_FREQUENCY:]))
                    print("Episode: {}, Avg. Reward: {}".format(episode_i, reward_ep20avg_list[-1]))
                    # print("Avg. Reward: {}".format(np.mean(np.array(reward_ep20avg_list[-1]))))
                # 保存最高分的模型参数
                if episode_i > 1000 and reward_episode_list[-1] > max(3000, max_episode_reward):
                    best_model_parameters = copy.deepcopy(agent.model.state_dict())
                    max_episode_reward = reward_episode_list[-1]
        
        # 倒着计算回报
        return_u = 0
        for t in reversed(range(len(reward_list))):
            return_u = reward_list[t] + agent.gamma * return_u
            return_list.insert(0,return_u)
        
        log_prob_list = [(agent.gamma**i) * num for i,num in enumerate(log_prob_list)] # 为每项乘折扣 gamma^(t-1)

        # 将列表转为tensor([[],...,[]])
        log_prob_list = torch.cat(log_prob_list)
        value_list = torch.cat(value_list)
        # next_value_list = torch.cat(next_value_list)
        return_list = torch.cat(return_list)
        reward_list = torch.cat(reward_list)
        # mask_list = torch.cat(mask_list)

        # 更新
        delta_list = return_list - value_list # 含梯度
        policy_loss = -(log_prob_list * delta_list.detach()).sum() # 实际上是一个折扣后的加权求和 (其他思路(不折扣)：1. 对n个乘积求均值；2. 对n个乘积先归一化后求和)
        value_loss = 0.5 * delta_list.pow(2).mean()

        #loss = policy_loss + value_loss + 0.001 * entropy
        loss = policy_loss + 3 * value_loss # 相当于增大value网络的学习率
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

    # 保存最高分的模型参数
    agent.save_trained_model(best_model_parameters) # 保存模型参数
    print("Training is over! The best episode reward is {}".format(max_episode_reward))

    # 作图
    ep_axis = range(PRINT_FREQUENCY, n_episodes+1, PRINT_FREQUENCY)
    reward_avg_list = [np.mean(reward_episode_list[:i-1]) for i in ep_axis]
    plt.plot(ep_axis, reward_ep20avg_list, label=f"Avg.{PRINT_FREQUENCY}_most_recent ep_rewards")
    plt.plot(ep_axis, reward_avg_list, label="Avg. ep_rewards")
    # plt.plot(metrics['ep'], metrics['max'], label="max rewards")
    plt.legend(loc=0)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('./Figures/REINFORCE_with_b.png')
    plt.show()

    if is_test:
        test(agent, device)

# device = torch.device(get_default_device()) # 定义device
device = 'cpu' # MacM1pro

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
        agent = Agent(input_dim, output_dim, device=device)
        env.close()
        agent.load_pretrained_model(model_path="./Models/cartpole-REINFb-56937.pth")
        if is_test:
            test(agent, device=device, is_sample = False)

    else:
        train(env, input_dim, output_dim, is_test, device=device)

'''
+ 0.001 & mean()
Episode: 1500, Avg. Reward: 966.7999877929688

sum()
Episode: 1480, Avg. Reward: 631.7000122070312

- 0.001 & sum()
Episode: 1480, Avg. Reward: 473.20001220703125

+ 0.001 & sum()
Episode: 1500, Avg. Reward: 418.1000061035156

3e-4 -> 4e-4:
Episode: 1500, Avg. Reward: 1113.699951171875
参数敏感，且不稳定

-> 5e-4
Episode: 1500, Avg. Reward: 1209.699951171875

loss_critic * 5
Episode: 1500, Avg. Reward: 1509.9000244140625
Episode: 1500, Avg. Reward: 1444.699951171875

Training is over! The best episode reward is 56937.0
'''


