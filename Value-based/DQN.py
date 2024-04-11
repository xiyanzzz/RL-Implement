import random
import gym
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm

# 经验回放池
class Replay_buffer:
    def __init__(self, n_s, n_a):
        self.n_s = n_s
        self.n_a = n_a
        self.BUFFER_SIZE = 10000
        self.BATCH_SIZE = 64
        self.t_buf = 0
        self.t_max = 0

        # 因为s r a每个大小不一样，先申请空间(空或者随机初始化)
        self.all_s = np.empty(shape=(self.BUFFER_SIZE, self.n_s), dtype=np.float32)
        self.all_a = np.random.randint(low=0, high=n_a, size=self.BUFFER_SIZE, dtype=np.uint8)
        self.all_r = np.empty(shape=self.BUFFER_SIZE, dtype=np.float32)
        self.all_done = np.random.randint(low=0, high=2, size=self.BUFFER_SIZE, dtype=np.uint8)
        self.all_s_ = np.empty(shape=(self.BUFFER_SIZE, self.n_s), dtype=np.float32)

    def add_experience(self, s, a, r, done, s_):
        self.all_s[self.t_buf] = s
        self.all_a[self.t_buf] = a
        self.all_r[self.t_buf] = r
        self.all_done[self.t_buf] = done
        self.all_s_[self.t_buf] = s_
        self.t_buf = (self.t_buf + 1) % self.BUFFER_SIZE # 既加1，又最大重置，替换掉前面旧的经验
        self.t_max = max(self.t_max, self.t_buf) # 一开始随t_buf逐渐增加，到t_buf重置后不再跟随并保持不变，用来检查经验池经验数量

    # opt +  加光标点击多处 同时输入
    def get_batch(self):
        # 存的经验大于batch时随机抽, 不够时有多少取多少
        if self.t_max >= self.BATCH_SIZE: # 
            indices = random.sample(range(self.t_max), self.BATCH_SIZE) # 从有效索引中随机取索引
        else:
            indices = range(0, self.t_max)

        batch_s = []
        batch_a = []
        batch_r = []
        batch_done = []
        batch_s_ = []

        for idx in indices:
            batch_s.append(self.all_s[idx])
            batch_a.append(self.all_a[idx])
            batch_r.append(self.all_r[idx])
            batch_done.append(self.all_done[idx])
            batch_s_.append(self.all_s_[idx])
        # 按住option复制光标，双击能选中变量一键复制
        batch_s_tensor = torch.as_tensor(np.asarray(batch_s), dtype=torch.float32)
        batch_a_tensor = torch.as_tensor(np.asarray(batch_a), dtype=torch.int64).unsqueeze(-1) # (2,) -> (2,1) batch_a作为index必须int64
        batch_r_tensor = torch.as_tensor(np.asarray(batch_r), dtype=torch.float32).unsqueeze(-1)
        batch_done_tensor = torch.as_tensor(np.asarray(batch_done), dtype=torch.float32).unsqueeze(-1)
        batch_s__tensor = torch.as_tensor(np.asarray(batch_s_), dtype=torch.float32)
        

        return batch_s_tensor, batch_a_tensor, batch_r_tensor, batch_done_tensor, batch_s__tensor

# 神经网络
class DQN(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=n_input, out_features=64),
            nn.Tanh(),
            nn.Linear(in_features=64, out_features=n_output)
        )

    def forward(self, x):
        return self.net(x)



class Agent:
    def __init__(self, n_input, n_output):
        self.n_input = n_input
        self.n_output = n_output
        #=====================#
        self.Gamma = 0.99
        self.learning_rate = 0.1
        #=====================#
        self.buffer = Replay_buffer(self.n_input, self.n_output) 

        self.main_net = DQN(self.n_input, self.n_output) 
        self.target_net = copy.deepcopy(self.main_net)

        self.loss = nn.functional.smooth_l1_loss
        self.optimizer = torch.optim.Adam(self.main_net.parameters(), lr=self.learning_rate)

    def get_action(self, obs):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        q_value = self.main_net(obs_tensor).unsqueeze(0) # forward
        max_q_idx = torch.argmax(input=q_value) # 输出idx
        a_q_max = max_q_idx.item()
        return a_q_max
    # 加载预训练模型
    def load_pretrained_model(self, model_path="./Models/cartpole-dqn.pth"):
        self.main_net.load_state_dict(torch.load(model_path))
    # 保存模型参数
    def save_trained_model(self,model_parameters, model_path="./Models/cartpole-dqn.pth"):
        torch.save(model_parameters, model_path)

# 用训练好的模型玩一局，pygame可视化
def test(agent):
    env_test = gym.make("CartPole-v1", render_mode="human") # gym>=0.25.0
    s, _ = env_test.reset()
    done = False
    while not done:
        a = agent.get_action(s)
        s, _, done, _, _ = env_test.step(a)
        # env_test.render() # 不需要手动render
    env_test.close()


def train(env_train, input_dim, output_dim, is_test):
    #======================#
    epsilon_max = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.0005

    n_episode = 5000 # 5000正好，可以略降 或 设置max_step
    TARGET_UPDATE_FREQUENCY = 100
    #======================#
    agent = Agent(n_input=input_dim, n_output=output_dim)
    best_model_parameters = agent.main_net.state_dict()

    reward_array = np.empty(shape=n_episode) # 记录各回合的累计奖励
    avg_episode_reward = [] # 所有过往回合平均累计奖励 (每100回合统计一次)
    max_episode_reward = 0 # 最大单回合累计奖励
    for episode_i in tqdm(range(1, n_episode+1)):
        episode_reward = 0 # 单回合累计奖励
        s, _ = env_train.reset()
        done = False
        step_i = 0

        while not done: # 原来的for step_i in range(n_time_step):出现逻辑问题，手动break导致n_time_step实际并没有跑满，epsilon的实际降幅跨度较大
            step_i +=1
    # 1.根据epsilon-greedy策略选择行动
            # epsilon从1到0.05随episode_i指数衰减
            epsilon = epsilon_min + (epsilon_max - epsilon_min) * np.exp(-epsilon_decay * episode_i)
            int_random = random.random()
            if int_random <= epsilon:
                a = env_train.action_space.sample()
            else:
                a = agent.get_action(s) 
    # 2.执行行动获取观测值，储存到buffer
            s_, r, done, info, _ = env_train.step(a)
            agent.buffer.add_experience(s, a, r, done, s_) 
    # 3.后续工作
            s = s_
            episode_reward += r

            if done:
                reward_array[episode_i-1] = episode_reward
                # 保存累计奖励最多的模型的参数
                if episode_i >= 4000 and episode_reward > max(10000, max_episode_reward):
                    best_model_parameters = copy.deepcopy(agent.main_net.state_dict()) # 最后一次更新后因为没有跑过，不知道累计奖励，(可以考虑放test里)但影响不大
                    max_episode_reward = episode_reward

    # 4.从buffer抽样进行mini-batch训练（没有预存足够的experience）
        # 4.1 抽batch
        batch_s, batch_a, batch_r, batch_done, batch_s_ = agent.buffer.get_batch() 
        # 4.2 计算target
        with torch.no_grad(): # 不加不影响结果，optimi不含targetNN的参数
            target_q_values = agent.target_net(batch_s_) 

        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
        ''' 操作解释
        [[s_3]             [[q3(a1), q3(a2)]               [[max(q3)]
        [s_1]             [q1(a1), q2(a2)]                 [max(q2)]
        ...     ->(Q_net)   ...                 ->(max)    ...
        [s_7]]            [q7(a1), q7(a2)]]                [max(q7)]]
        '''

        # 另一种方法: DDQN, 选择a*与计算maxq分开, 以缓解最大化带来的的高估; 效果不佳, 可能需要额外调参(?)
        '''
        q_values_next = agent.main_net(batch_s_)
        index_a_max = torch.argmax(q_values_next, axis=1, keepdim=True) # 最大Q对应的动作a* (列)
        max_target_q_values = target_q_values.gather(1, index_a_max) # 按a*取Q_target
        '''

        # (1-batch_done): 若s_为last step，y_target = r 
        target_values = batch_r + agent.Gamma * (1-batch_done) * max_target_q_values 
        # 4.3 计算q_t
        q_values = agent.main_net(batch_s)
        a_q_values = torch.gather(input=q_values, dim=1, index=batch_a) # 计算是计算所有a对应的q，取只取batch_a对应的q(s,a)
        # 4.4 计算损失
        loss = agent.loss(target_values, a_q_values) # 类似L1 Loss的函数，默认返回mean(batch_loss)
        # 4.5 更新参数
        agent.optimizer.zero_grad() 
        loss.backward()
        agent.optimizer.step()

    # 5.更新target_net
        if episode_i % TARGET_UPDATE_FREQUENCY == 0:
            agent.target_net.load_state_dict(agent.main_net.state_dict())

            # 6. 打印统计量
            avg_last100_reward = np.mean(reward_array[episode_i-TARGET_UPDATE_FREQUENCY:episode_i]) # 最近100回合的平均累计奖励
            avg_episode_reward.append(np.mean(reward_array[:episode_i-1]))
            print("Episode: {},\tAvg.{} Reward: {:.2f},\tAvg.all Reward: {:.2f}".format(episode_i, TARGET_UPDATE_FREQUENCY, 
                  avg_last100_reward,
                  avg_episode_reward[-1]))

    agent.save_trained_model(best_model_parameters) # 保存模型参数
    print("Training is over! The best episode reward is {}".format(max_episode_reward))
    env_train.close()

    # 7. 画图
    plt.plot(range(1, n_episode+1, TARGET_UPDATE_FREQUENCY), avg_episode_reward)
    plt.xlabel('Episode')
    plt.ylabel('Avg. Reward')
    plt.savefig('./Figures/DQN_cartpole.png') # 图片保存路径
    plt.show()

    # 8. 测试
    if is_test:
        test(agent)
        '''
        episode_reward = test(agent) # 为test加上return
        if episode_reward > max_episode_reward:
            agent.save_trained_model(agent.main_net.state_dict()) # 因为没有探索性，所以累计奖励并不公平
        '''


# 可以选择从本地加载预训练好的模型 或 从0训练
if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    #==========================#
    is_load_model = False # 训练模式
    # is_load_model = True # 注释则训练，取消注释则加载已有模型

    is_test = True
    #==========================#

    if is_load_model: # 是否从指定路径中加载模型参数
        agent = Agent(n_input=input_dim, n_output=output_dim)
        env.close()
        agent.load_pretrained_model(model_path="./Models/cartpole-dqn-1353606.pth")
        if is_test:
            test(agent)

    else:
        train(env, input_dim, output_dim, is_test)

