import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from TD3 import Config, TD3Agent

def display_frames_as_gif(frames, env_name, agent_name, return_):
    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval = 5)
    anim.save(f"./GIF/{env_name}_result_{agent_name}_{return_:.2f}.gif", writer="pillow", fps = 30)



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
        # Noise
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

    torch.manual_seed(config.env_seed)

    env = gym.make(env_name, render_mode="rgb_array_list")
    STATE_DIM = env.observation_space.shape[0]
    ACTION_DIM = env.action_space.shape[0]
    ACTION_BOUND = env.action_space.high[0]
    agent = TD3Agent(STATE_DIM, ACTION_DIM, ACTION_BOUND, config=config)

    s, _ = env.reset()
    done = False
    return_episode = 0
    while not done:
        a = agent.actor(torch.tensor(s, dtype=torch.float32)).detach().numpy()
        s, r, done, _, _ = env.step(a)
        return_episode += r
    print(return_episode)
    frames = env.render()
    display_frames_as_gif(frames, env_name, config.algor_name, return_episode)



