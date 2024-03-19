import random
import gym
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import animation
from REINFORCEwithBaseline import Agent as REIN_Agent
from A2CwithMultiStepTD import Agent as A2C_Agent

def display_frames_as_gif(frames, agent_name):
    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval = 5)
    anim.save(f"./CartPole_v1_result_{agent_name}.gif", writer="pillow", fps = 30)

    
def test(agent, agent_name, device='cpu', is_sample = True):
    env_test = gym.make("CartPole-v1", render_mode="rgb_array_list") # gym >= 0.26.0
    s, _ = env_test.reset()
    done = False
    for _ in range(500):
        if agent_name == 'REINF':
            dist, _ = agent.model(torch.as_tensor(s, dtype=torch.float32).to(device))
        elif agent_name == 'A2C':
             dist = agent.actor(torch.as_tensor(s, dtype=torch.float32).to(device))
        if is_sample:
            a = dist.sample().item()
        else:
            a = torch.argmax(dist.probs).item()
        s, _, _, _, _ = env_test.step(a)
    frames = env_test.render()
    print(len(frames), frames[0].shape)
    display_frames_as_gif(frames, agent_name)



if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    agent_name = "A2C"
    agent_name = "REINF"

    if agent_name == 'REINF':
        agent = REIN_Agent(input_dim, output_dim, device='cpu')
        agent.load_pretrained_model(model_path="./Models/cartpole-REINFb-56937.pth")

    elif agent_name == 'A2C':
        agent = A2C_Agent(input_dim, output_dim, device='cpu')
        agent.load_pretrained_model(model_path="./Models/cartpole-multi-step-endless.pth")

    test(agent, agent_name, is_sample = True)


'''
参考链接：
https://blog.csdn.net/ice_bear221/article/details/123735643
https://blog.csdn.net/qq_33361420/article/details/112471755
https://github.com/openai/gym/pull/2671
'''