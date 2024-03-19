import gym
import matplotlib.pyplot as plt
from matplotlib import animation
from DQN_CartPole import Agent


def display_frames_as_gif(frames):
    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval = 5)
    anim.save("./CartPole_v1_result.gif", writer="pillow", fps = 30)

    
def test(agent):
    env_test = gym.make("CartPole-v1", render_mode="rgb_array_list") # gym >= 0.26.0
    s, _ = env_test.reset()
    done = False
    for _ in range(500):
        a = agent.get_action(s)
        s, _, _, _, _ = env_test.step(a)
    frames = env_test.render()
    print(len(frames), frames[0].shape)
    #display_frames_as_gif(frames)



if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    agent = Agent(n_input=input_dim, n_output=output_dim)
    env.close()
    agent.load_pretrained_model(model_path="./Models/cartpole-dqn-1353606.pth")
    test(agent)

'''
参考链接：
https://blog.csdn.net/ice_bear221/article/details/123735643
https://blog.csdn.net/qq_33361420/article/details/112471755
https://github.com/openai/gym/pull/2671
'''