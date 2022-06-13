import gym
import time

env = gym.make("LunarLander-v2")
actions = {"nop": 0, "fire_left": 1, "main_engine": 2, "right_engine": 3}


def policy(obs, t):
    return actions["nop"]


def sample_policy(obs, t):
    action = env.action_space.sample()
    return action


TIME_LIMIT = 300
EPISODE_LIMIT = 3
for i_episode in range(EPISODE_LIMIT):
    observation = env.reset()
    for t in range(TIME_LIMIT):
        env.render()
        print(observation)
        time.sleep(0.001)
        if i_episode % 2 == 0:
            action = sample_policy(observation, t)
        else:
            action = policy(observation, t)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
