import gym
import time

env = gym.make("LunarLander-v2")
actions = {"no_action": 0, "left_engine": 1, "main_engine": 2, "right_engine": 3}


def heuristic_tyro(observation, t):
    x_pos, y_pos, x_vel, y_vel, angle, angular_vel, leg1_contact, leg2_contact = observation
    obs_dict = {
        "x_pos": x_pos,
        "y_pos": y_pos,
        "x_vel:": x_vel,
        "y_vel": y_vel,
        "angle": angle,
        "angular_vel": angular_vel,
        "leg1_contact": leg1_contact,
        "leg2_contact": leg2_contact
    }
    print(obs_dict)
    action = None
    if leg1_contact == 1 or leg2_contact == 1:
        action = actions["no_action"]
    elif x_vel < -0.5:
        action = actions["right_engine"]
    elif x_vel > 0.5:
        action = actions["left_engine"]
    elif y_vel < -0.3:
        action = actions["main_engine"]
    elif angle < 0.05:
        action = actions["left_engine"]
    elif angle > 0.05:
        action = actions["right_engine"]
    elif y_vel < -0.1:
        action = actions["main_engine"]
    else:
        action = actions["no_action"]
    return action

def policy(obs, t):
    return actions["no_action"]


def sample_policy(obs, t):
    action = env.action_space.sample()
    return action


TIME_LIMIT = 300
EPISODE_LIMIT = 5
for i_episode in range(EPISODE_LIMIT):
    observation = env.reset()
    for t in range(TIME_LIMIT):
        env.render()
        time.sleep(0.001)
        action = heuristic_tyro(observation, t)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
