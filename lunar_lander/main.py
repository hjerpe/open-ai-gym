import gym
import time
import numpy as np
import numpy.typing as npt


env = gym.make("LunarLander-v2")
actions: dict[str, int] = {
    "no_action": 0,
    "left_engine": 1,
    "main_engine": 2,
    "right_engine": 3,
}


def heuristic_tyro(observation: npt.NDArray[np.float_], time_: int) -> int:
    """
    :param observation: An eight dimensional np.array representing a state.
    :param time_: An integer denoting the current time step.
    :return: An integer encoding an action.
    """
    (
        x_pos,
        y_pos,
        x_vel,
        y_vel,
        angle,
        angular_vel,
        leg1_contact,
        leg2_contact,
    ) = observation
    obs_dict = {
        "x_pos": x_pos,
        "y_pos": y_pos,
        "x_vel": x_vel,
        "y_vel": y_vel,
        "angle": angle,
        "angular_vel": angular_vel,
        "leg1_contact": leg1_contact,
        "leg2_contact": leg2_contact,
    }
    action: int = -1
    if obs_dict["leg1_contact"] == 1 or obs_dict["leg2_contact"] == 1:
        action = actions["no_action"]
    elif obs_dict["x_vel"] < -0.5:
        action = actions["right_engine"]
    elif obs_dict["x_vel"] > 0.5:
        action = actions["left_engine"]
    elif obs_dict["y_vel"] < -0.3:
        action = actions["main_engine"]
    elif obs_dict["angle"] < 0.05:
        action = actions["left_engine"]
    elif obs_dict["angle"] > 0.05:
        action = actions["right_engine"]
    elif obs_dict["y_vel"] < -0.1:
        action = actions["main_engine"]
    else:
        action = actions["no_action"]
    return action


def zero_policy(observation: npt.NDArray[np.float_], time_: int) -> int:
    """
    :param observation: An eight dimensional np.array representing a state.
    :param time_: An integer denoting the current time step.
    :return: An integer encoding no action.
    """
    return actions["no_action"]


def sample_policy() -> int:
    """
    :return: An integer encoding a random action.
    """
    action = env.action_space.sample()
    return int(action)


TIME_LIMIT = 300
EPISODE_LIMIT = 5
for i_episode in range(EPISODE_LIMIT):
    observation_ = env.reset().astype(np.float_) # type: ignore
    for time_ in range(TIME_LIMIT):
        env.render()
        time.sleep(0.001)
        action = heuristic_tyro(observation_, time_)
        observation_, reward_, done_, info_ = env.step(action)
        if done_:
            print("Episode finished after {} timesteps".format(time_ + 1))
            break
env.close()
