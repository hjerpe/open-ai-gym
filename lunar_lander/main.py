import argparse
import gym
import time
import numpy as np
import numpy.typing as npt


env = gym.make("LunarLander-v2")
ACTIONS: dict[str, int] = {
    "no_action": 0,
    "left_engine": 1,
    "main_engine": 2,
    "right_engine": 3,
}


def heuristic_tyro(observation: npt.NDArray[np.float_]) -> int:
    """
    :param observation: An eight dimensional np.array representing a state.
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
        action = ACTIONS["no_action"]
    elif obs_dict["x_vel"] < -0.5:
        action = ACTIONS["right_engine"]
    elif obs_dict["x_vel"] > 0.5:
        action = ACTIONS["left_engine"]
    elif obs_dict["y_vel"] < -0.3:
        action = ACTIONS["main_engine"]
    elif obs_dict["angle"] < 0.05:
        action = ACTIONS["left_engine"]
    elif obs_dict["angle"] > 0.05:
        action = ACTIONS["right_engine"]
    elif obs_dict["y_vel"] < -0.1:
        action = ACTIONS["main_engine"]
    else:
        action = ACTIONS["no_action"]
    return action


def zero_policy() -> int:
    """
    :return: An integer encoding no action.
    """
    return ACTIONS["no_action"]


def sample_policy() -> int:
    """
    :return: An integer encoding a random action.
    """
    action = env.action_space.sample()
    return int(action)


def main(args):
    for i_episode in range(args.episodes):
        observation_ = env.reset().astype(np.float_)  # type: ignore
        for time_ in range(args.time_limit):
            env.render()
            time.sleep(0.001)

            if args.policy == "heuristic":
                action = heuristic_tyro(observation_)
            elif args.policy == "random":
                action = sample_policy()
            elif args.policy == "zero":
                action = zero_policy()

            observation_, reward_, done_, info_ = env.step(action)
            if done_:
                print("Episode finished after {} timesteps".format(time_ + 1))
                break
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Lunar Lander.")
    parser.add_argument(
        "--policy",
        choices=["heuristic", "random", "zero"],
        default="heuristic",
        help="what policy to use",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="how many episodes to run",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=300,
        help="time limit for each episode",
    )

    args = parser.parse_args()
    main(args)
