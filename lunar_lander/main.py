import argparse
import gym
import time
import numpy as np
import numpy.typing as npt
import mlflow
import os


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
    if args.mlflow_logging:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(args.mlflow_experiment_name)

    rewards = []
    for i_episode in range(1, args.episodes + 1):
        observation_ = env.reset().astype(np.float_)  # type: ignore
        for time_ in range(1, args.time_limit + 1):
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
                print(
                    f"Episode {i_episode} finished after {time_ + 1} timesteps with reward {reward_}"
                )
                rewards.append(reward_)
                break
    if args.mlflow_logging:
        with mlflow.start_run(run_name=args.mlflow_run_name):
            mlflow.log_param("Policy", args.policy)
            mlflow.log_param("Episodes", args.episodes)
            mlflow.log_param("Time limit", args.time_limit)
            mlflow.log_metric("Average reward", sum(rewards) / len(rewards))
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
    parser.add_argument(
        "--mlflow-logging",
        action="store_true",
        help="""log experiment in MLFlow.
        The following env variables needs to be set, MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD
        """,
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        default="lunar-lander",
        help="the name of the MLflow experiment",
    )
    parser.add_argument(
        "--mlflow-run-name",
        help="the name of the MLflow run",
    )

    args = parser.parse_args()
    main(args)
