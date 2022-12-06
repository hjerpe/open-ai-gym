import argparse
import gym
import time
import numpy as np
import numpy.typing as npt
import mlflow
import os
import datetime
import pickle
import random


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


class Discretizer:
    """
    Class which bins a floating point number into distinct buckets. Each distinct bin
    is represented by an integer.
    :param boundaries: List of floating point number denoting the boundaries for the
    buckets.
    """

    def __init__(self, boundaries: list[float] = [0, 0.5]):
        self.boundaries = list(boundaries)
        self.cardinality = len(self.boundaries) + 1

    def discretize(self, number: float) -> int:
        """
        Return a binned integer number from a number.
        :param number: Floating point number
        :return: Returns a discrete representation with integer denoting whether the number
            is smaller, between or greater than the numbers in the boundaries list.

        Example with boundaries = [0, 0.5]. The integer is a number 0,..,3. Where
        the number denote whether the number is,
        less than 0,
        less than 0.5,
        or greater than 0.5.
        """
        return int(np.digitize(number, self.boundaries))


class MapNDto1d:
    """
    Class which maps a N dimensional index into 1 dimensional index.
    That is returns a 1-1 mapping from a N_1 x N_2 x .. x N_d index to an
    integer index.
    Example:
        In 3 dimensions the index from 3D with dimensions N1xN2xN3 and
        where each index n_i is zero based, that is n_i assumes values in 0,..,N_i-1.
        Then the 1D index is given by n_1 + N_1*(n_2 + N_2*(n_3)).
        E.g., with a point (x,y,z) with values between 0,1,2 then the 1D index for the point
        (x,y,z) equals x + 3*(y + 3*(z)) = x+3y+9z, and (1,2,1) -> 1 + 3*(2 + 3*(1)) = 16
    """

    def __init__(self, cardinality_per_dimension: list[int]):
        """
        :param cardinality_per_dimension: List of integers denoting value range (the cardinality) of
        each dimension.
        """
        self._dimensions = cardinality_per_dimension

    def size(self) -> int:
        """
        :return: Return an integer denoting the size of the space.
        Example with x,y,z and values in 0,1,2 the size is N_x*N_y*N_z with N_i = 3
        and the size is 3*3*3 = 27.
        """
        product = 1
        for dim in self._dimensions:
            product *= dim
        return product

    def index_1d(self, point_: list[int]) -> int:
        """
        :param point_: List of integers denoting the value at each dimension.
        :return: An integer denoting the 1D index.
        """
        dim = 1
        one_dim_index = 0
        for index in range(len(point_)):
            one_dim_index += dim * point_[index]
            dim *= self._dimensions[index]
        return one_dim_index


class QLearning:
    """
    Observed states is a list of 8 number, and where the first 6 numbers
    are discrete.
    """

    def __init__(self, env, alpha=0.1, gamma=0.6, epsilon=0.1):

        self.env = env

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.all_epochs = []
        self.all_penalties = []

        # boundaries = [0, 0.5]
        self._discretizer = Discretizer([-1, -0.5, -0.1, -0.05, 0, 0.05, 0.1, 0.5, 1])
        self.size_discrete_dimension = self._discretizer.cardinality
        self.size_indicator_dimension = 2
        # Dimensions for continuous states
        dimensions = [self.size_discrete_dimension for _ in range(6)]
        # Extend with discrete states
        dimensions.extend([self.size_indicator_dimension for _ in range(2)])
        self.map_nd_to_1d = MapNDto1d(cardinality_per_dimension=dimensions)
        size_state_space = self.map_nd_to_1d.size()
        size_action_space = env.action_space.n
        self.q_table = np.zeros([size_state_space, size_action_space])

    def _discretize_number(self, number: float):
        """
        Return a binned integer number from a number.
        :param number: Floating point number
        :return: Returns a discrete representation with values in 0,..,3 whether the number
        is less than or greater than each of the numbers in the boundaries list.
        """
        return self._discretizer.discretize(number)

    def index_1d(self, state_: list[int]):
        return self.map_nd_to_1d.index_1d(state_)

    def get_discretized_state(self, list_of_states: list[float]):
        """
        Returns a discretized state of the input list. That is each float number is mapped to an
        integer.
        :param list_of_states: A list of 8 floating point numbers.
        :return: A list of 8 integers where each floating point number has been mapped to an integer
        """
        # Only the last two state dimensions are already discrete
        states_to_discretize = list_of_states[:-2]
        indicator_states = list_of_states[-2:]
        discretized_states = [0 for _ in range(len(list_of_states))]
        # Get index for
        start_index = 0
        for i in range(len(states_to_discretize)):
            discretized_state_ = self._discretize_number(states_to_discretize[i])
            discretized_states[i] = discretized_state_
            start_index = i

        for j in range(len(indicator_states)):
            state = int(indicator_states[j])
            index = start_index + 1 + j
            discretized_states[index] = state
        return discretized_states

    def _persist_q_table_to_disk(self, name=None):
        today_str = datetime.date.today().strftime("%Y%m%d")
        if not name:
            name = f"{self._get_model_dir()}/q_table_{today_str}"
        with open(name, "wb") as fp:
            pickle.dump(self.q_table, fp)
        return name

    def _get_model_dir(self):
        return "./data/models"

    def load_q_table_from_disk(self, name):
        name = f"{self._get_model_dir()}/{name}"
        with open(name, "rb") as fp:
            self.q_table = pickle.load(fp)
        status = f"Load Q-table {name}"
        return status

    def train(self, num_training_episodes=2e4):
        for i in range(1, int(num_training_episodes)):
            state = self.env.reset()
            discrete_state = self.get_discretized_state(state)
            state_index = self.index_1d(discrete_state)
            epochs, penalties, reward = 0, 0, 0
            done = False

            while not done:
                if random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()  # Explore
                else:
                    discrete_state = self.get_discretized_state(state)
                    state_index = self.index_1d(discrete_state)
                    action = np.argmax(self.q_table[state_index])

                next_state, reward, done, info = self.env.step(action)
                discrete_next_state = self.get_discretized_state(next_state)
                next_state_index = self.index_1d(discrete_next_state)

                old_value = self.q_table[state_index, action]
                next_max = np.max([self.q_table[next_state_index]])
                new_value = (1 - self.alpha) * old_value + self.alpha * (
                    reward + self.gamma * next_max
                )
                self.q_table[state_index, action] = new_value
                if reward == -100:
                    penalties += 1
                state = next_state
                epochs += 1

            if i % 100 == 0:
                print(f"Episode: {i}")
                print(f"Num epochs: {epochs}")
                print(self.q_table)
        status = self._persist_q_table_to_disk()
        print(f"Persisting Q-table {status}")
        print("Training finished.\n")

    def next_action(self, state):
        discretized_state = self.get_discretized_state(state)
        state_index = self.index_1d(discretized_state)
        action = np.argmax(self.q_table[state_index])
        return action


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
    q_learn = QLearning(env=env)
    status = q_learn.load_q_table_from_disk(name="q_table_20221205")
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
            elif args.policy == "discretized-q-learning":
                action = q_learn.next_action(state=observation_)

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
        choices=["heuristic", "random", "zero", "discretized-q-learning"],
        default="discretized-q-learning",
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
