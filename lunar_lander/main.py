import gym
import time
import numpy as np
import random
import pickle
import datetime

env = gym.make("LunarLander-v2")
actions = {"no_action": 0, "left_engine": 1, "main_engine": 2, "right_engine": 3}


class QLearning:

    def __init__(self, env, alpha=0.1, gamma=0.6, epsilon=0.1):

        self.env = env

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.all_epochs = []
        self.all_penalties = []

        size_state_space = self._size_state_space()
        size_action_space = env.action_space.n
        self.q_table = np.zeros([size_state_space, size_action_space])

    def _discretize_number(self, number):
        """
        Return a binned integer number from a number.
        :param number: Floating point number
        :return: Returns a discrete representation with values 1, 2, 3, 4 whether the number
        is less than -0,5, less than 0, less than 0.5, or greater than 1.
        """
        return np.digitize(number, [-0.5, -0.1, -0.05, 0, 0.05, 0.1, 0.5]) + 1

    def _state_integer_index(self, li):
        """
        Returns a 1-1 mapping from a
        (6*size_discrete_state_dimension + 2*size_indicator_state_dimension)
        dimensional vector to a 1D integer index.
        :param li: A list of 8 floating point numbers.
        :return: An 1-1 mapping a
        (6*size_discrete_state_dimension + 2*size_indicator_state_dimension) dimensional
        vector into an integer.
        """
        states_to_discretize = li[:-2]
        indicator_states = li[-2:]
        size_discrete_state_dimensions = 8
        size_indicator_state_dimension = 2
        index = 0
        for i in range(len(states_to_discretize)):
            s = self._discretize_number(states_to_discretize[i])
            index += s*(size_discrete_state_dimensions-i)
        for i in range(len(indicator_states)):
            s = self._discretize_number(indicator_states[i])
            index += s*(size_indicator_state_dimension-i)
        print(self.q_table.shape)
        print(index)
        return index

    def _size_state_space(self):
        return 6*8 + 2*2

    def _persist_q_table_to_disk(self, name=None):
        today_str = datetime.date.today().strftime("%Y%m%d")
        if not name:
            name = f"{self._get_model_dir()}/q_table_{today_str}"
        with open(name, "wb") as fp:
            pickle.dump(self.q_table, fp)
        return name

    def _get_model_dir(self):
        return "./data/models"

    def load_q_table_from_disk(self, name="q_table_20220619"):
        name = f"{self._get_model_dir()}/{name}"
        with open(name, "rb") as fp:
            self.q_table = pickle.load(fp)
        status = f"Load Q-table {name}"
        return status

    def train(self, num_training_episodes=1e5):
        for i in range(1, int(num_training_episodes)):
            state = self.env.reset()
            state_index = self._state_integer_index(state)
            epochs, penalties, reward = 0, 0, 0
            done = False

            while not done:
                if random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()  # Explore
                else:
                    state_index = self._state_integer_index(state)
                    action = np.argmax(self.q_table[state_index])

                next_state, reward, done, info = self.env.step(action)
                next_state_index = self._state_integer_index(next_state)

                old_value = self.q_table[state_index, action]
                next_max = np.max([self.q_table[next_state_index]])
                new_value = ((1-self.alpha)*old_value + self.alpha*(reward + self.gamma*next_max))
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
        state_index = self._state_integer_index(state)
        action = np.argmax(self.q_table[state_index])
        return action


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
    if obs_dict["leg1_contact"] == 1 or leg2_contact == 1:
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


def sample_policy(obs, t):
    action = env.action_space.sample()
    return action


q_learn = QLearning(env=env)
q_learn.train()
status = q_learn.load_q_table_from_disk(name="q_table_20220627")
print(status)
TIME_LIMIT = 400
EPISODE_LIMIT = 5
for i_episode in range(EPISODE_LIMIT):
    observation = env.reset()
    for t in range(TIME_LIMIT):
        env.render()
        time.sleep(0.001)
        action = q_learn.next_action(state=observation)
        #action = heuristic_tyro(observation, t)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()