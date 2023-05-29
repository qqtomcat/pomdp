from typing import Tuple
from unittest import TestCase

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from tianshou.data import Batch
from collections import OrderedDict

from .base import BaseSimulator, BaseReward

"""
Env used in the paper "Glucose Level Control Using Temporal Difference Method"
time unit is 1 min
action unit is muU/ml
time unit is 1 hour
action unit is muU/ml

State:
    {G, I}
    G: plasma glycemia
    I: insulinemia

Action:
    A float

Reward
    A float.
"""

"""
ODE model used in the paper 'Glucose Level Control Using Temporal Difference Method"
time unit is 1 min
action unit is muU/ml
"""


class NooriReward(BaseReward):
    def __init__(self):
        super().__init__()

    def count_reward(self, state) -> float:
        G = state["G"]
        reward = - np.abs(G - np.mean([5.2, 5.6, 5.12]))
        return float(reward)


class TestNooriReward(TestCase):
    def __init__(self, methodName: str, Reward: NooriReward) -> None:
        super().__init__(methodName)
        self.Reward = Reward
        self.tested_name = "NooriReward"

    def test_count_reward(self):
        print(f"--------------------{self.tested_name} test count_reward--------------------")
        state = {'G': 6.14, 'I': 88.1}
        reward = self.Reward.count_reward(state)
        print(reward)


class NooriODE(BaseSimulator):
    def __init__(self):
        super().__init__()
        """
        State:
            {G, I}
            G: plasma glycemia
            I: insulinemia

        Action:
            A float
        """
    def activate(self) -> Tuple["init_state"]:
        init_state = {"G": np.array([np.mean([6.14, 7.856, 10.37])], dtype=np.float32),
                   "I": np.array([0], dtype=np.float32),}
        return init_state

    def update(self, action: float, state: dict, integral_num: int = 1) -> Tuple["next_state"]:
        G, I = state["G"], state["I"]
        for _i in range(integral_num):
            f = (G / 9) ** 3.205 / 1 + (G / 9) ** 3.205
            G_dot = -np.mean([3.11e-5, 3.21e-5, 3.11e-5]) * I * G + \
                    np.mean([0.003, 0.0023, 0.003]) / np.mean([0.187, 0.175, 0.187])
            I_dot = -np.mean([1.211e-2, 1.215e-2, 1.211e-10]) * I + np.mean(
                [1.573, 2.735, 0.242]) / np.mean([0.25, 0.215, 0.25]) * f + action
            G, I = G + G_dot / integral_num, I + I_dot / integral_num
        # count death probability
        return {"G": np.maximum(G, np.array([0.0], dtype=np.float32)),
                "I": np.maximum(I, np.array([0.0], dtype=np.float32))}


class TestNooriODE(TestCase):
    def __init__(self, methodName: str, Simulator: NooriODE) -> None:
        super().__init__(methodName)
        self.Simulator = Simulator
        self.tested_name = "NooriODE"

    def test_activate(self):
        print(
            f"--------------------{self.tested_name} test activate--------------------")
        state = self.Simulator.activate()
        print(state)

    def test_update(self):
        print(
            f"--------------------{self.tested_name} test update--------------------")
        action = 10
        state = {"G": 6.14, "I": 99.669}
        print(self.Simulator.update(action=action, state=state, integral_num=1))


class NooriDiabetesEnv(gym.Env):
    def __init__(self, max_t: int = 10000):
        super().__init__()
        self.Simulator = NooriODE()
        self.Reward = NooriReward()
        self.max_t = max_t
        self.observation_space = spaces.Dict(
            {
                'G': spaces.Box(low=0.0, high=20.0, shape=(1,), dtype=np.float32),
                'I': spaces.Box(low=0.0, high=500.0, shape=(1,), dtype=np.float32)
            }
        )
        self.action_space = spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32)

    def _random_init_state(self, init_state:dict)->dict:
        random_state = OrderedDict()
        for key, value in init_state.items():
            random_value = self.np_random.normal(value, 0.1*value, size=(1,))
            random_value = np.maximum(random_value, np.array([0.0]))
            random_value = np.minimum(random_value, np.array([1.0]))
            random_value = random_value.astype(np.float32)
            random_state[key] = random_value
        return random_state

    def reset(self, seed: int = None, random_init:bool=True, **kwargs) -> Tuple["init_state"]:
        super().reset(seed=seed)
        # init parameters
        self.states = [None] * self.max_t
        self.actions = [None] * self.max_t
        self.rewards = [None] * self.max_t
        self.infos = [None] * self.max_t
        self.t = 0
        self.terminated = False
        self.truncated = False

        # get init state
        init_state = self.Simulator.activate()
        self.states[self.t] = init_state
        if random_init:
            init_state = self._random_init_state(init_state)

        info = {}
        self.infos[self.t] = info
        return init_state, info

    def step(self, action: float) -> Tuple["next_state", "reward", "terminated"]:
        """
        step the env with given action, and get next state and reward of this action
        """
        if self.terminated == True or self.truncated==True:
            print("This treat is end, please call reset or export")
            return None, None, True

        self.actions[self.t] = action
        state_next = self.Simulator.update(
            action=self.actions[self.t], state=self.states[self.t])
        self.states[self.t + 1] = state_next

        # check whether the treatment end
        if self.t + 1 == self.max_t - 1:
            self.terminated = True

        reward = self.Reward.count_reward(state=self.states[self.t])
        self.rewards[self.t] = reward

        info_next = {}
        self.infos[self.t+1] = info_next

        self.t += 1
        return state_next, reward, self.terminated, self.truncated, info_next

    def export(self) -> Batch:
        """
        export the batch data generated during the interaction
        """
        batch = Batch(
            obs=self.states[:self.t + 1], act=self.actions[:self.t], rew=self.rewards[:self.t], info=self.infos[:self.t])
        return batch


class TestNooriDiabetesEnv(TestCase):
    def __init__(self, methodName: str, Env: NooriDiabetesEnv) -> None:
        super().__init__(methodName)
        self.Env = Env
        self.tested_name = "NooriDiabetesEnv"

    def test_reset(self):
        print(
            f"--------------------{self.tested_name} test reset--------------------")
        state = self.Env.reset()
        print(state)

    def test_step(self):
        print(
            f"--------------------{self.tested_name} test step--------------------")
        action = 20.0
        state1, reward1, terminated = self.Env.step(action)
        state2, reward2, terminated = self.Env.step(action)
        state3, reward3, terminated = self.Env.step(action)
        print(f"1st:\tstate:{state1}\treward:{reward1}")
        print(f"2st:\tstate:{state2}\treward:{reward2}")
        print(f"3st:\tstate:{state3}\treward:{reward3}")

    def test_export(self):
        print(
            f"--------------------{self.tested_name} test export--------------------")
        batch = self.Env.export()
        print(batch)
