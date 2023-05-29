from typing import Tuple
from unittest import TestCase
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from tianshou.data import Batch
from collections import OrderedDict

from .base import BaseSimulator, BaseReward

"""
Env used in the paper 'Control of Blood Glucose for Type-1 Diabetes by Using Reinforcement Learning with Feedforward Algorithm'
and "Reinforcement-Learning Optimal Control for Type-1 Diabetes"
time unit is 1 hour
action unit is muU/ml

 State:
    {D1, D2, g, x}
    D1: Amount of Glucose in Compartment 1
    D2: Amount of Glucose in Compartment 2
    g: Plasma glucode concentration
    x: Interstitial insulin activity

Action:
    {D, i}
    D: Amount of CHO intake
    i: Plasma insulin concentration

Reward
    A float.
"""

"""
ODE model used in the paper 'Control of Blood Glucose for Type-1 Diabetes by Using Reinforcement Learning with Feedforward Algorithm'
and "Reinforcement-Learning Optimal Control for Type-1 Diabetes"
time unit is 1 hour
action unit is muU/ml
"""


class NgoODE(BaseSimulator):
    def __init__(self):
        super().__init__()
        """
        State:
            {D1, D2, g, x}
            D1: Amount of Glucose in Compartment 1
            D2: Amount of Glucose in Compartment 2
            g: Plasma glucode concentration
            x: Interstitial insulin activity

        Action:
            {D, i}
            D: Amount of CHO intake
            i: Plasma insulin concentration
        """

    def activate(self) -> Tuple["init_state"]:
        init_state = {"D1": np.array([np.random.randint(0, 10)], dtype=np.float32),
                   "D2": np.array([np.random.randint(0, 10)], dtype=np.float32),
                   "g": np.array([np.random.randint(60, 130)], dtype=np.float32),
                   "x": np.array([0.0], dtype=np.float32)
                   }
        return OrderedDict(init_state)

    def update(self, action: dict, state: dict, integral_num: int = 60) -> Tuple["next_state"]:
        D1, D2, g, x = state["D1"], state["D2"], state["g"], state["x"]
        D, i = action["D"], action["i"]
        for _ in range(integral_num):
            D1_dot = 0.8 * D - D1 / 10
            D2_dot = D1 / 10 - D2 / 10
            g_dot = -0.2 * g - x * g + D2 / 10
            x_dot = -0.028 * x + 1e-4 * 2730 * (i - 7.326)
            D1, D2, g, x = D1 + D1_dot / integral_num, D2 + D2_dot / \
                           integral_num, g + g_dot / integral_num, x + x_dot / integral_num
        # count death probability
        return OrderedDict({"D1": np.maximum(D1, np.array([0.0], dtype=np.float32)),
                            "D2": np.maximum(D2, np.array([0.0], dtype=np.float32)),
                            "g": np.maximum(g, np.array([0.0], dtype=np.float32)),
                            "x": np.maximum(x, np.array([0.0], dtype=np.float32))})


class NgoReward(BaseReward):
    def __init__(self):
        super().__init__()

    def count_reward(self, state) -> float:
        g, x = state["g"], state["x"]
        reward = - ((g - 80) ** 2 + 0.1 * x ** 2)
        return float(reward)


class TestNgoReward(TestCase):
    def __init__(self, methodName: str, Reward: NgoReward) -> None:
        super().__init__(methodName)
        self.Reward = Reward
        self.tested_name = "NgoReward"

    def test_count_reward(self):
        print(f"--------------------{self.tested_name} test count_reward--------------------")
        state = {'D1': 38.43168128, 'D2': 8.7683212544, 'g': 151.25225338282326, 'x': 0}
        reward = self.Reward.count_reward(state)
        print(reward)


class TestNgoODE(TestCase):
    def __init__(self, methodName: str, Simulator: NgoODE) -> None:
        super().__init__(methodName)
        self.Simulator = Simulator
        self.tested_name = "NgoODE"

    def test_activate(self):
        print(
            f"--------------------{self.tested_name} test activate--------------------")
        state = self.Simulator.activate()
        print(state)

    def test_update(self):
        print(
            f"--------------------{self.tested_name} test update--------------------")
        action = {"D": 100, "i": 8}
        state = {"D1": np.random.randint(0, 10),
                 "D2": np.random.randint(0, 10),
                 "g": np.random.randint(60, 130),
                 "x": 1.0
                 }
        print(self.Simulator.update(action=action, state=state, integral_num=5))


class NgoDiabetesEnv(gym.Env):
    def __init__(self, max_t: int = 24):
        super().__init__()
        self.Simulator = NgoODE()
        self.Reward = NgoReward()
        self.max_t = max_t
        self.observation_space = spaces.Dict(
            {
                'D1': spaces.Box(low=0.0, high=500.0, shape=(1,), dtype=np.float32),
                'D2': spaces.Box(low=0.0, high=500.0, shape=(1,), dtype=np.float32),
                'g': spaces.Box(low=0.0, high=600.0, shape=(1,), dtype=np.float32),
                'x': spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Dict(
            {
                'D': spaces.Box(low=0.0, high=500.0, shape=(1,), dtype=np.float32),
                'i': spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),
            }
        )

    def reset(self, seed: int = None, random_init:bool=True, **kwargs) -> Tuple["init_state"]:
        assert random_init == True, "This environment can only have a random initial state"
        super().reset(seed=seed)
        np.random.seed(seed)

        # init parameters
        self.states = [None] * self.max_t
        self.actions = [None] * self.max_t
        self.rewards = [None] * self.max_t
        self.infos = [None] * self.max_t
        self.t = 0
        self.terminated = False
        self.truncated = False

        # get init state
        state = self.Simulator.activate()
        self.states[self.t] = state
        info = {}
        self.infos[self.t] = info

        return state, info

    def step(self, action: float) -> Tuple["next_state", "reward", "terminated"]:
        """
        step the env with given action, and get next state and reward of this action
        """
        if self.terminated == True or self.truncated==True:
            print("This treat is end, please call reset or export")
            return None, None, self.terminated, self.truncated, {}

        self.actions[self.t] = action
        state_next = self.Simulator.update(
            action=self.actions[self.t], state=self.states[self.t])
        self.states[self.t + 1] = state_next

        # check whether the treatment end
        if self.t + 1 == self.max_t - 1:
            self.terminated = True

        reward = self.Reward.count_reward(state=self.states[self.t])
        self.rewards[self.t] = reward
        
        info = {}
        self.infos[self.t+1] = info

        self.t += 1
        return state_next, reward, self.terminated, self.truncated, info

    def export(self) -> Batch:
        """
        export the batch data generated during the interaction
        """
        batch = Batch(
            obs=self.states[:self.t + 1], act=self.actions[:self.t], rew=self.rewards[:self.t], info=self.infos[:self.t])
        return batch


class TestNgoDiabetesEnv(TestCase):
    def __init__(self, methodName: str, Env: NgoDiabetesEnv) -> None:
        super().__init__(methodName)
        self.Env = Env
        self.tested_name = "NgoDiabetesEnv"

    def test_reset(self):
        print(
            f"--------------------{self.tested_name} test reset--------------------")
        state = self.Env.reset()
        print(state)

    def test_step(self):
        print(
            f"--------------------{self.tested_name} test step--------------------")
        action = {"D": 100, "i": 8}
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
