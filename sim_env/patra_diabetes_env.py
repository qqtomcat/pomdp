from typing import Tuple, Union
from unittest import TestCase

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from tianshou.data import Batch
from collections import OrderedDict

from .base import BaseSimulator, BaseReward

"""
Env used in the paper 'Agent-based Simulation for Bolld Glucose Control'
time unit is 1 min
action unit is muU/ml

State:
    {G, X, I}
    G refers to the plasma glucose concentration at time t
    X refers to the generalized insulin variable for the remote compartment
    I refers to the plasma insulin concentration at time t

Action:
    A Float

Reward
    A float.
"""

"""
ODE model used in the paper 'Agent-based Simulation for Bolld Glucose Control'
time unit is 1 min
action unit is muU/ml
"""


class PatraReward(BaseReward):
    def __init__(self):
        super().__init__()

    def count_reward(self, state) -> float:
        G = state["G"]
        reward = - np.abs(G - 80)
        return float(reward)


class TestPatraReward(TestCase):
    def __init__(self, methodName: str, Reward: PatraReward) -> None:
        super().__init__(methodName)
        self.Reward = Reward
        self.tested_name = "PatraReward"

    def test_count_reward(self):
        print(f"--------------------{self.tested_name} test count_reward--------------------")
        state = {'G': 193.95971450054037, 'X': 0.024449829012528145, 'I': 47.26151970895934}
        reward = self.Reward.count_reward(state)
        print(reward)


class PatraODE(BaseSimulator):
    def __init__(self):
        super().__init__()
        """
        State:
            G refers to the plasma glucose concentration at time t
            X refers to the generalized insulin variable for the remote compartment
            I refers to the plasma insulin concentration at time t
        """

    def activate(self) -> Tuple["init_state"]:
        init_state = {"G": np.array([np.random.randint(180, 220)], dtype=np.float32),
                      "X": np.array([0], dtype=np.float32),
                      "I": np.array([np.random.randint(50, 60)], dtype=np.float32)}
        return OrderedDict(init_state)

    def update(self, action: Union[dict, float], state: dict, t: int, integral_num: int = 2) -> Tuple["next_state"]:
        G, X, I = state["G"], state["X"], state["I"]
        for i in range(integral_num):
            u = action
            G_dot = -0.0316 * (G - 70) - X * G
            X_dot = -0.0107 * X + 5.3e-4 * (I - 7)
            I_dot = -0.264 * (I - 7) + 0.0042 * max(G - 80.2576, 0) * t + u
            G, X, I = G + G_dot / integral_num, X + X_dot / \
                      integral_num, I + I_dot / integral_num
        # count death probability
        return OrderedDict({"G": G, "X": X, "I": I})


class TestPatraODE(TestCase):
    def __init__(self, methodName: str, Simulator: PatraODE) -> None:
        super().__init__(methodName)
        self.Simulator = Simulator
        self.tested_name = "PatraODE"

    def test_activate(self):
        print(
            f"--------------------{self.tested_name} test activate--------------------")
        state = self.Simulator.activate()
        print(state)

    def test_update(self):
        print(
            f"--------------------{self.tested_name} test update--------------------")
        action = 20
        state = {"G": 200, "X": 0, "I": 55}
        print(self.Simulator.update(action=20, state=state, t=1, integral_num=5))


class PatraDiabetesEnv(gym.Env):
    def __init__(self, max_t: int = 600):
        super().__init__()
        self.Simulator = PatraODE()
        self.Reward = PatraReward()
        self.max_t = max_t
        self.observation_space = spaces.Dict(
            {
                'G': spaces.Box(low=0.0, high=300.0, shape=(1,), dtype=np.float32),
                'X': spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),
                'I': spaces.Box(low=0.0, high=200.0, shape=(1,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(
            low=0.0, high=100.0, shape=(), dtype=np.float32)

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
            action=self.actions[self.t], state=self.states[self.t], t=self.t + 1)
        self.states[self.t + 1] = state_next

        # check whether the treatment end
        if self.t + 1 == self.max_t - 1:
            self.terminated = True

        reward = self.Reward.count_reward(state=self.states[self.t])
        self.rewards[self.t] = reward

        info_next = {}
        self.infos[self.t] = info_next

        self.t += 1
        return state_next, reward, self.terminated, self.truncated, info_next

    def export(self) -> Batch:
        """
        export the batch data generated during the interaction
        """
        batch = Batch(
            obs=self.states[:self.t + 1], act=self.actions[:self.t], rew=self.rewards[:self.t], info=self.infos[:self.t])
        return batch


class TestPatraDiabetesEnv(TestCase):
    def __init__(self, methodName: str, Env: PatraDiabetesEnv) -> None:
        super().__init__(methodName)
        self.Env = Env
        self.tested_name = "PatraDiabetesEnv"

    def test_reset(self):
        print(
            f"--------------------{self.tested_name} test reset--------------------")
        state = self.Env.reset()
        print(state)

    def test_step(self):
        print(
            f"--------------------{self.tested_name} test step--------------------")
        action = 20.1
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
