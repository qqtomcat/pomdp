from typing import Tuple, Union
from unittest import TestCase

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from tianshou.data import Batch
from collections import OrderedDict

from .base import BaseSimulator, BaseReward

"""
Env used in the paper 'Reinforcement Learning Desigh for Cancer Clinical Trials'

State:
    {W, M, death_prob, is_dead}
    W refers to the toxicity accumulation degree
    M refers to the size of tumour
    death_prob refers to the probability of death at time t
    is_dead is 1 if the patient is dead

Action:
    A float between 0 and 1. 1 refers to the maximum dose level, vice versa.

Reward:
    An int. Max(reward)=15 if the patient is cured. Min(reward)=-60 if the patient is dead.
"""


class ZhaoReward(BaseReward):
    def __init__(self):
        super().__init__()

    def count_reward(self, state, state_next) -> float:
        if state_next["is_dead"] == 1:
            reward = -60.0
        else:
            W_next = state_next["W"]
            W = state["W"]
            W_reward = self._W_reward(W, W_next)
            M_next = state_next["M"]
            M = state["M"]
            M_reward = self._M_reward(M, M_next)
            reward = W_reward + M_reward
        return float(reward)

    def _M_reward(self, M, M_next) -> float:
        if M_next == 0:
            return 15.0
        elif M_next - M <= -0.5:
            return 5.0
        elif M_next - M >= 0.5:
            return -5.0
        else:
            return 0.0

    def _W_reward(self, W, W_next) -> float:
        if W_next - W <= -0.5:
            return 5.0
        elif W_next - W >= 0.5:
            return -5.0
        else:
            return 0


class TestZhaoReward(TestCase):
    def __init__(self, methodName: str, Reward: ZhaoReward) -> None:
        super().__init__(methodName)
        self.Reward = Reward
        self.tested_name = "ZhaoReward"

    def test_count_reward(self):
        print(f"--------------------{self.tested_name} test count_reward--------------------")
        state = {"W": 0.2, "M": 0.2, "death_prob": 0.0, "is_dead": 0}
        state_next = {"W": 0.5, "M": 0.4, "death_prob": 0.5, "is_dead": 0}
        reward = self.Reward.count_reward(state=state, state_next=state_next)
        print(reward)


class ZhaoODE(BaseSimulator):
    def __init__(self):
        super().__init__()

    def activate(self) -> Tuple["init_state"]:
        W = 2 * np.random.random()
        M = 2 * np.random.random()

        init_state = OrderedDict({"W": np.array([W], dtype=np.float32),
                            "M": np.array([M], dtype=np.float32),
                            "death_prob": np.array([0.0], dtype=np.float32),
                            "is_dead": 0})
        return init_state

    def update(self, action: Union[dict, float], state: dict, integral_num: int = 1) -> Tuple["next_state"]:
        W = state["W"]
        M = state["M"]
        D = action

        for i in range(integral_num):
            if i == 0:
                u = action
                W_dot = 0.1 * M + 1.2 * (D - 0.5)
                M_dot = 0.15 * W - 1.2 * (D - 0.5) if M > 0 else 0
                W_next = W + W_dot / integral_num
                M_next = M + M_dot / integral_num
            else:
                u = 0
                W_dot = 0.1 * M + 1.2 * (D - 0.5)
                M_dot = 0.15 * W - 1.2 * (D - 0.5) if M > 0 else 0
                W_next += W_dot / integral_num
                M_next += M_dot / integral_num

        # count death probability
        lamb = np.exp(0 + 1 * W + 1 * M)
        lamb_next = np.exp(0 + 1 * W_next + 1 * M_next)
        delta = (lamb + lamb_next) * 1 / 2
        F = np.exp(-delta)
        p = 1 - F
        is_dead = 0 if np.random.random() > p else 1
        state = {"W": W_next, "M": M_next, "death_prob": p, "is_dead": is_dead}
        return OrderedDict(state)


class TestZhaoODE(TestCase):
    def __init__(self, methodName: str, Simulator: ZhaoODE) -> None:
        super().__init__(methodName)
        self.Simulator = Simulator
        self.tested_name = "ZhaoODE"

    def test_activate(self):
        print(f"--------------------{self.tested_name} test activate--------------------")
        state = self.Simulator.activate()
        print(state)

    def test_update(self):
        print(f"--------------------{self.tested_name} test update--------------------")
        action = np.random.random()
        state = {"W": 2 * np.random.random(), "M": 2 * np.random.random()}
        print(self.Simulator.update(action=action, state=state, integral_num=2))


class ZhaoChemoEnv(gym.Env):

    def __init__(self, max_t: int = 10):
        self.Simulator = ZhaoODE()
        self.Reward = ZhaoReward()
        self.max_t = max_t
        self.observation_space = spaces.Dict(
            {
                "W": spaces.Box(low=0.0, high=5.0, shape=(1,), dtype=np.float32),
                "M": spaces.Box(low=0.0, high=5.0, shape=(1,), dtype=np.float32),
                "death_prob": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "is_dead": spaces.Discrete(1)
            }
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(), dtype=np.float32)
        self.reward_range = (-60, 20)

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
        init_state = self.Simulator.activate()
        self.states[self.t] = init_state
        info = {}
        self.infos[self.t] = info
        return init_state, info

    def step(self, action: float) -> Tuple["next_state", "reward", "terminated"]:
        """
        update the env with given action, and get next state and reward of this action
        """
        assert 0 <= action <= 1, "please give valid action"
        if self.terminated == True or self.truncated == True:
            print("This treat is end, please call reset or export")
            return None, None, self.terminated, self.truncated, {}

        self.actions[self.t] = action
        state_next = self.Simulator.update(
            action=self.actions[self.t], state=self.states[self.t])
        # check whether the patient is dead
        self.states[self.t + 1] = state_next
        if self.states[self.t + 1]["M"] < 0 or self.t + 1 == self.max_t - 1:
            self.terminated = True
        if self.states[self.t + 1]["is_dead"] == 1:
            self.truncated = True

        reward = self.Reward.count_reward(
            state=self.states[self.t], state_next=state_next)
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
            obs=self.states[:self.t], act=self.actions[:self.t], rew=self.rewards[:self.t], info=self.infos[:self.t])
        return batch


class TestZhaoChemoEnv(TestCase):
    def __init__(self, methodName: str, Env: ZhaoChemoEnv) -> None:
        super().__init__(methodName)
        self.Env = Env
        self.tested_name = self.Env.__class__.__name__

    def test_reset(self):
        print(
            f"--------------------{self.tested_name} test reset--------------------")
        state = self.Env.reset()
        print(state)

    def test_step(self):
        print(
            f"--------------------{self.tested_name} test step--------------------")
        action = np.random.random()
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
