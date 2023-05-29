from typing import Tuple, Union
from unittest import TestCase
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium import ActionWrapper
from tianshou.data import Batch
from collections import OrderedDict

from .base import BaseReward
from .base import BaseSimulator

"""
Env used in the paper 'Reinforcement Learning Based Control of Tumor Growth with Chemotherapy'

State:
    {N, T, C, B}
    N refers to total Natural Killers cell population
    T refers to tumour cell population
    C refers to the number of circulating lymphocytes
    B refers to the chemotherapy drug concentration in the blood system (different with the original identity in the paper)

Action:
    A float between 0 and 1. 1 refers to the maximum dose level, vice versa.

Reward:
    A float.
"""


class HassaniReward(BaseReward):
    def __init__(self):
        super().__init__()

    def count_reward(self, state, state_next) -> float:
        T, T_next = state["T"], state_next["T"]
        reward = np.log(T / T_next)
        return reward


class TestHassaniReward(TestCase):
    def __init__(self, methodName: str, Reward: HassaniReward) -> None:
        super().__init__(methodName)
        self.Reward = Reward
        self.tested_name = "HassaniReward"

    def test_count_reward(self):
        print(f"--------------------{self.tested_name} test count_reward--------------------")
        state = {"N": 1, "T": 1, "C": 1, "B": 0}
        state_next = {"N": 1, "T": 0.25, "C": 0.1, "B": 0}
        reward = self.Reward.count_reward(state, state_next)
        print(reward)


class HassaniODE(BaseSimulator):
    def __init__(self):
        super().__init__()
        """
        State:
            N refers to total Natural Killers cell population
            T refers to tumour cell population
            C refers to the number of circulating lymphocytes
            B refers to the chemotherapy drug concentration in the blood system (different with the original identity in the paper)
        """

    def activate(self) -> Tuple["init_state"]:
        init_state = OrderedDict({"N": np.array([1.],dtype=np.float32),
                                    "T": np.array([1.],dtype=np.float32),
                                    "C": np.array([1.],dtype=np.float32), 
                                    "B": np.array([0.],dtype=np.float32)})
        return init_state

    def update(self, action: Union[dict, float], state: dict, integral_num: int = 1) -> Tuple["next_state"]:
        assert 0 <= action <= 1
        N, T, C, B = state["N"], state["T"], state["C"], state["B"]
        for i in range(integral_num):
            if i == 0:
                u = action
            else:
                u = 0
        T_dot = 4.31e-2 * T * (1 - 1.02e-14 * T) - 3.41e-10 * N * T - 0.8 * B * T
        N_dot = 1.2e-4 - 4.12e-2 * N + 1.5e-2 * (T / (2.02 + T)) * N - 2.0e-11 * N * T - 0.6 * N * B
        C_dot = 7.5e-8 - 1.2e-2 * C - 0.6 * B * C
        B_dot = -0.9 * B + u
        N, T, C, B = N + N_dot / integral_num, T + T_dot / integral_num, C + C_dot / integral_num, B + B_dot / integral_num
        # count death probability
        return OrderedDict({"N": N, "T": T, "C": C, "B": B})


class TestHassaniODE(TestCase):
    def __init__(self, methodName: str, Simulator: HassaniODE) -> None:
        super().__init__(methodName)
        self.Simulator = Simulator
        self.tested_name = "HassaniODE"

    def test_activate(self):
        print(
            f"--------------------{self.tested_name} test activate--------------------")
        state = self.Simulator.activate()
        print(state)

    def test_update(self):
        print(
            f"--------------------{self.tested_name} test update--------------------")
        action = np.random.random()
        state = {"N": 1, "T": 0.25, "C": 0.1, "B": 0}
        print(self.Simulator.update(action=action, state=state, integral_num=1))


class HassaniChemoEnv(gym.Env):
    def __init__(self, max_t: int = 30):
        self.Simulator = HassaniODE()
        self.Reward = HassaniReward()
        self.max_t = max_t
        self.observation_space = spaces.Dict(
            {
                'N': spaces.Box(low=0.0, high=2.0, shape=(1,), dtype=np.float32),
                'T': spaces.Box(low=0.0, high=2.0, shape=(1,), dtype=np.float32),
                'C': spaces.Box(low=0.0, high=2.0, shape=(1,), dtype=np.float32),
                'B': spaces.Box(low=0.0, high=2.0, shape=(1,), dtype=np.float32)
            }
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(), dtype=np.float32)

    def reset(self, seed:int = None, random_init:bool=True, **kwargs) -> Tuple["init_state"]:
        """
        called by the agent to generate init state and return
        """
        super().reset(seed=seed)
        # init parameters
        self.states = [None] * self.max_t
        self.actions = [None] * self.max_t
        self.rewards = [None] * self.max_t
        self.infos = [None] * self.max_t
        self.t = 0
        self.terminated = False
        self.truncated = False

        info = {}
        # get init state
        init_state = self.Simulator.activate()
        if random_init:
            init_state = self._random_init_state(init_state)
        self.states[self.t] = init_state
        
        return init_state, info

    def _random_init_state(self, init_state:dict)->dict:
        random_state = OrderedDict()
        for key, value in init_state.items():
            random_value = self.np_random.normal(value, 0.1*value, size=(1,))
            random_value = np.maximum(random_value, np.array([0.0]))
            random_value = np.minimum(random_value, np.array([2.0]))
            random_value = random_value.astype(np.float32)
            random_state[key] = random_value
        return random_state

    def step(self, action: float) -> Tuple["next_state", "reward", "terminated"]:
        """
        step the env with given action, and get next state and reward of this action
        """
        assert 0 <= action <= 1, "please give valid action"
        if self.terminated == True or self.truncated==True:
            print("This treat is end, please call reset or export")
            return None, None, self.terminated, self.truncated, {}

        self.actions[self.t] = action
        state_next = self.Simulator.update(
            action=self.actions[self.t], state=self.states[self.t])
        self.states[self.t + 1] = state_next

        if self.states[self.t + 1]["T"] == 0:
            self.truncated = True
        if self.t + 1 == self.max_t - 1:
            self.terminated = True

        reward = self.Reward.count_reward(
            state=self.states[self.t], state_next=state_next)
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


class TestHassaniChemoEnv(TestCase):
    def __init__(self, methodName: str, Env: HassaniChemoEnv) -> None:
        super().__init__(methodName)
        self.Env = Env
        self.tested_name = "HassaniChemoEnv"

    def test_reset(self):
        print(
            f"--------------------{self.tested_name} test reset--------------------")
        state = self.Env.reset()
        print(state)

    def test_step(self):
        print(
            f"--------------------{self.tested_name} test step--------------------")
        action = np.random.random()
        state1, reward1, terminated1 = self.Env.step(action, integral_num=1)
        state2, reward2, terminated2 = self.Env.step(action, integral_num=1)
        state3, reward3, terminated3 = self.Env.step(action, integral_num=1)
        print(f"1st:\tstate:{state1}\treward:{reward1}")
        print(f"2st:\tstate:{state2}\treward:{reward2}")
        print(f"3st:\tstate:{state3}\treward:{reward3}")

    def test_export(self):
        print(
            f"--------------------{self.tested_name} test export--------------------")
        batch = self.Env.export()
        print(batch)
