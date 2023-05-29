from typing import Tuple
from unittest import TestCase

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from tianshou.data import Batch
from collections import OrderedDict
from gymnasium import ActionWrapper

from .base import BaseSimulator, BaseReward

"""
Env used in the paper 'Reinforcement learning in models of adaptive medical treatment strategies'

State:
    {N_n, N_t, Gamma_n, Gamma_t}
    N_n refers to the population of normal cells
    N_t refers to the population of tumour cells
    Gamma_n refers to the radio dose equivalent for normal cells (Gy)
    Gamma_t refers to the radio dose equivalent for tumour cells (Gy)

Action:
    {d, t_r, t_i}
    d refers to the total radio dosage for this treatment (Gy)
    t_r refers to the radiation exposure time (min)
    t_i refers to the time inteval between two treatment (day)

Reward
    An int.
"""


class VincentReward(BaseReward):
    def __init__(self):
        super().__init__()

    def count_reward(self, state: dict, state_0: dict, terminated: bool) -> float:
        N_n, N_t = state["N_n"], state["N_t"]
        N_n0, N_t0 = state_0["N_n"], state_0["N_t"]
        sigma_n = 1 if N_n > N_n0 else N_n / N_n0
        sigma_t = 1 if N_t > N_t0 else N_t / N_t0
        if sigma_n <= 0.9:
            reward = -1
        elif sigma_n > 0.9 and terminated:
            reward = 1 - sigma_t
        else:
            reward = 0
        return float(reward)


class TestVincentReward(TestCase):
    def __init__(self, methodName: str, Reward: VincentReward) -> None:
        super().__init__(methodName)
        self.Reward = Reward
        self.tested_name = "VincentReward"

    def test_count_reward(self):
        print(
            f"--------------------{self.tested_name} test count_reward--------------------")
        state_0 = {"N_n": 1e11, "N_t": 1e11, "Gamma_n": 0., "Gamma_t": 0}
        state = {"N_n": 9.52e10, "N_t": 15.42e10, "Gamma_n": 0., "Gamma_t": 0}
        reward = self.Reward.count_reward(state=state, state_0=state_0, terminated=False)
        self.assertEqual(reward, 0)
        print(reward)


class VincentODE(BaseSimulator):
    def __init__(self):
        super().__init__()
        """
        State:
            N_n refers to the population of normal cells
            N_t refers to the population of tumour cells
            Gamma_n refers to the radio dose equivalent for normal cells
            Gamma_t refers to the radio dose equivalent for tumour cells
        Action:
            d refers to the total radio dosage for this treatment
            t_r refers to the radiation exposure time (time unit is 1 min)
            t_i refers to the time inteval between two treatment (time unit is 1 day)

        As t can be modified by action, the operation timestamp is detached from unit time, 
        which is more closer to the realism.
        """

    def activate(self) -> Tuple["init_state"]:
        N_n = np.random.randint(1e10, 1e11)
        N_t = np.random.randint(1e10, 1e11)
        init_state = {"N_n": N_n, "N_t": N_t,
                       "Gamma_n": np.array([0.], dtype=np.float32),
                       "Gamma_t": np.array([0.], dtype=np.float32)}
        return OrderedDict(init_state)

    def update(self, action: dict, state: dict, integral_num: int = 60 * 24) -> Tuple["next_state"]:
        N_n, N_t, Gamma_n, Gamma_t = state["N_n"], state["N_t"], state["Gamma_n"], state["Gamma_t"]
        d, t_r, t_i = action["d"], action['t_r'], action['t_i']

        r = d / t_r
        r = r * 60 * 24  # r's unit is Gy/min， it should be transmited to time unit Gy/day

        for i in range(int(t_i * integral_num)):
            if i < t_r:
                R = r
            else:
                R = 0

            if i / integral_num < 2:
                N_n_dot = -(0.15 + 2 * 0.079 * Gamma_n) * R * N_n
                N_t_dot = -(1.43 + 2 * 0.13 * Gamma_t) * R * N_t
            else:
                N_n_dot = -(0.15 + 2 * 0.079 * Gamma_n) * R * N_n + 0.15 * N_n * (1 - N_n / 2e11)
                N_t_dot = -(1.43 + 2 * 0.13 * Gamma_t) * R * N_t + 0.15 * N_t * (1 - N_t / 2e11)
            Gamma_n_dot = R - 71 * Gamma_n ** 2
            Gamma_t_dot = R - 40 * Gamma_t ** 2

            N_n, N_t, = max(N_n + N_n_dot / integral_num, 0), max(N_t + N_t_dot / integral_num, 0)
            Gamma_n = np.maximum(Gamma_n + Gamma_n_dot / integral_num, np.array([0.0], dtype=np.float32))
            Gamma_t = np.maximum(Gamma_t + Gamma_t_dot / integral_num, np.array([0.0], dtype=np.float32))
        return OrderedDict({"N_n": N_n, "N_t": N_t, "Gamma_n": Gamma_n, "Gamma_t": Gamma_t})


class TestVincentODE(TestCase):
    def __init__(self, methodName: str, Simulator: VincentODE) -> None:
        super().__init__(methodName)
        self.Simulator = Simulator
        self.tested_name = "VincentODE"

    def test_activate(self):
        print(
            f"--------------------{self.tested_name} test activate--------------------")
        state = self.Simulator.activate()
        print(state)

    def test_update(self):
        print(
            f"--------------------{self.tested_name} test update--------------------")
        action = {"d": 0.64, "t_r": 2, "t_i": 1}
        state = {"N_n": 1e11, "N_t": 1e11, "Gamma_n": 0., "Gamma_t": 0}
        print(self.Simulator.update(action=action, state=state, integral_num=24 * 60))


class VincentRadioEnv(gym.Env):
    def __init__(self, max_t: int = 10):
        super().__init__()
        self.Simulator = VincentODE()
        self.Reward = VincentReward()
        self.max_t = max_t
        self.observation_space = spaces.Dict(
            {
                'N_n': spaces.Discrete(n=200000000000),
                'N_t': spaces.Discrete(n=200000000000),
                'Gamma_n': spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),
                'Gamma_t': spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32)
            }
        )
        self.action_space = spaces.Dict(
            {
                "d": spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
                "t_r": spaces.Discrete(60),
                "t_i": spaces.Discrete(30)
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
        init_state = self.Simulator.activate()
        self.states[self.t] = init_state
        info = {}
        self.infos[self.t] = info
        return init_state, info

    def step(self, action: dict) -> Tuple["next_state", "reward", "terminated"]:
        """
        step the env with given action, and get next state and reward of this action
        """
        if self.terminated == True or self.truncated == True:
            print("This treat is end, please call reset or export")
            return None, None, self.terminated, self.truncated, {}
        
        self.actions[self.t] = action
        state_next = self.Simulator.update(
            action=self.actions[self.t], state=self.states[self.t])
        self.states[self.t + 1] = state_next

        # check whether the treatment end
        if self.t + 1 == self.max_t - 1:
            self.terminated = True
        if state_next["N_t"] == 0:
            self.truncated = True

        reward = self.Reward.count_reward(
            state=self.states[self.t + 1], state_0=self.states[0], terminated=self.terminated)
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


# class TestVincentRadioEnv(TestCase):
#     def __init__(self, methodName: str, Env: VincentRadioEnv) -> None:
#         super().__init__(methodName)
#         self.Env = Env
#         self.tested_name = "VincentRadioEnv"

#     def test_reset(self):
#         print(
#             f"--------------------{self.tested_name} test reset--------------------")
#         state = self.Env.reset()
#         print(state)

#     def test_step(self):
#         print(
#             f"--------------------{self.tested_name} test step--------------------")
#         action = {"d": 0.64, "t_r": 2, "t_i": 1}
#         state1, reward1, terminated = self.Env.step(action)
#         state2, reward2, terminated = self.Env.step(action)
#         state3, reward3, terminated = self.Env.step(action)
#         print(f"1st:\tstate:{state1}\treward:{reward1}")
#         print(f"2st:\tstate:{state2}\treward:{reward2}")
#         print(f"3st:\tstate:{state3}\treward:{reward3}")

#     def test_export(self):
#         print(
#             f"--------------------{self.tested_name} test export--------------------")
#         batch = self.Env.export()
#         print(batch)

