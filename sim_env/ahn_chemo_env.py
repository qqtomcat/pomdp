from unittest import TestCase
from typing import Tuple, Union, Dict, Any
import gymnasium as gym
from gymnasium import spaces
from gymnasium import ActionWrapper
import numpy as np
from tianshou.data import Batch
from collections import OrderedDict

from .base import BaseSimulator, BaseReward


"""
Env used in the paper 'Drug Scheduling of Cancer Chemotherapy based on Natural Actor-critic Approach'
    and 'Reinforcem net Learning-based control of drug dosing for cancer chemotherapy treatment'

State:
    {N, T, I, B}
    N refers to the number of normal cell
    T refers to the number of tumour cell
    I refers to the number of immune cell
    B refers to the drug concentration in the blood system

Action:
    A float between 0 and 1. 1 refers to the maximum dose level, vice versa.

Reward
    A float. 
"""


class AhnReward(BaseReward):
    def __init__(self):
        super().__init__()

    def count_reward(self, state, action, reward_last) -> float:
        N, T, I = state["N"], state["T"], state["I"]
        u = action
        reward_dot = N - T + I - u
        reward = reward_last + reward_dot
        return reward


class AhnODE(BaseSimulator):
    def __init__(self):
        super().__init__()
        """
        State:
            N refers to the number of normal cell
            T refers to the number of tumour cell
            I refers to the number of immune cell
            B refers to the drug concentration in the blood system
        """

    def activate(self) -> Dict[str, Union[float, int]]:
        """
        The default init state is the case 2 in the paper 'Drug scheduling of cancer chemotherapy ...'
        as its simulation results looks more plausible.
        params:
            random_init:bool
        """
        init_state = OrderedDict({"N": np.array([1.0],dtype=np.float32), 
                                  "T": np.array([0.25],dtype=np.float32),
                                  "I": np.array([0.1],dtype=np.float32), 
                                    "B": np.array([0.0],dtype=np.float32),})
        return init_state

    def update(self, action: Union[dict, float], state: dict, integral_num: int = 5) -> Dict[str, Union[float, Any]]:
        assert 0 <= action <= 1
        N, T, I, B = state["N"], state["T"], state["I"], state["B"]
        for i in range(integral_num):
            if i == 0:
                u = action
            else:
                u = 0
            N_dot = 1.0 * N * (1 - 1.0 * N) - 1.0 * T * N - 0.1 * (1 - np.exp(-B)) * N
            T_dot = 1.5 * T * (1 - 1.0 * T) - 0.5 * I * T - 1.0 * T * N - 0.3 * (1 - np.exp(-B)) * T
            I_dot = 0.33 + 0.01 * I * T / (0.3 + T) - 1.0 * I * T - \
                    0.2 * I - 0.2 * (1 - np.exp(-B)) * I
            B_dot = -1.0 * B + u
            N, T, I, B = N + N_dot / integral_num, T + T_dot / \
                         integral_num, I + I_dot / integral_num, B + B_dot / integral_num
        # count death probability
        return OrderedDict({"N": N, "T": T, "I": I, "B": B})


class AhnChemoEnv(gym.Env):
    def __init__(self, max_t: int = 30):
        super().__init__()
        self.Simulator = AhnODE()
        self.Reward = AhnReward()
        self.max_t = max_t
        self.observation_space = spaces.Dict(
            {
                'N': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                'T': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                'I': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                'B': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            }
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(), dtype=np.float32)
        self.meta_data = {'action_type':'continuous', 'reward_range':(0, np.inf)}
        

    def _random_init_state(self, init_state:dict)->dict:
        random_state = OrderedDict()
        for key, value in init_state.items():
            random_value = self.np_random.normal(value, 0.1*value, size=(1,))
            random_value = np.maximum(random_value, np.array([0.0]))
            random_value = np.minimum(random_value, np.array([1.0]))
            random_value = random_value.astype(np.float32)
            random_state[key] = random_value
        return random_state

    def reset(self, seed: int = None, random_init:bool=True, **kwargs) -> Dict[str, Union[float, int]]:
        """
        params:
            seed: random seed
            random_init: whether randomize the initial state. If this is set to be True, the initial state
                    will be a sample of N(default_value, 0.1*default_value).
        """
        super().reset(seed=seed)
        self.states = [None] * self.max_t
        self.actions = [None] * self.max_t
        self.rewards = [None] * self.max_t
        self.infos = [None] * self.max_t
        self.t = 0
        self.terminated = False
        self.truncated = False
        # get init state
        init_state = self.Simulator.activate()
        if random_init:
            init_state = self._random_init_state(init_state)
        self.states[self.t] = init_state

        init_info = {}
        self.infos[self.t] = init_info
        return init_state, init_info

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

        info_next = {}
        self.infos[self.t + 1] = info_next

        # check whether the treatment end\
        if self.states[self.t + 1]["T"] == 0:
            self.truncated = True
        if self.t + 1 == self.max_t - 1:
            self.terminated = True

        reward_last = 0 if self.t == 0 else self.rewards[self.t - 1]
        reward = self.Reward.count_reward(
            state=self.states[self.t], action=action, reward_last=reward_last)
        self.rewards[self.t] = reward

        self.t += 1
        return state_next, reward, self.terminated, False, info_next

    def export(self) -> Batch:
        """
        export the batch data generated during the interaction
        """
        batch = Batch(
            obs=self.states[:self.t + 1], act=self.actions[:self.t], rew=self.rewards[:self.t], info=self.infos[:self.t])
        return batch


class TestAhnReward(TestCase):
    def __init__(self, methodName: str, Reward: AhnReward) -> None:
        super().__init__(methodName)
        self.Reward = Reward
        self.tested_name = "AhnReward"

    def test_count_reward(self):
        print(f"--------------------{self.tested_name} test count_reward--------------------")
        state = {"N": 1, "T": 0.25, "I": 0.1, "B": 0}
        action = np.random.random()
        reward_last = 0
        reward = self.Reward.count_reward(state, action, reward_last)
        print(reward)


class TestAhnODE(TestCase):
    def __init__(self, methodName: str, Simulator: AhnODE) -> None:
        super().__init__(methodName)
        self.Simulator = Simulator
        self.tested_name = "AhnODE"

    def test_activate(self):
        print(
            f"--------------------{self.tested_name} test activate--------------------")
        state = self.Simulator.activate()
        print(state)

    def test_update(self):
        print(
            f"--------------------{self.tested_name} test update--------------------")
        action = np.random.random()
        state = {"N": 1, "T": 0.25, "I": 0.1, "B": 0}
        print(self.Simulator.update(action=action, state=state, integral_num=5))


class TestAhnChemoEnv(TestCase):
    def __init__(self, methodName: str, Env: AhnChemoEnv) -> None:
        super().__init__(methodName)
        self.Env = Env
        self.tested_name = "AhnChemoEnv"

    def test_reset(self):
        print(
            f"--------------------{self.tested_name} test reset--------------------")
        state, info = self.Env.reset(random_init=True)
        print(state in self.Env.observation_space)
        print(self.Env.observation_space.sample())
        print(state)

    def test_step(self):
        print(
            f"--------------------{self.tested_name} test step--------------------")
        action = np.random.random()
        state1, reward1, terminated, truncated, info = self.Env.step(action)
        state2, reward2, terminated, truncated, info = self.Env.step(action)
        state3, reward3, terminated, truncated, info = self.Env.step(action)
        print(f"1st:\tstate:{state1}\treward:{reward1}")
        print(f"2st:\tstate:{state2}\treward:{reward2}")
        print(f"3st:\tstate:{state3}\treward:{reward3}")

    def test_export(self):
        print(
            f"--------------------{self.tested_name} test export--------------------")
        batch = self.Env.export()
        print(batch)


class AhnDiscretizeActionWrapper(ActionWrapper):
    def __init__(self, env, n_act:int):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box), "This wrapper only works with continuous action spaces (gym.spaces.Dict)"
        # Change the action space to be discrete
        self.n_act = n_act
        self.action_space = spaces.Discrete(self.n_act)
        self.meta_data = env.meta_data
        self.meta_data["action_type"] = "discrete"

    def action(self, act):
        range_min, range_max = self.env.action_space.low, self.env.action_space.high
        edges = np.linspace(range_min, range_max, self.n_act+1)
        # 使用digitize函数离散化
        digitized_act = np.digitize(act, edges, right=False) - 1
        return digitized_act
    

def create_discrete_act_ahn_env(max_t:int=30, n_act:int=5):
    env = AhnChemoEnv(max_t)
    wrapped_env = AhnDiscretizeActionWrapper(env, n_act)
    return wrapped_env

