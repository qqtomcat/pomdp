from typing import Tuple
from unittest import TestCase

import gymnasium as gym
from gymnasium import spaces
from gymnasium import ActionWrapper
import numpy as np
from tianshou.data import Batch
from collections import OrderedDict

from .base import BaseSimulator, BaseReward


class TangReward(BaseReward):
    def __init__(self):
        super().__init__()

    def count_reward(self, state) -> float:
        stage = state["stage"]
        if stage == 0:
            return -1
        elif stage == 1:
            return 0
        elif stage == 2:
            return 1


class TestTangReward(TestCase):
    def __init__(self, methodName: str, Reward: TangReward) -> None:
        super().__init__(methodName)
        self.Reward = Reward
        self.tested_name = "TangReward"

    def test_count_reward(self):
        print(f"--------------------{self.tested_name} test count_reward--------------------")
        state = {
            "hr": 0,
            "sbp": 1,
            "o2": 1,
            "glu": 0,
            "diab": 1,
            "stage": 1,
            "abx": 0,
            "vaso": 0,
            "vent": 0,
        }
        reward = self.Reward.count_reward(state)
        print(reward)


"""
Env used in the paper 'Model Selecction for Offline Reinforcement Learning: Practical Considerations for Healthcare Settings'

State:
    State:
        hr: Heart Rate, {0, 1, 2}:={"L", "N", "H"}
        sbp: Systolic Blood Pressure,  {0, 1, 2}:={"L", "N", "H"}
        o2: Percent Oxygen,  {0, 1}:={"L", "N"}
        glu: Glucose, {0, 1, 2, 3, 4}:={"LL", "L", "N", "H", "H"}
        diab: Diabetes Indicator, {0, 1}:={yes, no}
        stage: Patient's stage, {0, 1, 2}:= {dead, alive, cured}
        abx: antibiotics, {0, 1}:={off, on}
        vaso: Vasopressors, {0, 1}:={off, on}
        vent: Mechanical Ventilation, {0, 1}:={off, on}

Action:
    abx: antibiotics, {0, 1}:={off, on}
    vaso: Vasopressors, {0, 1}:={off, on}
    vent: Mechanical Ventilation, {0, 1}:={off, on}

Reward
    An int.
"""

"""
Simulator used in the paper 'Model Selecction for Offline Reinforcement Learning: Practical Considerations for Healthcare Settings'
"""


class TangSim(BaseSimulator):
    def __init__(self):
        super().__init__()
        """
        State:
            hr: Heart Rate, {0, 1, 2}:={"L", "N", "H"}
            sbp: Systolic Blood Pressure,  {0, 1, 2}:={"L", "N", "H"}
            o2: Percent Oxygen,  {0, 1}:={"L", "N"}
            glu: Glucose, {0, 1, 2, 3, 4}:={"LL", "L", "N", "H", "H"}
            diab: Diabetes Indicator, {0, 1}:={yes, no}
            stage: Patient's stage, {0, 1, 2}:= {dead, alive, cured}
            abx: antibiotics, {0, 1}:={off, on}
            vaso: Vasopressors, {0, 1}:={off, on}
            vent: Mechanical Ventilation, {0, 1}:={off, on}
        """

    def _get_patient_stage(self, hr: int, sbp: int, o2: int, glu: int) -> int:
        """
        If 3 or more vital are abnormal, the patient will die.
        If all the vital are normal, the patient is cured.
        """
        abnormal_count = 0
        if hr != 1:
            abnormal_count += 1
        if sbp != 1:
            abnormal_count += 1
        if o2 != 1:
            abnormal_count += 1
        if glu != 2:
            abnormal_count += 1

        if abnormal_count == 0:
            patient_state = 2
        elif abnormal_count < 3:
            patient_state = 1
        else:
            patient_state = 0

        return patient_state

    def activate(self) -> Tuple["init_state"]:
        while True:
            hr = np.random.randint(0, 3)
            sbp = np.random.randint(0, 3)
            o2 = np.random.randint(0, 2)
            glu = np.random.randint(0, 5)
            patient_stage = self._get_patient_stage(hr, sbp, o2, glu)
            if patient_stage == 1:
                break

        diab = 1 if np.random.random() < 0.2 else 0
        abx = np.random.randint(0, 2)
        vaso = np.random.randint(0, 2)
        vent = np.random.randint(0, 2)

        init_state = {
            "hr": int(hr),
            "sbp": int(sbp),
            "o2": int(o2),
            "glu": int(glu),
            "diab": int(diab),
            "stage": int(patient_stage),
            "abx": int(abx),
            "vaso": int(vaso),
            "vent": int(vent),
        }
        return OrderedDict(init_state)

    def update(self, action: dict, state: dict) -> Tuple["next_state"]:
        patient_stage = state["stage"]
        if patient_stage != 1:
            print("end state has reached, please reset the env")
            return None
        hr, sbp, o2, glu = state["hr"], state["sbp"], state["o2"], state["glu"]
        diab = state["diab"]
        abx_current, vaso_current, vent_current = state["abx"], state["vaso"], state["vent"]
        abx_new, vaso_new, vent_new = action["abx"], action["vaso"], action["vent"]

        # abx
        if abx_new == 1:
            hr = 1 if np.random.random() < 0.5 and hr == 2 else hr
            sbp = 1 if np.random.random() < 0.5 and sbp == 2 else sbp
        elif abx_current == 1 and abx_new == 0:
            hr = 2 if np.random.random() < 0.1 and hr == 1 else hr
            sbp = 2 if np.random.random() < 0.5 and sbp == 1 else sbp

        # vent
        if vent_new == 1:
            o2 = 1 if np.random.random() < 0.7 and o2 == 0 else o2
        elif vent_current == 1 and vent_new == 0:
            o2 = 0 if np.random.random() < 0.1 and o2 == 1 else o2

        # vaso
        if vaso_new == 1:
            if diab == 0:
                if sbp == 0:
                    sbp = 1 if np.random.random() < 0.7 else sbp
                elif sbp == 1:
                    sbp = 2 if np.random.random() < 0.7 else sbp

            elif diab == 1:
                if sbp == 0:
                    if np.random.random() < 0.5:
                        sbp = 1
                    elif np.random.random() < 0.4:
                        sbp = 2
                elif sbp == 1:
                    sbp = 2 if np.random.random() < 0.9 else sbp

                if np.random.random() < 0.5:
                    glu = min(glu + 1, 4)

        elif vaso_current == 1 and vaso_new == 0:
            if diab == 0:
                if np.random.random() < 0.1:
                    sbp = max(0, sbp - 1)
            elif diab == 1:
                if np.random.random() < 0.05:
                    sbp = max(0, sbp - 1)

        # fluctuate
        if np.random.random() < 0.1:
            hr = max(0, sbp - 1) if np.random.random() < 0.5 else min(2, sbp + 1)
            sbp = max(0, sbp - 1) if np.random.random() < 0.5 else min(2, sbp + 1)
            o2 = max(0, sbp - 1) if np.random.random() < 0.5 else min(1, sbp + 1)

        if np.random.random() < 0.3 and diab == 1:
            glu = max(0, glu - 1) if np.random.random() < 0.5 else min(4, sbp + 1)

        patient_stage = self._get_patient_stage(hr, sbp, o2, glu)

        state = {
            "hr": int(hr),
            "sbp": int(sbp),
            "o2": int(o2),
            "glu": int(glu),
            "diab": int(diab),
            "stage": int(patient_stage),
            "abx": int(abx_new),
            "vaso": int(vaso_new),
            "vent": int(vent_new),
        }
        return OrderedDict(state)


class TestTangSim(TestCase):
    def __init__(self, methodName: str, Simulator: TangSim) -> None:
        super().__init__(methodName)
        self.Simulator = Simulator
        self.tested_name = "TangSim"

    def test_activate(self):
        print(
            f"--------------------{self.tested_name} test activate--------------------")
        state = self.Simulator.activate()
        print(state)

    def test_update(self):
        print(
            f"--------------------{self.tested_name} test update--------------------")
        action = {
            "abx": 1,
            "vaso": 1,
            "vent": 1,
        }
        state = {
            "hr": 0,
            "sbp": 1,
            "o2": 1,
            "glu": 0,
            "diab": 1,
            "stage": 1,
            "abx": 0,
            "vaso": 0,
            "vent": 0,
        }
        print(self.Simulator.update(action=action, state=state))


class TangSepsisEnv(gym.Env):
    def __init__(self, max_t: int = 1000):
        super().__init__()
        self.max_t = max_t
        self.Simulator = TangSim()
        self.Reward = TangReward()
        self.observation_space = spaces.Dict(
            {
                'hr': spaces.Discrete(3),
                'sbp': spaces.Discrete(3),
                'o2': spaces.Discrete(2),
                'glu': spaces.Discrete(5),
                "stage": spaces.Discrete(3),
                "diab": spaces.Discrete(2),
                "abx": spaces.Discrete(2),
                "vaso": spaces.Discrete(2),
                "vent": spaces.Discrete(2),
            }
        )
        self.action_space = spaces.Dict(
            {
                "abx": spaces.Discrete(2),
                "vaso": spaces.Discrete(2),
                "vent": spaces.Discrete(2),
            }
        )

    def reset(self, seed: int = None, random_init=True, **kwargs) -> Tuple["init_state"]:
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
        if self.states[self.t + 1]["stage"] in (0, 1):
            self.truncated = True

        reward = self.Reward.count_reward(state=self.states[self.t + 1])
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


# class TestTangSepsisEnv(TestCase):
#     def __init__(self, methodName: str, Env: TangSepsisEnv) -> None:
#         super().__init__(methodName)
#         self.Env = Env
#         self.tested_name = "TangSepsisEnv"

#     def test_reset(self):
#         print(
#             f"--------------------{self.tested_name} test reset--------------------")
#         state = self.Env.reset()
#         print(state)

#     def test_step(self):
#         print(
#             f"--------------------{self.tested_name} test step--------------------")
#         action = {
#             "abx": 1,
#             "vaso": 1,
#             "vent": 1,
#         }
#         state1, reward1, terminated = self.Env.step(action)
#         state2, reward2, terminated = self.Env.step(action)
#         print(f"1st:\tstate:{state1}\treward:{reward1}")
#         print(f"2st:\tstate:{state2}\treward:{reward2}")

#     def test_export(self):
#         print(
#             f"--------------------{self.tested_name} test export--------------------")
#         batch = self.Env.export()
#         print(batch)

class TangDiscretizeActionWrapper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        # Ensure the action space is continuous
        assert isinstance(env.action_space, gym.spaces.Dict), "This wrapper only works with continuous action spaces (gym.spaces.Dict)"

        # Change the action space to be discrete
        self.action_space = gym.spaces.Dict(
            {
                "abx": spaces.Box(low=0.0, high=2.0, shape=(), dtype=np.float32),
                "vaso": spaces.Box(low=0.0, high=2.0, shape=(), dtype=np.float32),
                "vent": spaces.Box(low=0.0, high=2.0, shape=(), dtype=np.float32),
            }
        )

    def action(self, act):
        discrete_action = {k: np.round(v) for k, v in act.items()}

        # ensure discrete actions are within the correct range
        discrete_action = {k: min(max(v, self.action_space[k].low), self.action_space[k].high) for k, v in discrete_action.items()}

        return self.env.step(discrete_action)
    

def create_tang_env_fn():
    env = TangSepsisEnv()
    wrapped_env = TangDiscretizeActionWrapper(env)
    return wrapped_env