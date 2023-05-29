from typing import Optional, Callable, Tuple
from unittest import TestCase
from collections import OrderedDict
import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from tianshou.data import Batch

from utils import get_neighbors, get_neighbors_indices
from .base import BaseReward
from .base import BaseSimulator

"""
Agent-based system used in the paper 'Simulation-based optimization of radiotherapy: Agent-based modeling and reinforcement learning'
and "Multi-objective optimization of radiotherapy: distributed Q-learning and agent-based simulation"

A few modifications have been made due to the lack of necessary mechanism descriptions in those papers.

Time unit is 1 hour

State:
    normal_cell_num: the number of normal cell
    tumour_cell_num: the number of tumour cell
    total_killed_normal_cell_num: the total number of killed normal cell
    total_killed_tumour_cell_num: the total number of killed tumour cell
    step_killed_normal_cell_num: the number of killed normal cell at this step
    step_killed_tumour_cell_num: the number of killed tumour cell at this step

Action:
    A float

Reward
    An int.
"""

"""
Agent-based system used in the paper 'Simulation-based optimization of radiotherapy: Agent-based modeling and reinforcement learning'
and "Multi-objective optimization of radiotherapy: distributed Q-learning and agent-based simulation"

A few modifications have been made due to the lack of necessary mechanism descriptions in those papers.

Time unit is 1 hour
"""


class Cell:
    def __init__(self):
        self.age = 0
        self.stage = "G1"
        self.oxygen_consumption = min(4.32, np.random.normal(0.216, 0.072))
        self.is_dead = False

    def step(self, available_oxygen: float, space_left: int) -> Tuple["is_divided", "is_dead"]:
        if self.is_dead:
            return False, self.is_dead

        able_to_divide = False
        if available_oxygen < self.oxygen_consumption:
            self.is_dead = True
        else:
            self.age += 1
            if self.stage == "G1" and self.age == 12:
                self.age = 0
                self.stage = "S"
            elif self.stage == "S" and self.age == 6:
                self.age = 0
                self.stage = "G2"
            elif self.stage == "G2" and self.age == 4:
                self.age = 0
                self.stage = "M"
            elif self.stage == "M" and self.age == 2:
                self.age = 0
                self.stage = "G1"
                able_to_divide = True

        return able_to_divide, self.is_dead

    def divide(self) -> Callable:
        return self.__class__()

    def irradiated(self, D: float = 2.5) -> bool:
        survival_prob = np.exp(-0.03 * np.random.random() * D - 1.0 * np.random.random() * D ** 2)
        if self.stage == "G2":
            radiation_received = 1.25 * np.random.random()
        elif self.stage == "G1":
            radiation_received = 0.75 * np.random.random()
        else:
            radiation_received = 1.0 * np.random.random()
        if radiation_received > survival_prob:
            self.is_dead = True
        return self.is_dead

    def bystander_effect(self, radiated_neighbor_num: int):
        bystander_survival_prob = 0.73 * np.random.random() + 0.37
        bystander_dose = min(1.0, 0.05 * radiated_neighbor_num)
        bystander_radiation_received = bystander_dose * np.random.random()
        if bystander_radiation_received > bystander_survival_prob:
            self.is_dead = True
        return self.is_dead


class NormalCell(Cell):
    def __init__(self):
        super().__init__()

    def step(self, available_oxygen: float, space_left: int) -> Tuple["is_divided", "is_dead"]:
        able_to_divide, is_dead = super().step(available_oxygen, space_left=space_left)
        is_divided = False if space_left < 1 else able_to_divide
        return is_divided, is_dead


class TumourCell(Cell):
    def __init__(self):
        super().__init__()

    def step(self, available_oxygen: float, space_left: int) -> Tuple["is_divided", "is_dead"]:
        able_to_divide, is_dead = super().step(available_oxygen, space_left=space_left)
        self.is_dead = False
        return able_to_divide, self.is_dead

    def is_dead(self):
        return self.died


class Tissue:
    def __init__(self, tissue_shape: tuple = (100, 100), microvessel_ratio: float = 0.0225) -> None:
        """
            Tissue is the toppest structure in this simulating system.
            It simulates a 2D platform, where cells live in patchs, consume oxygen, and interact with each other.
            Parameters:
                tissue_shape: tissue's shape (1 patch has a diameter of 0.02 mm)
                microvessel_ratio: the ratio of cells containing microvessel in this tissue.
                                    Those kind of cell distribute randomly in the tissue.
                seed: used to control the np.random
        """
        self.tissue_shape = tissue_shape
        self.microvessel_ratio = microvessel_ratio
        self.t = 0

        # constact patches
        patches = []
        for i in range(self.tissue_shape[0]):
            column = []
            for j in range(self.tissue_shape[1]):
                has_microvessel = np.random.rand() < self.microvessel_ratio
                patch = Patch(space=1, has_microvessel=has_microvessel)
                column.append(patch)
            patches.append(column)
        self.patches = np.array(patches)
        self.oxygen_distribution = np.zeros(self.tissue_shape)
        self.irradiated_mask = np.zeros(self.tissue_shape)

        self.killed_normal_cell_num = 0
        self.killed_tumour_cell_num = 0
        self.tumour_cell_num = 0
        self.normal_cell_num = 0

        # debug info
        self._patch_colors = np.zeros(
            (self.tissue_shape[0], self.tissue_shape[1], 3))

    def step(self, radiation_dosage: float = 0.0, debug=False):
        assert self.patches.shape == self.tissue_shape, "There is something wrong with the tissue, please reconstruct it"

        self._synchronize_oxygen_distribution()
        self._empty_patch_recover()

        # reset count
        self.tumour_cell_num = 0
        self.normal_cell_num = 0

        # irradiated and bystander effect
        killed_normal_cell_num_at_this_stage = 0
        killed_tumour_cell_num_at_this_stage = 0

        if radiation_dosage != 0.0:
            killed_normal_cell_num_r, killed_tumour_cell_num_r = self._radiation_effect(
                radiation_dosage=radiation_dosage)
            killed_normal_cell_num_at_this_stage += killed_normal_cell_num_r
            killed_tumour_cell_num_at_this_stage += killed_tumour_cell_num_r

        self._synchronize_irradiated_mask()

        if np.sum(self.irradiated_mask) > 0:
            killed_normal_cell_num_b, killed_tumour_cell_num_b = self._bystander_effect()
            killed_normal_cell_num_at_this_stage += killed_normal_cell_num_b
            killed_tumour_cell_num_at_this_stage += killed_tumour_cell_num_b

        self.killed_normal_cell_num += killed_normal_cell_num_at_this_stage
        self.killed_tumour_cell_num += killed_tumour_cell_num_at_this_stage

        # cell step and transit
        for i in range(self.tissue_shape[0]):
            for j in range(self.tissue_shape[1]):
                # step
                neighboring_oc = get_neighbors(
                    self.oxygen_distribution, (i, j))
                transit_tumour_cell = self.patches[i, j].step(
                    neighboring_oc=neighboring_oc, cell_with_microvessl_ratio=self.microvessel_ratio)

                # tumour transit mechanism
                if transit_tumour_cell is not None:
                    neighbours_indices = get_neighbors_indices(
                        self.oxygen_distribution, (i, j))
                    transit_indices = neighbours_indices[np.random.randint(
                        0, len(neighbours_indices))]
                    self.patches[transit_indices[0], transit_indices[1]].add_tumour_cell(
                        transit_tumour_cell)

        # count tumour cell num
        for i in range(self.tissue_shape[0]):
            for j in range(self.tissue_shape[1]):
                self.tumour_cell_num += self.patches[i, j].tumour_cell_number()
                self.normal_cell_num += self.patches[i, j].normal_cell_number()
                self._patch_colors[i, j] = self.patches[i, j].get_color()

        self.t += 1

        if debug == True:
            print(
                f"Tumour cell number at time {self.t}: {self.tumour_cell_num}")
            cells_img = cv2.resize(self._patch_colors, (1000, 1000))
            cv2.imwrite(f"cell_{self.t}.jpg", cells_img)

        return OrderedDict({"normal_cell_num": int(self.normal_cell_num),
                "tumour_cell_num": int(self.tumour_cell_num),
                "total_killed_normal_cell_num": int(self.killed_normal_cell_num),
                "total_killed_tumour_cell_num": int(self.killed_tumour_cell_num),
                "step_killed_normal_cell_num": int(killed_normal_cell_num_at_this_stage),
                "step_killed_tumour_cell_num": int(killed_tumour_cell_num_at_this_stage)})

    def _synchronize_oxygen_distribution(self):
        for i in range(self.tissue_shape[0]):
            for j in range(self.tissue_shape[1]):
                self.oxygen_distribution[i, j] = self.patches[i,
                                                              j].get_oxygen_concentration()

    def _synchronize_irradiated_mask(self):
        for i in range(self.tissue_shape[0]):
            for j in range(self.tissue_shape[1]):
                self.irradiated_mask[i, j] = 1 if self.patches[i,
                                                               j].is_radiated() else 0

    def _empty_patch_recover(self):
        for i in range(self.tissue_shape[0]):
            for j in range(self.tissue_shape[1]):
                if self.patches[i, j].is_empty():
                    neighbours_indices = get_neighbors_indices(
                        self.oxygen_distribution, (i, j))
                    transit_indices = neighbours_indices[np.random.randint(
                        0, len(neighbours_indices))]
                    if self.patches[transit_indices[0], transit_indices[1]].is_empty():
                        pass
                    else:
                        self.patches[i, j].recover()

    def _radiation_effect(self, radiation_dosage):
        killed_normal_cell_num_r, killed_tumour_cell_num_r = 0, 0
        for i in range(self.tissue_shape[0]):
            for j in range(self.tissue_shape[1]):
                killed_normal_cell_num, killed_tumour_cell_num = self.patches[i, j].irradiated(
                    radiation_dosage)
                killed_normal_cell_num_r += killed_normal_cell_num
                killed_tumour_cell_num_r += killed_tumour_cell_num
        return killed_normal_cell_num_r, killed_tumour_cell_num_r

    def _bystander_effect(self):
        killed_normal_cell_num_b, killed_tumour_cell_num_b = 0, 0
        for i in range(self.tissue_shape[0]):
            for j in range(self.tissue_shape[1]):
                radiated_neighbor_num = sum(get_neighbors(
                    self.irradiated_mask, (i, j), radius=2))

                killed_normal_cell_num, killed_tumour_cell_num = self.patches[i, j].bystander_effect(
                    radiated_neighbor_num)
                killed_normal_cell_num_b += killed_normal_cell_num
                killed_tumour_cell_num_b += killed_tumour_cell_num
        return killed_normal_cell_num_b, killed_tumour_cell_num_b

    def add_tumour_cell(self, patch_indices: tuple = (50, 50)):
        self.patches[patch_indices[0], patch_indices[1]].add_tumour_cell()


class Patch:
    def __init__(self, space: int = 1, has_microvessel: bool = False) -> None:
        """
            Patch is the tissue's basic unit.
            In the refered paper, each patch have space for only one cell (However, tumour cells do not care about this).
            Parameters:
                space: the capacity of the patch
                has_microvessel: does this patch has vessel (the only oxygen enter method)
        """
        self.space = space
        self.normal_cells = [NormalCell()]
        self.tumour_cells = []
        self.oxygen_concentration = 1
        self.has_microvessel = has_microvessel
        self.recover_delay = 24
        # two tags for cell transfer
        self.abnormal = False
        self.empty = False
        # radiated flag
        self.radiated = False
        # visual mode
        self._color = (0, 0, 0)

    def get_oxygen_concentration(self):
        return self.oxygen_concentration

    def add_tumour_cell(self, cell=None):
        if cell is None:
            self.tumour_cells.append(TumourCell())
        else:
            self.tumour_cells.append(cell)

    def add_normal_cell(self, cell=None):
        if cell is None:
            self.normal_cells.append(NormalCell())
        else:
            self.normal_cells.append(cell)

    def _cells_step(self, cells: list):
        cell_num = len(self.normal_cells) + len(self.tumour_cells)
        available_oxygen_for_each_cell = self.oxygen_concentration / max(cell_num, 1)
        space_left = self.space - cell_num
        cells_next = []
        for cell in cells:
            is_divided, is_dead = cell.step(
                available_oxygen_for_each_cell, space_left)

            if not is_dead:
                cells_next.append(cell)

            if is_divided:
                cells_next.append(cell.divide())

        return cells_next

    def step(self, neighboring_oc: float, cell_with_microvessl_ratio: float) -> Optional[Cell]:
        """
            neighboring_oc: neighboring oxygen concentration
            cell_with_microvessl_ratio: ratio of cells with microvessel in this tissue
        """
        # cells step
        self.normal_cells = self._cells_step(self.normal_cells)
        self.tumour_cells = self._cells_step(self.tumour_cells)
        cell_num = len(self.normal_cells) + len(self.tumour_cells)

        # oxygen diffusion
        vessel_oxygen = 1.5 * 0.216 / cell_with_microvessl_ratio if self.has_microvessel else 0
        transit_oxygen = 0.42 * np.mean(neighboring_oc)
        self.oxygen_concentration = (
                                            1 - 0.42) * self.oxygen_concentration + transit_oxygen - 0.216 * cell_num + vessel_oxygen
        self.oxygen_concentration = max(0, self.oxygen_concentration)

        if cell_num == 0:
            self.empty = True
            self.abnormal = False
            self._color = [0, 0, 0]
        elif len(self.tumour_cells) > 0:
            self.empty = False
            self.abnormal = True
            self._color = [0, 0, 255]
        else:
            self.empty = False
            self.abnormal = False
            self._color = [255, 255, 255]

        if self.radiated:
            self._color[1] = 255

        # transit tumour
        if len(self.tumour_cells) > 1:
            transit_tumour_cell = self.tumour_cells.pop(-1)
        else:
            transit_tumour_cell = None
        return transit_tumour_cell

    def irradiated(self, D) -> Tuple["killed_normal_cell_num", "killed_tumour_cell_num"]:
        killed_normal_cell_num = 0
        killed_tumour_cell_num = 0

        new_normal_cells = []
        new_tumour_cells = []

        for normal_cell in self.normal_cells:
            is_dead = normal_cell.irradiated(D)
            if is_dead:
                killed_normal_cell_num += 1
            else:
                new_normal_cells.append(normal_cell)
        self.normal_cells = new_normal_cells

        for tumour_cell in self.tumour_cells:
            is_dead = tumour_cell.irradiated(D)
            if is_dead:
                killed_tumour_cell_num += 1
            else:
                new_tumour_cells.append(tumour_cell)
        self.tumour_cells = new_tumour_cells

        if len(self.tumour_cells) == 0 and len(self.normal_cells) == 0:
            self.radiated = False
        else:
            self.radiated = True

        return killed_normal_cell_num, killed_tumour_cell_num

    def bystander_effect(self, radiated_neighbor_num) -> Tuple["killed_normal_cell_num", "killed_tumour_cell_num"]:
        killed_normal_cell_num = 0
        killed_tumour_cell_num = 0

        new_normal_cells = []
        new_tumour_cells = []

        for normal_cell in self.normal_cells:
            is_dead = normal_cell.bystander_effect(radiated_neighbor_num)
            if is_dead:
                killed_normal_cell_num += 1
            else:
                new_normal_cells.append(normal_cell)
        self.normal_cells = new_normal_cells

        for tumour_cell in self.tumour_cells:
            is_dead = tumour_cell.bystander_effect(radiated_neighbor_num)
            if is_dead:
                killed_tumour_cell_num += 1
            else:
                new_tumour_cells.append(tumour_cell)
        self.tumour_cells = new_tumour_cells

        if len(self.tumour_cells) == 0 and len(self.normal_cells) == 0:
            self.radiated = False
        else:
            self.radiated = True

        return killed_normal_cell_num, killed_tumour_cell_num

    def get_color(self):
        return self._color

    def is_abnormal(self):
        return self.abnormal

    def is_empty(self):
        return self.empty

    def tumour_cell_number(self):
        return len(self.tumour_cells)

    def normal_cell_number(self):
        return len(self.normal_cells)

    def is_radiated(self):
        return self.radiated

    def recover(self):
        self.recover_delay -= 1
        if self.recover_delay < 1:
            self.add_normal_cell()
            self.recover_delay = 24


class JalalimaneshABS(BaseSimulator):
    def __init__(self):
        super().__init__()
        self.tissue = Tissue()
        """
        State:
            normal_cell_num: the number of normal cell
            tumour_cell_num: the number of tumour cell
            total_killed_normal_cell_num: the total number of killed normal cell
            total_killed_tumour_cell_num: the total number of killed tumour cell
            step_killed_normal_cell_num: the number of killed normal cell at this step
            step_killed_tumour_cell_num: the number of killed tumour cell at this step
        """

    def activate(self) -> Tuple["init_state"]:
        for tumour_cell_indice in np.random.randint(0, 100, size=(10, 2)):
            self.tissue.add_tumour_cell(tumour_cell_indice)
        for i in range(300):
            self.tissue.step()
        state = self.tissue.step()
        return state

    def update(self, action: float, debug=False) -> Tuple["next_state"]:
        state = self.tissue.step(action, debug=debug)
        return state


class TestJalalimaneshABS(TestCase):
    def __init__(self, methodName: str, Simulator: JalalimaneshABS) -> None:
        super().__init__(methodName)
        self.Simulator = Simulator
        self.tested_name = "JalalimaneshABS"

    def test_activate(self):
        print(
            f"--------------------{self.tested_name} test activate--------------------")
        state = self.Simulator.activate()
        print(state)

    def test_update(self):
        print(
            f"--------------------{self.tested_name} test update--------------------")
        action = np.random.random()
        print(self.Simulator.update(action=action, debug=True))


class JalalimaneshReward(BaseReward):
    def __init__(self):
        super().__init__()

    def count_reward(self, state) -> float:
        total_killed_normal_cell_num = state["total_killed_normal_cell_num"]
        total_killed_tumour_cell_num = state["total_killed_tumour_cell_num"]
        step_killed_normal_cell_num = state["step_killed_normal_cell_num"]
        reward_invasive = 3 * total_killed_tumour_cell_num - 1 * total_killed_normal_cell_num - 0.1 * step_killed_normal_cell_num
        reward_moderate = 2 * total_killed_tumour_cell_num - 1 * total_killed_normal_cell_num - 0.3 * step_killed_normal_cell_num
        reward_conservative = 1 * total_killed_tumour_cell_num - 2 * total_killed_normal_cell_num - 0.5 * step_killed_normal_cell_num
        return reward_moderate


class TestJalalimaneshReward(TestCase):
    def __init__(self, methodName: str, Reward: JalalimaneshReward) -> None:
        super().__init__(methodName)
        self.Reward = Reward
        self.tested_name = "JalalimaneshReward"

    def test_count_reward(self):
        print(f"--------------------{self.tested_name} test count_reward--------------------")
        state = {'normal_cell_num': 5296, 'tumour_cell_num': 17932, 'total_killed_normal_cell_num': 1978,
                 'total_killed_tumour_cell_num': 7076, 'step_killed_normal_cell_num': 1978,
                 'step_killed_tumour_cell_num': 7076}
        reward = self.Reward.count_reward(state)
        print(reward)


class JalalimaneshRadioEnv(gym.Env):
    def __init__(self, max_t: int = 365 * 24):  # I hope the treatment ends in one year
        super().__init__()
        self.Simulator = JalalimaneshABS()
        self.Reward = JalalimaneshReward()
        self.max_t = max_t
        self.observation_space = spaces.Dict(
            {
                'normal_cell_num': spaces.Discrete(100000),
                'tumour_cell_num': spaces.Discrete(100000),
                'total_killed_normal_cell_num': spaces.Discrete(100000),
                'total_killed_tumour_cell_num': spaces.Discrete(100000),
                'step_killed_normal_cell_num': spaces.Discrete(100000),
                'step_killed_tumour_cell_num': spaces.Discrete(100000),
            }
        )
        self.action_space = spaces.Box(
            low=0.0, high=5.0, shape=(), dtype=np.float32)
        

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

    def step(self, action: float, debug=False) -> Tuple["next_state", "reward", "terminated"]:
        """
        step the env with given action, and get next state and reward of this action
        """
        if self.terminated == True or self.truncated==True:
            print("This treat is end, please call reset or export")
            return None, None, self.terminated, self.truncated, {}

        self.actions[self.t] = action
        state_next = self.Simulator.update(
            action=self.actions[self.t], debug=True)
        self.states[self.t + 1] = state_next

        # check whether the treatment end
        if self.t + 1 == self.max_t - 1:
            self.terminated = True
        if self.states[self.t + 1]["tumour_cell_num"] == 0:
            self.truncated = True

        reward = self.Reward.count_reward(
            state=self.states[self.t])
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


class TestJalalimaneshRadioEnv(TestCase):
    def __init__(self, methodName: str, Env: JalalimaneshRadioEnv) -> None:
        super().__init__(methodName)
        self.Env = Env
        self.tested_name = "JalalimaneshRadioEnv"

    def test_reset(self):
        print(
            f"--------------------{self.tested_name} test reset--------------------")
        init_state, info = self.Env.reset(seed=1)
        print(init_state in self.Env.observation_space)
        print(init_state)
        print(self.Env.observation_space.sample())
        # assert init_state in self.Env.observation_space, "invalid initial observation"

    def test_step(self):
        print(
            f"--------------------{self.tested_name} test step--------------------")
        action = np.random.random()
        state1, reward1, terminated = self.Env.step(action, debug=True)
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
