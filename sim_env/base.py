from abc import abstractmethod
from typing import Tuple
from typing import Union


class BaseSimulator:
    def __init__(self):
        pass

    @abstractmethod
    def activate(self, random_init: bool) -> "init_state":
        """
        actiuvate the simulator and return the init state
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, action: Union[dict, float], state: dict, integral_num: int = 10) -> "next_state":
        """
        update the model from certain state with given action
        action is the agent's choosen action
        state is the t-1 state
        integral_num is the number of integral in a unit time (i.e. 1/dt)
        """
        raise NotImplementedError


class BaseReward:
    def __init__(self):
        pass

    @abstractmethod
    def count_reward(self, *args, **kwargs) -> "reward":
        """
        actiuvate the simulator and return the init state
        """
        raise NotImplementedError
