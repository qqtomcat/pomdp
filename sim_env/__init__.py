"""
This module contains realizations of different simulating envs.
"""
from gymnasium.envs.registration import register

from .ahn_chemo_env import AhnChemoEnv, create_discrete_act_ahn_env
from .hassani_chemo_env import HassaniChemoEnv
from .jalalimanesh_radio_env import JalalimaneshRadioEnv
from .ngo_diabetes_env import NgoDiabetesEnv
from .noori_diabetes_env import NooriDiabetesEnv
from .patra_diabetes_env import PatraDiabetesEnv
from .tang_sepsis_env import TangSepsisEnv, create_tang_env_fn
from .vincent_radio_env import VincentRadioEnv
from .zhao_chemo_env import ZhaoChemoEnv

register(
     id="AhnChemoEnv/Continuous",
     entry_point="sim_env:AhnChemoEnv",
     max_episode_steps=3000,
)

register(
     id="AhnChemoEnv/Discrete",
     entry_point="sim_env:create_discrete_act_ahn_env",
     max_episode_steps=3000,
)

register(
     id="Cancer/HassaniChemoEnv-v0",
     entry_point="sim_env:HassaniChemoEnv",
     max_episode_steps=3000,
)

register(
     id="Cancer/JalalimaneshRadioEnv-v0",
     entry_point="sim_env:JalalimaneshRadioEnv",
     max_episode_steps=365 * 24,
)

register(
     id="Cancer/VincentRadioEnv-v0",
     entry_point="sim_env:VincentRadioEnv",
     max_episode_steps=3000,
)

register(
     id="Cancer/ZhaoChemoEnv-v0",
     entry_point="sim_env:ZhaoChemoEnv",
     max_episode_steps=10,
)

register(
     id="Diabetes/NgoDiabetesEnv-v0",
     entry_point="sim_env:NgoDiabetesEnv",
     max_episode_steps=3000,
)

register(
     id="Diabetes/NooriDiabetesEnv-v0",
     entry_point="sim_env:NooriDiabetesEnv",
     max_episode_steps=3000,
)

register(
     id="Diabetes/PatraDiabetesEnv-v0",
     entry_point="sim_env:PatraDiabetesEnv",
     max_episode_steps=3000,
)

register(
     id="Sepsis/TangSepsisEnv-v0",
     entry_point="sim_env:create_tang_env_fn",
     max_episode_steps=3000,
)

