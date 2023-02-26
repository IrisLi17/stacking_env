import gym
from bullet_envs.env.robot_arm_env import ArmStack, ArmPickAndPlace
from bullet_envs.env.primitive_env import DrawerObjEnv, DrawerObjEnvState
from bullet_envs.env.pixel_stacking import PixelStack
from bullet_envs.env.primitive_stacking import ArmStack as StateStack
import numpy as np


gym.register(
    "BulletStack-v1", entry_point=ArmStack, max_episode_steps=None,
    kwargs=dict(
        robot="panda", reward_type="sparse", n_object=6,
        n_to_stack=[1, 2, 3, 4, 5, 6], action_dim=4,
    )
)

gym.register(
    "BulletPickAndPlace-v1", entry_point=ArmPickAndPlace, max_episode_steps=100,
    kwargs=dict(
        robot="panda", reward_type="sparse", n_object=6, action_dim=4
    )
)

gym.register(
    "BulletDrawer-v1", entry_point=DrawerObjEnv, max_episode_steps=20
)

gym.register(
    "BulletDrawerState-v1", entry_point=DrawerObjEnvState, max_episode_steps=20
)

gym.register(
    "BulletPixelStack-v1", entry_point=PixelStack, max_episode_steps=30,
)

gym.register(
    "BulletStack-v2", entry_point=StateStack, max_episode_steps=None,
    kwargs=dict(
        n_object=6, reward_type="sparse", action_dim=7, generate_data=True, primitive=True,
        n_to_stack=np.array([[1, 2, 3]]), name="allow_rotation"
    ),
)
