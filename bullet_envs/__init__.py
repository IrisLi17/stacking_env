import gym
from bullet_envs.env.robot_arm_env import ArmStack, ArmPickAndPlace
from bullet_envs.env.primitive_env import DrawerObjEnv


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