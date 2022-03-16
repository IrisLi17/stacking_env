import gym
from env.robot_arm_env import ArmStack


gym.register(
    "BulletStack-v1", entry_point=ArmStack, max_episode_steps=None,
    kwargs=dict(
        robot="panda", reward_type="sparse", n_object=6,
        n_to_stack=[1, 2, 3, 4, 5, 6], action_dim=4,
    )
)