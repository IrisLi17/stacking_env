import argparse
import os

from env.robot_arm_env import ArmStack
# from fetch_stack import FetchStackEnv

import gym
from gym.wrappers import FlattenDictWrapper
from utils.monitor import Monitor
from utils.wrapper import DoneOnSuccessWrapper, SwitchGoalWrapper
import subprocess


PICK_ENTRY_POINT = {
    'BulletStack-v1': ArmStack,
}


def make_env(env_id, rank, log_dir=None, done_when_success=False, allow_switch_goal=False, flatten_dict=False,
             kwargs=None):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    :param env_id: (str) the environment ID
    :param rank: int
    :param log_dir: (str)
    :param done_when_success: (bool) terminate episode when reach a success
    :param allow_switch_goal: (bool) allows executing along a sequence of goals
    :param flatten_dict: (bool) convert dict obs to array
    :param kwargs: (dict) other parameters to create env
    :return: (Gym Environment) The environment
    """
    kwargs = kwargs.copy()
    max_episode_steps = None
    if 'max_episode_steps' in kwargs:
        max_episode_steps = kwargs['max_episode_steps']
        del kwargs['max_episode_steps']
    # gym.register(env_id, entry_point=PICK_ENTRY_POINT[env_id], max_episode_steps=max_episode_steps, kwargs=kwargs)
    env = gym.make(env_id)
    if "BulletStack" in env_id:
        from utils.wrapper import FlexibleTimeLimitWrapper
        env = FlexibleTimeLimitWrapper(env)
    if flatten_dict:
        env = FlattenDictWrapper(env, ['observation', 'achieved_goal', 'desired_goal'])
    if done_when_success:
        reward_offset = 0
        env = DoneOnSuccessWrapper(env, reward_offset)
    if allow_switch_goal:
        env = SwitchGoalWrapper(env)
    if log_dir is not None:
        info_keywords = ("is_success",)
        env = Monitor(env, os.path.join(log_dir, "%d.monitor.csv" % rank), info_keywords=info_keywords)
    return env


def get_env_kwargs(args):
    if args.env == "BulletStack-v1":
        return dict(
            robot=args.robot,
            reward_type=args.reward_type,
            n_object=args.n_object,
            n_to_stack=args.n_to_stack,
            action_dim=4 if not args.allow_rotation else 5,
        )
    else:
        raise NotImplementedError


def arg_parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', default='BulletStack-v1')
    parser.add_argument('--robot', choices=['panda'], default='panda')
    parser.add_argument('--allow_rotation', action="store_true", default=False,
                        help="Whether to enable rotation around z axis in action space.")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--reward_type', type=str, default='sparse')
    parser.add_argument('--n_object', type=int, default=2)
    parser.add_argument('--n_to_stack', type=int, nargs='+', default=1)
    parser.add_argument('--log_path', default=None, type=str)
    args = parser.parse_args()
    if isinstance(args.n_to_stack, int):
        args.n_to_stack = [args.n_to_stack]
    assert isinstance(args.n_to_stack, list)
    return args


def get_git_version():
    version_str = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    return version_str
