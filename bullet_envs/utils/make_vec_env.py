from bullet_envs.env.primitive_env import DrawerObjEnv
from bullet_envs.utils.monitor import Monitor
from bullet_envs.utils.wrapper import DoneOnSuccessWrapper, MVPVecPyTorch, VecPyTorch
from bullet_envs.vec_env.subproc_vec_env import SubprocVecEnv
import gym
import os


def make_env(env_id, rank, log_dir=None, info_keywords=("is_success",), kwargs={}):
    env = gym.make(env_id, **kwargs)
    env = DoneOnSuccessWrapper(env)
    if log_dir is not None:
        env = Monitor(env, os.path.join(log_dir, "%d.monitor.csv" % rank), info_keywords=info_keywords)
    return env

def make_vec_env(env_id, num_workers, device, reset_when_done=True, **kwargs):
    if "use_raw_img" in kwargs:
        use_raw_img = kwargs.pop("use_raw_img")
    else:
        use_raw_img = False
    def make_env_thunk(i):
        return lambda: make_env(env_id, i, **kwargs)
    venv = SubprocVecEnv([make_env_thunk(i) for i in range(num_workers)], 
                          reset_when_done=reset_when_done)
    if env_id == "BulletDrawerState-v1":
        mvp_venv = VecPyTorch(venv, device)
    else:
        mvp_venv = MVPVecPyTorch(venv, device, use_raw_img=use_raw_img)
    return mvp_venv
