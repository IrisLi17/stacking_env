# from bullet_envs.utils.make_env import arg_parse, get_env_kwargs, make_env
from bullet_envs.utils.make_vec_env import make_vec_env
import matplotlib.pyplot as plt
import numpy as np
import torch


def main():
    args = arg_parse()
    env_kwargs = get_env_kwargs(args)
    env = make_env(args.env, 0, args.log_path, done_when_success=True, flatten_dict=True, kwargs=env_kwargs)
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        img = env.render(mode="rgb_array")
        plt.imshow(img)
        plt.pause(0.1)


def main2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_vec_env("BulletDrawer-v1", 16, device)
    obs = env.reset()
    done = np.array([False] * env.num_envs)
    while not done[0]:
        action = torch.cat([
            torch.randint(0, 4, size=(env.num_envs, 1)), 
            torch.rand(size=(env.num_envs, 4)) * 2 - 1
        ], dim=-1).float().to(device)
        obs, reward, done, info = env.step(action)
        print(reward)

if __name__ == "__main__":
    main2()
