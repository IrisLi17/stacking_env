from utils.make_env import arg_parse, get_env_kwargs, make_env
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    main()
